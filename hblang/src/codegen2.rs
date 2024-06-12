#![allow(dead_code)]

use std::{collections::hash_map, rc::Rc};

use crate::{
    ident::Ident,
    parser::{self, FileId},
    HashMap,
};

type Offset = u32;
type Size = u32;

mod stack {
    use std::num::NonZeroU32;

    use super::{Offset, Size};

    pub struct Id(NonZeroU32);

    impl Id {
        fn index(&self) -> usize {
            self.0.get() as usize - 1
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                unreachable!("stack id leaked: {:?}", self.0);
            }
        }
    }

    #[derive(PartialEq)]
    struct Meta {
        size:   Size,
        offset: Offset,
        rc:     u32,
    }

    #[derive(Default)]
    pub struct Alloc {
        height:     Size,
        max_height: Size,
        meta:       Vec<Meta>,
    }

    impl Alloc {
        pub fn alloc(&mut self, size: Size) -> Id {
            self.meta.push(Meta {
                size,
                offset: 0,
                rc: 1,
            });

            self.height += size;
            self.max_height = self.max_height.max(self.height);

            Id(unsafe { NonZeroU32::new_unchecked(self.meta.len() as u32) })
        }

        pub fn free(&mut self, id: Id) {
            let meta = &mut self.meta[id.index()];
            std::mem::forget(id);
            meta.rc -= 1;
            if meta.rc != 0 {
                return;
            }
            meta.offset = self.height;
            self.height -= meta.size;
        }

        pub fn dup_id(&mut self, id: &Id) -> Id {
            self.meta[id.index()].rc += 1;
            Id(id.0)
        }

        pub fn finalize_leaked(&mut self) {
            for meta in self.meta.iter_mut().filter(|m| m.rc > u32::MAX / 2) {
                meta.offset = self.height;
                self.height -= meta.size;
            }
        }

        pub fn clear(&mut self) {
            self.height = 0;
            self.max_height = 0;
            self.meta.clear();
        }

        pub fn final_offset(&self, id: u32, extra_offset: Offset) -> Offset {
            (self.max_height - self.meta[id as usize].offset) + extra_offset
        }
    }
}

mod reg {
    pub const STACK_PTR: Reg = 254;
    pub const ZERO: Reg = 0;
    pub const RET_ADDR: Reg = 31;

    type Reg = u8;

    #[derive(Default)]
    pub struct Id(Reg, bool);

    impl Id {
        pub fn reg(self) -> Reg {
            self.0
        }

        pub fn as_ref(&self) -> Self {
            Self(self.0, false)
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            if !std::thread::panicking() && self.1 {
                unreachable!("reg id leaked: {:?}", self.0);
            }
        }
    }

    #[derive(Default, PartialEq, Eq)]
    pub struct Alloc {
        free:     Vec<Reg>,
        max_used: Reg,
    }

    impl Alloc {
        fn init(&mut self) {
            self.free.clear();
            self.free.extend((32..=253).rev());
            self.max_used = RET_ADDR;
        }

        fn allocate(&mut self) -> Id {
            let reg = self.free.pop().expect("TODO: we need to spill");
            self.max_used = self.max_used.max(reg);
            Id(reg, true)
        }

        fn free(&mut self, reg: Id) {
            assert!(reg.1);
            self.free.push(reg.0);
            std::mem::forget(reg);
        }

        fn pushed_size(&self) -> usize {
            ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
        }
    }
}

mod ty {
    use crate::{
        ident,
        lexer::TokenKind,
        parser::{self, Expr},
    };

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct Id(pub u32);

    impl Id {
        pub fn is_signed(self) -> bool {
            (I8..=INT).contains(&self.0)
        }

        pub fn is_unsigned(self) -> bool {
            (U8..=UINT).contains(&self.0)
        }

        fn strip_pointer(self) -> Self {
            match self.expand() {
                Kind::Ptr(_) => Id(INT),
                _ => self,
            }
        }

        pub fn is_pointer(self) -> bool {
            matches!(Kind::from_ty(self), Kind::Ptr(_))
        }

        pub fn try_upcast(self, ob: Self) -> Option<Self> {
            let (oa, ob) = (Self(self.0.min(ob.0)), Self(self.0.max(ob.0)));
            let (a, b) = (oa.strip_pointer(), ob.strip_pointer());
            Some(match () {
                _ if oa == ob => oa,
                _ if a.is_signed() && b.is_signed() || a.is_unsigned() && b.is_unsigned() => ob,
                _ if a.is_unsigned() && b.is_signed() && a.0 - U8 < b.0 - I8 => ob,
                _ => return None,
            })
        }

        pub fn expand(self) -> Kind {
            Kind::from_ty(self)
        }
    }

    const fn array_to_lower_case<const N: usize>(array: [u8; N]) -> [u8; N] {
        let mut result = [0; N];
        let mut i = 0;
        while i < N {
            result[i] = array[i].to_ascii_lowercase();
            i += 1;
        }
        result
    }
    // const string to lower case

    macro_rules! builtin_type {
        ($($name:ident;)*) => {
            $(pub const $name: Builtin = ${index(0)};)*

            mod __lc_names {
                use super::*;
                $(pub const $name: &[u8] = &array_to_lower_case(unsafe {
                    *(stringify!($name).as_ptr() as *const [u8; stringify!($name).len()]) });)*
            }

            pub fn from_str(name: &str) -> Option<Builtin> {
                match name.as_bytes() {
                    $(__lc_names::$name => Some($name),)*
                    _ => None,
                }
            }

            pub fn to_str(ty: Builtin) -> &'static str {
                match ty {
                    $(${index(0)} => unsafe { std::str::from_utf8_unchecked(__lc_names::$name) },)*
                    v => unreachable!("invalid type: {}", v),
                }
            }
        };
    }

    builtin_type! {
        UNDECLARED;
        NEVER;
        VOID;
        TYPE;
        BOOL;
        U8;
        U16;
        U32;
        UINT;
        I8;
        I16;
        I32;
        INT;
    }

    macro_rules! type_kind {
        ($(#[$meta:meta])* $vis:vis enum $name:ident {$( $variant:ident, )*}) => {
            $(#[$meta])*
            $vis enum $name {
                $($variant($variant),)*
            }

            impl $name {
                const FLAG_BITS: u32 = (${count($variant)} as u32).next_power_of_two().ilog2();
                const FLAG_OFFSET: u32 = std::mem::size_of::<Id>() as u32 * 8 - Self::FLAG_BITS;
                const INDEX_MASK: u32 = (1 << (32 - Self::FLAG_BITS)) - 1;

                $vis const fn from_ty(ty: Id) -> Self {
                    let (flag, index) = (ty.0 >> Self::FLAG_OFFSET, ty.0 & Self::INDEX_MASK);
                    match flag {
                        $(${index(0)} => Self::$variant(index),)*
                        _ => unreachable!(),
                    }
                }

                $vis const fn compress(self) -> Id {
                    let (index, flag) = match self {
                        $(Self::$variant(index) => (index, ${index(0)}),)*
                    };
                   Id((flag << Self::FLAG_OFFSET) | index)
                }
            }
        };
    }

    type_kind! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Kind {
            Builtin,
            Struct,
            Ptr,
            Func,
            Global,
            Module,
        }
    }

    impl Default for Kind {
        fn default() -> Self {
            Self::Builtin(UNDECLARED)
        }
    }

    pub struct Display<'a> {
        tys:   &'a super::Types,
        files: &'a [parser::Ast],
        ty:    Id,
    }

    impl<'a> Display<'a> {
        pub fn new(tys: &'a super::Types, files: &'a [parser::Ast], ty: Id) -> Self {
            Self { tys, files, ty }
        }

        fn rety(&self, ty: Id) -> Self {
            Self::new(self.tys, self.files, ty)
        }
    }

    impl<'a> std::fmt::Display for Display<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use Kind as TK;
            match TK::from_ty(self.ty) {
                TK::Module(idx) => write!(f, "module{}", idx),
                TK::Builtin(ty) => write!(f, "{}", to_str(ty)),
                TK::Ptr(ty) => {
                    write!(f, "^{}", self.rety(self.tys.pointers[ty as usize].base))
                }
                _ if let Some((key, _)) = self
                    .tys
                    .symbols
                    .iter()
                    .find(|(sym, &ty)| sym.file != u32::MAX && ty == self.ty.0)
                    && let Some(name) = self.files[key.file as usize].exprs().iter().find_map(
                        |expr| match expr {
                            Expr::BinOp {
                                left: &Expr::Ident { name, id, .. },
                                op: TokenKind::Decl,
                                ..
                            } if id == key.ident => Some(name),
                            _ => None,
                        },
                    ) =>
                {
                    write!(f, "{name}")
                }
                TK::Struct(idx) => {
                    let record = &self.tys.structs[idx as usize];
                    write!(f, "{{")?;
                    for (i, &super::Field { name, ty }) in record.fields.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        let name = &self.tys.names[ident::range(name)];
                        write!(f, "{name}: {}", self.rety(ty))?;
                    }
                    write!(f, "}}")
                }
                TK::Func(idx) => write!(f, "fn{idx}"),
                TK::Global(idx) => write!(f, "global{idx}"),
            }
        }
    }
}

struct Reloc {
    code_offset: Offset,
    /// code_offset - sub_offset = instr_offset
    sub_offset:  u8,
    width:       u8,
}

impl Reloc {
    fn new(code_offset: u32, sub_offset: u8, width: u8) -> Self {
        Self {
            code_offset,
            sub_offset,
            width,
        }
    }

    fn apply_stack_offset(&self, code: &mut [u8], stack: &stack::Alloc) {
        let bytes = &code[self.code_offset as usize..][..self.width as usize];
        let id = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
        let extra_offset = u32::from_ne_bytes(bytes[4..].try_into().unwrap());
        self.write_offset(code, stack.final_offset(id, extra_offset) as i64);
    }

    fn apply_jump(&self, code: &mut [u8], to: u32) {
        let offset = to as i64 - (self.code_offset as i64 - self.sub_offset as i64);
        self.write_offset(code, offset);
    }

    fn write_offset(&self, code: &mut [u8], offset: i64) {
        let bytes = offset.to_ne_bytes();
        code[self.code_offset as usize..][..self.width as usize]
            .copy_from_slice(&bytes[..self.width as usize]);
    }
}

struct Value {
    ty:  ty::Id,
    loc: Loc,
}

struct Loc {
    reg:     reg::Id,
    derefed: bool,
    stack:   Option<stack::Id>,
    offset:  Offset,
}

struct Loop {
    var_count:  u32,
    offset:     u32,
    reloc_base: u32,
}

struct Variable {
    id:    Ident,
    value: Value,
}

#[derive(Default)]
struct ItemCtx {
    id:      ty::Kind,
    ret:     ty::Id,
    ret_reg: reg::Id,

    stack: stack::Alloc,
    regs:  reg::Alloc,

    stack_relocs: Vec<Reloc>,
    ret_relocs:   Vec<Reloc>,
    loop_relocs:  Vec<Reloc>,
    loops:        Vec<Loop>,
    variables:    Vec<Variable>,
}

impl ItemCtx {
    pub fn dup_loc(&mut self, loc: &Loc) -> Loc {
        Loc {
            reg:     loc.reg.as_ref(),
            derefed: loc.derefed,
            stack:   loc.stack.as_ref().map(|s| self.stack.dup_id(s)),
            offset:  loc.offset,
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
struct SymKey {
    file:  FileId,
    ident: Ident,
}

struct Func {
    offset: Offset,
    args:   Rc<[ty::Id]>,
    ret:    ty::Id,
}

struct Global {
    code:   Offset,
    offset: Offset,
    dep:    ty::Global,
    ty:     ty::Id,
}

struct Field {
    name: Ident,
    ty:   ty::Id,
}

struct Struct {
    fields: Rc<[Field]>,
    ast:    parser::ExprRef,
}

struct Ptr {
    base: ty::Id,
}

#[derive(Default)]
struct Types {
    symbols: HashMap<SymKey, u32>,
    names:   String,

    funcs:    Vec<Func>,
    globals:  Vec<Global>,
    structs:  Vec<Struct>,
    pointers: Vec<Ptr>,
}

impl Types {
    pub fn make_ptr(&mut self, base: ty::Id) -> ty::Ptr {
        let id = SymKey {
            file:  u32::MAX,
            ident: base.0,
        };

        match self.symbols.entry(id) {
            hash_map::Entry::Occupied(occ) => *occ.get(),
            hash_map::Entry::Vacant(vac) => {
                self.pointers.push(Ptr { base });
                *vac.insert(self.pointers.len() as u32 - 1)
            }
        }
    }
}

#[derive(Default)]
struct Output {
    code: Vec<u8>,
    func_relocs: Vec<(ty::Func, Reloc)>,
    global_relocs: Vec<(ty::Global, Reloc)>,
    tasks: Vec<ty::Func>,
}

#[derive(Default)]
pub struct Codegen {
    pub files: Vec<parser::Ast>,
    tys:       Types,
    ci:        ItemCtx,
    output:    Output,
}

impl Codegen {
    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, &self.files, ty)
    }
}
