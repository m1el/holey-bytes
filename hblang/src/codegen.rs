use std::{ops::Range, rc::Rc, usize};

use crate::{
    ident::{self, Ident},
    instrs::{self, *},
    lexer::TokenKind,
    log,
    parser::{self, find_symbol, idfl, Expr, ExprRef, FileId, Pos},
    HashMap,
};

use self::reg::{RET_ADDR, STACK_PTR, ZERO};

type Offset = u32;
type Size = u32;

mod stack {
    use std::num::NonZeroU32;

    use super::{Offset, Size};

    #[derive(Debug, PartialEq, Eq)]
    pub struct Id(NonZeroU32);

    impl Id {
        fn index(&self) -> usize {
            (self.0.get() as usize - 1) & !(1 << 31)
        }

        pub fn repr(&self) -> u32 {
            self.0.get()
        }

        pub fn as_ref(&self) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(self.0.get() | 1 << 31) })
        }

        pub fn is_ref(&self) -> bool {
            self.0.get() & (1 << 31) != 0
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            if !std::thread::panicking() && !self.is_ref() {
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
        height: Size,
        pub max_height: Size,
        meta: Vec<Meta>,
    }

    impl Alloc {
        pub fn allocate(&mut self, size: Size) -> Id {
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
            if id.is_ref() {
                return;
            }
            let meta = &mut self.meta[id.index()];
            std::mem::forget(id);
            meta.rc -= 1;
            if meta.rc != 0 {
                return;
            }
            meta.offset = self.height;
            self.height -= meta.size;
        }

        pub fn finalize_leaked(&mut self) {
            for meta in self.meta.iter_mut().filter(|m| m.rc > 0) {
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
            debug_assert_ne!(id, 0);
            (self.max_height - self.meta[(id as usize - 1) & !(1 << 31)].offset) + extra_offset
        }
    }
}

mod reg {
    pub const STACK_PTR: Reg = 254;
    pub const ZERO: Reg = 0;
    pub const RET: Reg = 1;
    pub const RET_ADDR: Reg = 31;

    type Reg = u8;

    #[derive(Default, Debug, PartialEq, Eq)]
    pub struct Id(Reg, bool);

    impl Id {
        pub const RET: Self = Id(RET, false);

        pub fn get(&self) -> Reg {
            self.0
        }

        pub fn as_ref(&self) -> Self {
            Self(self.0, false)
        }

        pub fn is_ref(&self) -> bool {
            !self.1
        }
    }

    impl From<u8> for Id {
        fn from(value: u8) -> Self {
            Self(value, false)
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
        pub fn init(&mut self) {
            self.free.clear();
            self.free.extend((32..=253).rev());
            self.max_used = RET_ADDR;
        }

        pub fn allocate(&mut self) -> Id {
            let reg = self.free.pop().expect("TODO: we need to spill");
            self.max_used = self.max_used.max(reg);
            Id(reg, true)
        }

        pub fn free(&mut self, reg: Id) {
            if reg.1 {
                self.free.push(reg.0);
                std::mem::forget(reg);
            }
        }

        pub fn pushed_size(&self) -> usize {
            ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
        }
    }
}

pub mod ty {
    use std::{num::NonZeroU32, ops::Range};

    use crate::{
        lexer::TokenKind,
        parser::{self, Expr},
    };

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;
    pub type Param = u32;

    #[derive(Clone, Copy)]
    pub struct Tuple(pub u32);

    impl Tuple {
        const LEN_BITS: u32 = 5;
        const MAX_LEN: usize = 1 << Self::LEN_BITS;
        const LEN_MASK: usize = Self::MAX_LEN - 1;

        pub fn new(pos: usize, len: usize) -> Option<Self> {
            if len >= Self::MAX_LEN {
                return None;
            }

            Some(Self((pos << Self::LEN_BITS | len) as u32))
        }

        pub fn view(self, slice: &[Id]) -> &[Id] {
            &slice[self.0 as usize >> Self::LEN_BITS..][..self.len()]
        }

        pub fn range(self) -> Range<usize> {
            let start = self.0 as usize >> Self::LEN_BITS;
            start..start + self.len()
        }

        pub fn len(self) -> usize {
            self.0 as usize & Self::LEN_MASK
        }

        pub fn is_empty(self) -> bool {
            self.0 == 0
        }

        pub fn empty() -> Self {
            Self(0)
        }

        pub fn repr(&self) -> u32 {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct Id(NonZeroU32);

    impl Default for Id {
        fn default() -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(UNDECLARED) })
        }
    }

    impl Id {
        pub const fn from_bt(bt: u32) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(bt) })
        }

        pub fn is_signed(self) -> bool {
            (I8..=INT).contains(&self.repr())
        }

        pub fn is_unsigned(self) -> bool {
            (U8..=UINT).contains(&self.repr())
        }

        pub fn strip_pointer(self) -> Self {
            match self.expand() {
                Kind::Ptr(_) => Kind::Builtin(UINT).compress(),
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
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 < b.repr() - I8 => ob,
                _ => return None,
            })
        }

        pub fn expand(self) -> Kind {
            Kind::from_ty(self)
        }

        pub const fn repr(self) -> u32 {
            self.0.get()
        }
    }

    impl From<[u8; 8]> for Id {
        fn from(id: [u8; 8]) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(u64::from_ne_bytes(id) as _) })
        }
    }

    impl From<u32> for Id {
        fn from(id: u32) -> Self {
            Kind::Builtin(id).compress()
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
            $(pub const $name: Builtin = ${index(0)} + 1;)*

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
                    $($name => unsafe { std::str::from_utf8_unchecked(__lc_names::$name) },)*
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
                    let (flag, index) = (ty.repr() >> Self::FLAG_OFFSET, ty.repr() & Self::INDEX_MASK);
                    match flag {
                        $(${index(0)} => Self::$variant(index),)*
                        _ => unreachable!(),
                    }
                }

                $vis const fn compress(self) -> Id {
                    let (index, flag) = match self {
                        $(Self::$variant(index) => (index, ${index(0)}),)*
                    };
                   Id(unsafe { NonZeroU32::new_unchecked((flag << Self::FLAG_OFFSET) | index) })
                }

                $vis const fn inner(self) -> u32 {
                    match self {
                        $(Self::$variant(index) => index,)*
                    }
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
        pub(super) fn new(tys: &'a super::Types, files: &'a [parser::Ast], ty: Id) -> Self {
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
                    write!(f, "^{}", self.rety(self.tys.ptrs[ty as usize].base))
                }
                _ if let Some((key, _)) = self
                    .tys
                    .syms
                    .iter()
                    .find(|(sym, &ty)| sym.file != u32::MAX && ty == self.ty)
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
                    for (i, &super::Field { ref name, ty }) in record.fields.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
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

#[derive(Clone, Copy, Debug)]
struct Reloc {
    offset:     Offset,
    sub_offset: u8,
    width:      u8,
}

impl Reloc {
    fn new(offset: u32, sub_offset: u8, width: u8) -> Self {
        Self {
            offset,
            sub_offset,
            width,
        }
    }

    fn apply_stack_offset(&self, code: &mut [u8], stack: &stack::Alloc) {
        let bytes = &code[self.offset as usize + self.sub_offset as usize..][..self.width as usize];
        let (id, off) = Self::unpack_srel(u64::from_ne_bytes(bytes.try_into().unwrap()));
        self.write_offset(code, stack.final_offset(id, off) as i64);
    }

    fn pack_srel(id: &stack::Id, off: u32) -> u64 {
        ((id.repr() as u64) << 32) | (off as u64)
    }

    fn unpack_srel(id: u64) -> (u32, u32) {
        ((id >> 32) as u32, id as u32)
    }

    fn apply_jump(&self, code: &mut [u8], to: u32) {
        let offset = to as i64 - self.offset as i64;
        self.write_offset(code, offset);
    }

    fn write_offset(&self, code: &mut [u8], offset: i64) {
        let bytes = offset.to_ne_bytes();
        let slice =
            &mut code[self.offset as usize + self.sub_offset as usize..][..self.width as usize];
        slice.copy_from_slice(&bytes[..self.width as usize]);
        if slice.contains(&0x83) {
            panic!()
        }
    }
}

struct Value {
    ty:  ty::Id,
    loc: Loc,
}

impl Value {
    fn new(ty: impl Into<ty::Id>, loc: impl Into<Loc>) -> Self {
        Self {
            ty:  ty.into(),
            loc: loc.into(),
        }
    }

    fn void() -> Self {
        Self {
            ty:  ty::VOID.into(),
            loc: Loc::imm(0),
        }
    }

    fn imm(value: u64) -> Self {
        Self {
            ty:  ty::UINT.into(),
            loc: Loc::imm(value),
        }
    }

    fn ty(ty: ty::Id) -> Self {
        Self {
            ty:  ty::TYPE.into(),
            loc: Loc::Ct {
                value: (ty.repr() as u64).to_ne_bytes(),
            },
        }
    }
}

enum LocCow<'a> {
    Ref(&'a Loc),
    Owned(Loc),
}

impl<'a> LocCow<'a> {
    fn as_ref(&self) -> &Loc {
        match self {
            Self::Ref(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<'a> From<&'a Loc> for LocCow<'a> {
    fn from(value: &'a Loc) -> Self {
        Self::Ref(value)
    }
}

impl<'a> From<Loc> for LocCow<'a> {
    fn from(value: Loc) -> Self {
        Self::Owned(value)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Loc {
    Rt {
        derefed: bool,
        reg:     reg::Id,
        stack:   Option<stack::Id>,
        offset:  Offset,
    },
    Ct {
        value: [u8; 8],
    },
}

impl Loc {
    fn stack(stack: stack::Id) -> Self {
        Self::Rt {
            stack:   Some(stack),
            reg:     reg::STACK_PTR.into(),
            derefed: true,
            offset:  0,
        }
    }

    fn reg(reg: impl Into<reg::Id>) -> Self {
        let reg = reg.into();
        assert!(reg.get() != 0);
        Self::Rt {
            derefed: false,
            reg,
            stack: None,
            offset: 0,
        }
    }

    fn imm(value: u64) -> Self {
        Self::Ct {
            value: value.to_ne_bytes(),
        }
    }

    fn ty(ty: ty::Id) -> Self {
        Self::imm(ty.repr() as _)
    }

    fn offset(mut self, offset: u32) -> Self {
        match &mut self {
            Self::Rt { offset: off, .. } => *off += offset,
            _ => unreachable!("offseting constant"),
        }
        self
    }

    fn as_ref(&self) -> Self {
        match *self {
            Loc::Rt {
                derefed,
                ref reg,
                ref stack,
                offset,
            } => Loc::Rt {
                derefed,
                reg: reg.as_ref(),
                stack: stack.as_ref().map(stack::Id::as_ref),
                offset,
            },
            Loc::Ct { value } => Self::Ct { value },
        }
    }

    fn into_derefed(mut self) -> Self {
        match &mut self {
            Self::Rt { derefed, .. } => *derefed = true,
            val => unreachable!("{val:?}"),
        }
        self
    }

    fn assert_valid(&self) {
        assert!(!matches!(self, Self::Rt { reg, .. } if reg.get() == 0));
    }

    fn take_owned(&mut self) -> Option<Self> {
        if self.is_ref() {
            return None;
        }

        Some(std::mem::replace(self, self.as_ref()))
    }

    fn is_ref(&self) -> bool {
        matches!(self, Self::Rt { reg, stack, .. } if reg.is_ref() && stack.as_ref().map_or(true, stack::Id::is_ref))
    }

    fn to_ty(&self) -> Option<ty::Id> {
        match *self {
            Self::Ct { value } => Some(ty::Id::from(value)),
            Self::Rt { .. } => None,
        }
    }
}

impl From<reg::Id> for Loc {
    fn from(reg: reg::Id) -> Self {
        Loc::reg(reg)
    }
}

impl Default for Loc {
    fn default() -> Self {
        Self::Ct { value: [0; 8] }
    }
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
    file:    FileId,
    id:      ty::Kind,
    ret:     ty::Id,
    ret_reg: reg::Id,

    task_base: usize,
    snap:      Snapshot,

    stack: stack::Alloc,
    regs:  reg::Alloc,

    stack_relocs: Vec<Reloc>,
    ret_relocs:   Vec<Reloc>,
    loop_relocs:  Vec<Reloc>,
    loops:        Vec<Loop>,
    vars:         Vec<Variable>,
}

impl ItemCtx {
    // pub fn dup_loc(&mut self, loc: &Loc) -> Loc {
    //     match *loc {
    //         Loc::Rt {
    //             derefed,
    //             ref reg,
    //             ref stack,
    //             offset,
    //         } => Loc::Rt {
    //             reg: reg.as_ref(),
    //             derefed,
    //             stack: stack.as_ref().map(|s| self.stack.dup_id(s)),
    //             offset,
    //         },
    //         Loc::Ct { value } => Loc::Ct { value },
    //     }
    // }

    fn finalize(&mut self, output: &mut Output) {
        self.stack.finalize_leaked();
        for rel in self.stack_relocs.drain(..) {
            rel.apply_stack_offset(&mut output.code[self.snap.code..], &self.stack)
        }

        let ret_offset = output.code.len() - self.snap.code;
        for rel in self.ret_relocs.drain(..) {
            rel.apply_jump(&mut output.code[self.snap.code..], ret_offset as _);
        }

        self.finalize_frame(output);
        self.stack.clear();

        debug_assert!(self.loops.is_empty());
        debug_assert!(self.loop_relocs.is_empty());
        debug_assert!(self.vars.is_empty());
    }

    fn finalize_frame(&mut self, output: &mut Output) {
        let mut cursor = self.snap.code;
        let mut allocate = |size| (cursor += size, cursor).1;

        let pushed = self.regs.pushed_size() as i64;
        let stack = self.stack.max_height as i64;

        let mut exmpl = Output::default();
        exmpl.emit_prelude();

        debug_assert!(output.code[self.snap.code..].starts_with(&exmpl.code));

        write_reloc(&mut output.code, allocate(3), -(pushed + stack), 8);
        write_reloc(&mut output.code, allocate(8 + 3), stack, 8);
        write_reloc(&mut output.code, allocate(8), pushed, 2);

        output.emit(ld(RET_ADDR, STACK_PTR, stack as _, pushed as _));
        output.emit(addi64(STACK_PTR, STACK_PTR, (pushed + stack) as _));
    }

    fn free_loc(&mut self, src: impl Into<LocCow>) {
        if let LocCow::Owned(Loc::Rt { reg, stack, .. }) = src.into() {
            self.regs.free(reg);
            if let Some(stack) = stack {
                self.stack.free(stack);
            }
        }
    }
}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
    if value.contains(&131) {
        panic!();
    }
}

#[derive(PartialEq, Eq, Hash)]
struct SymKey {
    file:  u32,
    ident: u32,
}

impl SymKey {
    pub fn pointer_to(ty: ty::Id) -> Self {
        Self {
            file:  u32::MAX,
            ident: ty.repr(),
        }
    }
}

#[derive(Clone, Copy)]
struct Sig {
    args: ty::Tuple,
    ret:  ty::Id,
}

#[derive(Clone, Copy)]
struct Func {
    expr:   ExprRef,
    sig:    Option<Sig>,
    offset: Offset,
}

struct Global {
    offset: Offset,
    ty:     ty::Id,
}

struct Field {
    name: Rc<str>,
    ty:   ty::Id,
}

struct Struct {
    fields: Rc<[Field]>,
}

struct Ptr {
    base: ty::Id,
}

struct ParamAlloc(Range<u8>);

impl ParamAlloc {
    pub fn next(&mut self) -> u8 {
        self.0.next().expect("too many paramteters")
    }

    fn next_wide(&mut self) -> u8 {
        (self.next(), self.next()).0
    }
}

#[derive(Default)]
struct Types {
    syms: HashMap<SymKey, ty::Id>,

    funcs:   Vec<Func>,
    args:    Vec<ty::Id>,
    globals: Vec<Global>,
    structs: Vec<Struct>,
    ptrs:    Vec<Ptr>,
}

impl Types {
    fn parama(&self, ret: impl Into<ty::Id>) -> ParamAlloc {
        ParamAlloc(2 + (9..=16).contains(&self.size_of(ret.into())) as u8..12)
    }

    fn offset_of(&self, idx: ty::Struct, field: Result<&str, usize>) -> Option<(Offset, ty::Id)> {
        let record = &self.structs[idx as usize];
        let until = match field {
            Ok(str) => record.fields.iter().position(|f| f.name.as_ref() == str)?,
            Err(i) => i,
        };

        let mut offset = 0;
        for &Field { ty, .. } in &record.fields[..until] {
            offset = Self::align_up(offset, self.align_of(ty));
            offset += self.size_of(ty);
        }

        Some((offset, record.fields[until].ty))
    }

    fn make_ptr(&mut self, base: ty::Id) -> ty::Id {
        ty::Kind::Ptr(self.make_ptr_low(base)).compress()
    }

    fn make_ptr_low(&mut self, base: ty::Id) -> ty::Ptr {
        let id = SymKey::pointer_to(base);

        self.syms
            .entry(id)
            .or_insert_with(|| {
                self.ptrs.push(Ptr { base });
                ty::Kind::Ptr(self.ptrs.len() as u32 - 1).compress()
            })
            .expand()
            .inner()
    }

    fn align_up(value: Size, align: Size) -> Size {
        (value + align - 1) & !(align - 1)
    }

    fn size_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Ptr(_) => 8,
            ty::Kind::Builtin(ty::VOID) => 0,
            ty::Kind::Builtin(ty::NEVER) => unreachable!(),
            ty::Kind::Builtin(ty::INT | ty::UINT) => 8,
            ty::Kind::Builtin(ty::I32 | ty::U32 | ty::TYPE) => 4,
            ty::Kind::Builtin(ty::I16 | ty::U16) => 2,
            ty::Kind::Builtin(ty::I8 | ty::U8 | ty::BOOL) => 1,
            ty::Kind::Struct(ty) => {
                let mut offset = 0u32;
                let record = &self.structs[ty as usize];
                for &Field { ty, .. } in record.fields.iter() {
                    let align = self.align_of(ty);
                    offset = Self::align_up(offset, align);
                    offset += self.size_of(ty);
                }
                offset
            }
            ty => unimplemented!("size_of: {:?}", ty),
        }
    }

    fn align_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Struct(t) => self.structs[t as usize]
                .fields
                .iter()
                .map(|&Field { ty, .. }| self.align_of(ty))
                .max()
                .unwrap(),
            _ => self.size_of(ty).max(1),
        }
    }
}

mod task {
    use super::Offset;

    pub fn unpack(offset: Offset) -> Result<Offset, usize> {
        if offset >> 31 != 0 {
            Err((offset & !(1 << 31)) as usize)
        } else {
            Ok(offset)
        }
    }

    pub fn id(index: usize) -> Offset {
        1 << 31 | index as u32
    }
}

struct FTask {
    file: FileId,
    id:   ty::Func,
}

#[derive(Default, Clone, Copy)]
pub struct Snapshot {
    code:    usize,
    funcs:   usize,
    globals: usize,
}

#[derive(Default)]
struct Output {
    code:    Vec<u8>,
    funcs:   Vec<(ty::Func, Reloc)>,
    globals: Vec<(ty::Global, Reloc)>,
}

impl Output {
    fn emit_addi(&mut self, dest: u8, op: u8, delta: u64) {
        if delta == 0 {
            if dest != op {
                self.emit(cp(dest, op));
            }
            return;
        }

        self.emit(addi64(dest, op, delta));
    }

    fn emit(&mut self, (len, instr): (usize, [u8; instrs::MAX_SIZE])) {
        // let name = instrs::NAMES[instr[0] as usize];
        // log::dbg!(
        //     "{:08x}: {}: {}",
        //     self.code.len(),
        //     name,
        //     instr
        //         .iter()
        //         .take(len)
        //         .skip(1)
        //         .map(|b| format!("{:02x}", b))
        //         .collect::<String>()
        // );
        self.code.extend_from_slice(&instr[..len]);
    }

    fn emit_prelude(&mut self) {
        self.emit(instrs::addi64(STACK_PTR, STACK_PTR, 0));
        self.emit(instrs::st(RET_ADDR, STACK_PTR, 0, 0));
    }

    fn emit_entry_prelude(&mut self) {
        self.emit(jal(RET_ADDR, reg::ZERO, 0));
        self.emit(tx());
    }

    fn append(&mut self, val: &mut Self) {
        for (_, rel) in val.globals.iter_mut().chain(&mut val.funcs) {
            rel.offset += self.code.len() as Offset;
        }

        self.code.append(&mut val.code);
        self.funcs.append(&mut val.funcs);
        self.globals.append(&mut val.globals);
    }

    fn pop(&mut self, stash: &mut Self, snap: &Snapshot) {
        for (_, rel) in self.globals[snap.globals..]
            .iter_mut()
            .chain(&mut self.funcs[snap.funcs..])
        {
            rel.offset -= snap.code as Offset;
            rel.offset += stash.code.len() as Offset;
        }

        stash.code.extend(self.code.drain(snap.code..));
        stash.funcs.extend(self.funcs.drain(snap.funcs..));
        stash.globals.extend(self.globals.drain(snap.globals..));
    }

    fn trunc(&mut self, snap: &Snapshot) {
        self.code.truncate(snap.code);
        self.globals.truncate(snap.globals);
        self.funcs.truncate(snap.funcs);
    }

    fn write_trap(&mut self, trap: Trap) {
        let len = self.code.len();
        self.code.resize(len + std::mem::size_of::<Trap>(), 0);
        unsafe { std::ptr::write_unaligned(self.code.as_mut_ptr().add(len) as _, trap) }
    }

    fn snap(&mut self) -> Snapshot {
        Snapshot {
            code:    self.code.len(),
            funcs:   self.funcs.len(),
            globals: self.globals.len(),
        }
    }

    fn emit_call(&mut self, func_id: ty::Func) {
        let reloc = Reloc::new(self.code.len() as _, 3, 4);
        self.funcs.push((func_id, reloc));
        self.emit(jal(RET_ADDR, ZERO, 0));
    }
}

#[derive(Default, Debug)]
struct Ctx {
    loc: Option<Loc>,
    ty:  Option<ty::Id>,
}

impl Ctx {
    pub fn with_loc(self, loc: Loc) -> Self {
        Self {
            loc: Some(loc),
            ..self
        }
    }

    pub fn with_ty(self, ty: impl Into<ty::Id>) -> Self {
        Self {
            ty: Some(ty.into()),
            ..self
        }
    }

    fn into_value(self) -> Option<Value> {
        Some(Value {
            ty:  self.ty.unwrap(),
            loc: self.loc?,
        })
    }
}

impl From<Value> for Ctx {
    fn from(value: Value) -> Self {
        Self {
            loc: Some(value.loc),
            ty:  Some(value.ty),
        }
    }
}

#[derive(Default)]
struct Pool {
    cis:      Vec<ItemCtx>,
    outputs:  Vec<Output>,
    arg_locs: Vec<Loc>,
}

#[derive(Default)]
pub struct LoggedMem {
    pub mem: hbvm::mem::HostMemory,
}

impl hbvm::mem::Memory for LoggedMem {
    unsafe fn load(
        &mut self,
        addr: hbvm::mem::Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), hbvm::mem::LoadError> {
        log::dbg!(
            "load: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(addr.get() as *const u8, count)
                .iter()
                .rev()
                .map(|&b| format!("{b:02x}"))
                .collect::<String>()
        );
        self.mem.load(addr, target, count)
    }

    unsafe fn store(
        &mut self,
        addr: hbvm::mem::Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), hbvm::mem::StoreError> {
        log::dbg!(
            "store: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(source, count)
                .iter()
                .rev()
                .map(|&b| format!("{b:02x}"))
                .collect::<String>()
        );
        self.mem.store(addr, source, count)
    }

    unsafe fn prog_read<T: Copy>(&mut self, addr: hbvm::mem::Address) -> T {
        log::dbg!(
            "read-typed: {:x} {} {:?}",
            addr.get(),
            std::any::type_name::<T>(),
            if core::mem::size_of::<T>() == 1
                && let Some(nm) =
                    instrs::NAMES.get(std::ptr::read(addr.get() as *const u8) as usize)
            {
                nm.to_string()
            } else {
                core::slice::from_raw_parts(addr.get() as *const u8, core::mem::size_of::<T>())
                    .iter()
                    .map(|&b| format!("{:02x}", b))
                    .collect::<String>()
            }
        );
        self.mem.prog_read(addr)
    }
}

const VM_STACK_SIZE: usize = 1024 * 1024 * 2;

struct Comptime {
    vm:     hbvm::Vm<LoggedMem, 0>,
    _stack: Box<[u8; VM_STACK_SIZE]>,
}

impl Default for Comptime {
    fn default() -> Self {
        let mut stack = Box::<[u8; VM_STACK_SIZE]>::new_uninit();
        let mut vm = hbvm::Vm::default();
        let ptr = unsafe { stack.as_mut_ptr().cast::<u8>().add(VM_STACK_SIZE) as u64 };
        log::dbg!("stack_ptr: {:x}", ptr);
        vm.write_reg(STACK_PTR, ptr);
        Self {
            vm,
            _stack: unsafe { stack.assume_init() },
        }
    }
}

enum Trap {
    MakeStruct {
        file:        FileId,
        struct_expr: ExprRef,
    },
}

#[derive(Default)]
pub struct Codegen {
    pub files: Vec<parser::Ast>,
    tasks:     Vec<Option<FTask>>,

    tys:    Types,
    ci:     ItemCtx,
    output: Output,
    pool:   Pool,
    ct:     Comptime,
}

impl Codegen {
    pub fn generate(&mut self) {
        self.output.emit_entry_prelude();
        self.find_or_declare(0, 0, Err("main"), "");
        self.complete_call_graph_low();
        self.link();
    }

    pub fn dump(mut self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        let reloc = Reloc::new(0, 3, 4);
        self.output.funcs.push((0, reloc));
        self.link();
        out.write_all(&self.output.code)
    }

    fn expr(&mut self, expr: &Expr) -> Option<Value> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn build_struct(&mut self, fields: &[(&str, Expr)]) -> ty::Struct {
        let fields = fields
            .iter()
            .map(|&(name, ty)| Field {
                name: name.into(),
                ty:   self.ty(&ty),
            })
            .collect();
        self.tys.structs.push(Struct { fields });
        self.tys.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Value> {
        use {Expr as E, TokenKind as T};
        let value = match *expr {
            E::Mod { id, .. } => Some(Value::ty(ty::Kind::Module(id).compress())),
            E::Struct {
                fields, captured, ..
            } => {
                if captured.is_empty() {
                    Some(Value::ty(
                        ty::Kind::Struct(self.build_struct(fields)).compress(),
                    ))
                } else {
                    let values = captured
                        .iter()
                        .map(|&id| E::Ident {
                            pos: 0,
                            id,
                            name: "booodab",
                            index: u16::MAX,
                        })
                        .map(|expr| self.expr(&expr))
                        .collect::<Option<Vec<_>>>()?;
                    let values_size = values
                        .iter()
                        .map(|value| 4 + self.tys.size_of(value.ty))
                        .sum::<Size>();

                    let stack = self.ci.stack.allocate(values_size);
                    let mut ptr = Loc::stack(stack.as_ref());
                    for value in values {
                        self.store_sized(Loc::ty(value.ty), &ptr, 4);
                        ptr = ptr.offset(4);
                        let size = self.tys.size_of(value.ty);
                        self.store_sized(value.loc, &ptr, size);
                        ptr = ptr.offset(size);
                    }

                    self.stack_offset(2, STACK_PTR, Some(&stack), 0);
                    let val = self.eca(
                        Trap::MakeStruct {
                            file:        self.ci.file,
                            struct_expr: ExprRef::new(expr),
                        },
                        ty::TYPE,
                    );
                    self.ci.free_loc(Loc::stack(stack));
                    Some(val)
                }
            }
            E::UnOp {
                op: T::Xor, val, ..
            } => {
                let val = self.ty(val);
                Some(Value::ty(self.tys.make_ptr(val)))
            }
            E::Directive {
                name: "TypeOf",
                args: [expr],
                ..
            } => {
                let snap = self.output.snap();
                let value = self.expr(expr).unwrap();
                self.ci.free_loc(value.loc);
                self.output.trunc(&snap);
                Some(Value::ty(value.ty))
            }
            E::Directive {
                name: "eca",
                args: [ret_ty, args @ ..],
                ..
            } => {
                let ty = self.ty(ret_ty);

                let mut parama = self.tys.parama(ty);
                let base = self.pool.arg_locs.len();
                for arg in args {
                    let arg = self.expr(arg)?;
                    self.pass_arg(&arg, &mut parama);
                    self.pool.arg_locs.push(arg.loc);
                }
                for value in self.pool.arg_locs.drain(base..) {
                    self.ci.free_loc(value);
                }

                let loc = self.alloc_ret(ty, ctx);

                self.output.emit(eca());

                self.load_ret(ty, &loc);

                return Some(Value { ty, loc });
            }
            E::Directive {
                name: "sizeof",
                args: [ty],
                ..
            } => {
                let ty = self.ty(ty);
                return Some(Value::imm(self.tys.size_of(ty) as _));
            }
            E::Directive {
                name: "alignof",
                args: [ty],
                ..
            } => {
                let ty = self.ty(ty);
                return Some(Value::imm(self.tys.align_of(ty) as _));
            }
            E::Directive {
                name: "intcast",
                args: [val],
                ..
            } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, <expr>)`",
                    );
                };
                let mut val = self.expr(val)?;

                let from_size = self.tys.size_of(val.ty);
                let to_size = self.tys.size_of(ty);

                if from_size < to_size && val.ty.is_signed() {
                    let reg = self.loc_to_reg(val.loc, from_size);
                    let op = [sxt8, sxt16, sxt32][from_size.ilog2() as usize];
                    self.output.emit(op(reg.get(), reg.get()));
                    val.loc = Loc::reg(reg);
                }

                Some(Value { ty, loc: val.loc })
            }
            E::Directive {
                name: "bitcast",
                args: [val],
                ..
            } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, <expr>)`",
                    );
                };

                let size = self.tys.size_of(ty);

                ctx.ty = None;

                let val = self.expr_ctx(val, ctx)?;

                if self.tys.size_of(val.ty) != size {
                    self.report(
                        expr.pos(),
                        format_args!(
                            "cannot bitcast {} to {} (different sizes: {} != {size})",
                            self.ty_display(val.ty),
                            self.ty_display(ty),
                            self.tys.size_of(val.ty),
                        ),
                    );
                }

                debug_assert_eq!(
                    self.tys.align_of(val.ty),
                    self.tys.align_of(ty),
                    "TODO: might need stack relocation"
                );

                return Some(Value { ty, loc: val.loc });
            }
            E::Directive {
                name: "as",
                args: [ty, val],
                ..
            } => {
                let ty = self.ty(ty);
                ctx.ty = Some(ty);
                return self.expr_ctx(val, ctx);
            }
            E::Bool { value, .. } => Some(Value {
                ty:  ty::BOOL.into(),
                loc: Loc::imm(value as u64),
            }),
            E::Ctor {
                pos, ty, fields, ..
            } => {
                let Some(ty) = ty.map(|ty| self.ty(ty)).or(ctx.ty) else {
                    self.report(pos, "expected type, (it cannot be inferred)");
                };
                let size = self.tys.size_of(ty);

                let loc = ctx
                    .loc
                    .unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(size)));
                let ty::Kind::Struct(stuct) = ty.expand() else {
                    self.report(pos, "expected expression to evaluate to struct")
                };
                let field_count = self.tys.structs[stuct as usize].fields.len();
                if field_count != fields.len() {
                    self.report(
                        pos,
                        format_args!("expected {} fields, got {}", field_count, fields.len()),
                    );
                }

                for (i, (name, field)) in fields.iter().enumerate() {
                    let Some((offset, ty)) = self.tys.offset_of(stuct, name.ok_or(i)) else {
                        self.report(pos, format_args!("field not found: {name:?}"));
                    };
                    let loc = loc.as_ref().offset(offset);
                    let value = self.expr_ctx(field, Ctx::default().with_loc(loc).with_ty(ty))?;
                    self.ci.free_loc(value.loc);
                }

                return Some(Value { ty, loc });
            }
            E::Field { target, field } => {
                let checkpoint = self.output.code.len();
                let mut tal = self.expr(target)?;

                if let ty::Kind::Ptr(ty) = tal.ty.expand() {
                    tal.ty = self.tys.ptrs[ty as usize].base;
                    tal.loc = tal.loc.into_derefed();
                }

                match tal.ty.expand() {
                    ty::Kind::Struct(idx) => {
                        let Some((offset, ty)) = self.tys.offset_of(idx, Ok(field)) else {
                            self.report(target.pos(), format_args!("field not found: {field:?}"));
                        };
                        Some(Value {
                            ty,
                            loc: tal.loc.offset(offset),
                        })
                    }
                    ty::Kind::Builtin(ty::TYPE) => {
                        self.output.code.truncate(checkpoint);
                        match ty::Kind::from_ty(self.ty(target)) {
                            ty::Kind::Module(idx) => Some(Value::ty(
                                self.find_or_declare(target.pos(), idx, Err(field), "")
                                    .compress(),
                            )),
                            _ => unimplemented!(),
                        }
                    }
                    smh => self.report(
                        target.pos(),
                        format_args!("the field operation is not supported: {smh:?}"),
                    ),
                }
            }
            E::UnOp {
                op: T::Band,
                val,
                pos,
            } => {
                let mut val = self.expr(val)?;
                let Loc::Rt {
                    derefed: drfd @ true,
                    reg,
                    stack,
                    offset,
                } = &mut val.loc
                else {
                    self.report(
                        pos,
                        format_args!(
                            "cant take pointer of {} ({:?})",
                            self.ty_display(val.ty),
                            val.loc
                        ),
                    );
                };

                *drfd = false;
                let offset = std::mem::take(offset) as _;
                if reg.is_ref() {
                    let new_reg = self.ci.regs.allocate();
                    self.stack_offset(new_reg.get(), reg.get(), stack.as_ref(), offset);
                    *reg = new_reg;
                } else {
                    self.stack_offset(reg.get(), reg.get(), stack.as_ref(), offset);
                }

                // FIXME: we might be able to track this but it will be pain
                std::mem::forget(stack.take());

                Some(Value {
                    ty:  self.tys.make_ptr(val.ty),
                    loc: val.loc,
                })
            }
            E::UnOp {
                op: T::Mul,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                match val.ty.expand() {
                    ty::Kind::Ptr(ty) => Some(Value {
                        ty:  self.tys.ptrs[ty as usize].base,
                        loc: Loc::reg(self.loc_to_reg(val.loc, self.tys.size_of(val.ty)))
                            .into_derefed(),
                    }),
                    _ => self.report(
                        pos,
                        format_args!("expected pointer, got {}", self.ty_display(val.ty)),
                    ),
                }
            }
            E::BinOp {
                left: &E::Ident { id, .. },
                op: T::Decl,
                right,
            } => {
                let val = self.expr(right)?;
                let mut loc = self.make_loc_owned(val.loc, val.ty);
                let sym = parser::find_symbol(&self.cfile().symbols, id);
                if sym.flags & idfl::REFERENCED != 0 {
                    loc = self.spill(loc, self.tys.size_of(val.ty));
                }
                self.ci.vars.push(Variable {
                    id,
                    value: Value { ty: val.ty, loc },
                });
                Some(Value::void())
            }
            E::Call { func: fast, args } => {
                let func_ty = self.ty(fast);
                let ty::Kind::Func(mut func_id) = func_ty.expand() else {
                    self.report(fast.pos(), "can't call this, maybe in the future");
                };

                let func = self.tys.funcs[func_id as usize];
                let ast = self.cfile().clone();
                let E::BinOp {
                    right:
                        &E::Closure {
                            args: cargs, ret, ..
                        },
                    ..
                } = func.expr.get(&ast).unwrap()
                else {
                    unreachable!();
                };

                let sig = if let Some(sig) = func.sig {
                    sig
                } else {
                    let scope = self.ci.vars.len();
                    let arg_base = self.tys.args.len();

                    for (arg, carg) in args.iter().zip(cargs) {
                        let ty = self.ty(&carg.ty);
                        log::dbg!("arg: {}", self.ty_display(ty));
                        self.tys.args.push(ty);
                        let sym = parser::find_symbol(&ast.symbols, carg.id);
                        let loc = if sym.flags & idfl::COMPTIME == 0 {
                            // FIXME: could fuck us
                            Loc::default()
                        } else {
                            debug_assert_eq!(
                                ty,
                                ty::TYPE.into(),
                                "TODO: we dont support anything except type generics"
                            );
                            let arg = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                            self.tys.args.push(arg.loc.to_ty().unwrap());
                            arg.loc
                        };

                        self.ci.vars.push(Variable {
                            id:    carg.id,
                            value: Value { ty, loc },
                        });
                    }

                    let args = self.pack_args(expr.pos(), arg_base);
                    let ret = self.ty(ret);
                    self.ci.vars.truncate(scope);

                    let sym = SymKey {
                        file:  !args.repr(),
                        ident: func_id,
                    };
                    let ct = || {
                        let func_id = self.tys.funcs.len();
                        self.tys.funcs.push(Func {
                            offset: task::id(func_id),
                            sig:    Some(Sig { args, ret }),
                            expr:   func.expr,
                        });

                        self.tasks.push(Some(FTask {
                            // FIXME: this will fuck us
                            file: self.ci.file,
                            id:   func_id as _,
                        }));

                        ty::Kind::Func(func_id as _).compress()
                    };
                    func_id = self.tys.syms.entry(sym).or_insert_with(ct).expand().inner();

                    Sig { args, ret }
                };

                let mut parama = self.tys.parama(sig.ret);
                let mut values = Vec::with_capacity(args.len());
                let mut sig_args = sig.args.range();
                for (arg, carg) in args.iter().zip(cargs) {
                    let ty = self.tys.args[sig_args.next().unwrap()];
                    let sym = parser::find_symbol(&ast.symbols, carg.id);
                    if sym.flags & idfl::COMPTIME != 0 {
                        sig_args.next().unwrap();
                        continue;
                    }

                    // TODO: pass the arg as dest
                    let varg = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    self.pass_arg(&varg, &mut parama);
                    values.push(varg.loc);
                }

                for value in values {
                    self.ci.free_loc(value);
                }

                log::dbg!("call ctx: {ctx:?}");

                let loc = self.alloc_ret(sig.ret, ctx);
                self.output.emit_call(func_id);
                self.load_ret(sig.ret, &loc);
                return Some(Value { ty: sig.ret, loc });
            }
            E::Ident { id, .. } if ident::is_null(id) => Some(Value::ty(id.into())),
            E::Ident { id, index, .. }
                if let Some((var_index, var)) = self
                    .ci
                    .vars
                    .iter_mut()
                    .enumerate()
                    .find(|(_, v)| v.id == id) =>
            {
                let sym = parser::find_symbol(&self.files[self.ci.file as usize].symbols, id);
                let loc = match idfl::index(sym.flags) == dbg!(index)
                    && !self
                        .ci
                        .loops
                        .last()
                        .is_some_and(|l| l.var_count > var_index as u32)
                {
                    true => {
                        dbg!(
                            log::dbg!("braj: {expr}"),
                            std::mem::take(&mut var.value.loc)
                        )
                        .1
                    }
                    false => var.value.loc.as_ref(),
                };

                Some(Value {
                    ty: self.ci.vars[var_index].value.ty,
                    loc,
                })
            }
            E::Ident { id, name, .. } => match self
                .tys
                .syms
                .get(&SymKey {
                    ident: id,
                    file:  self.ci.file,
                })
                .copied()
                .map(ty::Kind::from_ty)
                .unwrap_or_else(|| self.find_or_declare(ident::pos(id), self.ci.file, Ok(id), name))
            {
                ty::Kind::Global(id) => self.handle_global(id),
                tk => Some(Value::ty(tk.compress())),
            },
            E::Return { val, .. } => {
                if let Some(val) = val {
                    let size = self.tys.size_of(self.ci.ret);
                    let loc = match size {
                        0 => Loc::default(),
                        1..=16 => Loc::reg(1),
                        _ => Loc::reg(self.ci.ret_reg.as_ref()).into_derefed(),
                    };
                    self.expr_ctx(val, Ctx::default().with_ty(self.ci.ret).with_loc(loc))?;
                }
                let off = self.local_offset();
                self.ci.ret_relocs.push(Reloc::new(off, 1, 4));
                self.output.emit(jmp(0));
                None
            }
            E::Block { stmts, .. } => {
                for stmt in stmts {
                    self.expr(stmt)?;
                }
                Some(Value::void())
            }
            E::Number { value, .. } => Some(Value {
                ty:  ctx.ty.map(ty::Id::strip_pointer).unwrap_or(ty::INT.into()),
                loc: Loc::imm(value),
            }),
            E::If {
                cond, then, else_, ..
            } => {
                log::dbg!("if-cond");
                let cond = self.expr_ctx(cond, Ctx::default().with_ty(ty::BOOL))?;
                let reg = self.loc_to_reg(&cond.loc, 1);
                let jump_offset = self.local_offset();
                self.output.emit(jeq(reg.get(), 0, 0));
                self.ci.free_loc(cond.loc);

                log::dbg!("if-then");
                let then_unreachable = self.expr(then).is_none();
                let mut else_unreachable = false;

                let mut jump = self.local_offset() as i64 - jump_offset as i64;

                if let Some(else_) = else_ {
                    log::dbg!("if-else");
                    let else_jump_offset = self.local_offset();
                    if !then_unreachable {
                        self.output.emit(jmp(0));
                        jump = self.local_offset() as i64 - jump_offset as i64;
                    }

                    else_unreachable = self.expr(else_).is_none();

                    if !then_unreachable {
                        let jump = self.local_offset() as i64 - else_jump_offset as i64;
                        log::dbg!("if-else-jump: {}", jump);
                        write_reloc(self.local_code(), else_jump_offset as usize + 1, jump, 4);
                    }
                }

                log::dbg!("if-then-jump: {}", jump);
                write_reloc(self.local_code(), jump_offset as usize + 3, jump, 2);

                (!then_unreachable || !else_unreachable).then_some(Value::void())
            }
            E::Loop { body, .. } => 'a: {
                log::dbg!("loop");

                let loop_start = self.local_offset();
                self.ci.loops.push(Loop {
                    var_count:  self.ci.vars.len() as _,
                    offset:     loop_start,
                    reloc_base: self.ci.loop_relocs.len() as u32,
                });
                let body_unreachable = self.expr(body).is_none();

                log::dbg!("loop-end");
                if !body_unreachable {
                    let loop_end = self.local_offset();
                    self.output.emit(jmp(loop_start as i32 - loop_end as i32));
                }

                let loop_end = self.local_offset();

                let loopa = self.ci.loops.pop().unwrap();
                let is_unreachable = loopa.reloc_base == self.ci.loop_relocs.len() as u32;
                for reloc in self.ci.loop_relocs.drain(loopa.reloc_base as usize..) {
                    reloc.apply_jump(&mut self.output.code[self.ci.snap.code..], loop_end);
                }

                let mut vars = std::mem::take(&mut self.ci.vars);
                for var in vars.drain(loopa.var_count as usize..) {
                    self.ci.free_loc(var.value.loc);
                }
                self.ci.vars = vars;

                if is_unreachable {
                    log::dbg!("infinite loop");
                    break 'a None;
                }

                Some(Value::void())
            }
            E::Break { .. } => {
                let offset = self.local_offset();
                self.ci.loop_relocs.push(Reloc::new(offset, 1, 4));
                self.output.emit(jmp(0));
                None
            }
            E::Continue { .. } => {
                let loop_ = self.ci.loops.last().unwrap();
                let offset = self.local_offset();
                self.output.emit(jmp(loop_.offset as i32 - offset as i32));
                None
            }
            E::BinOp {
                left,
                op: op @ (T::And | T::Or),
                right,
            } => {
                let lhs = self.expr_ctx(left, Ctx::default().with_ty(ty::BOOL))?;
                let lhs = self.loc_to_reg(lhs.loc, 1);
                let jump_offset = self.output.code.len() + 3;
                let op = if op == T::And { jeq } else { jne };
                self.output.emit(op(lhs.get(), 0, 0));

                if let Some(rhs) = self.expr_ctx(right, Ctx::default().with_ty(ty::BOOL)) {
                    let rhs = self.loc_to_reg(rhs.loc, 1);
                    self.output.emit(cp(lhs.get(), rhs.get()));
                }

                let jump = self.output.code.len() as i64 - jump_offset as i64;
                write_reloc(&mut self.output.code, jump_offset, jump, 2);

                Some(Value {
                    ty:  ty::BOOL.into(),
                    loc: Loc::reg(lhs),
                })
            }
            E::BinOp { left, op, right } => 'ops: {
                let left = self.expr(left)?;

                if op == T::Assign {
                    let value = self.expr_ctx(right, Ctx::from(left)).unwrap();
                    self.ci.free_loc(value.loc);
                    return Some(Value::void());
                }

                if let ty::Kind::Struct(_) = left.ty.expand() {
                    let right = self.expr_ctx(right, Ctx::default().with_ty(left.ty))?;
                    _ = self.assert_ty(expr.pos(), left.ty, right.ty);
                    return self.struct_op(op, left.ty, ctx, left.loc, right.loc);
                }

                let lsize = self.tys.size_of(left.ty);

                let lhs = self.loc_to_reg(left.loc, lsize);
                log::dbg!("{expr}");
                let right = self.expr_ctx(right, Ctx::default().with_ty(left.ty))?;
                let rsize = self.tys.size_of(right.ty);

                let ty = self.assert_ty(expr.pos(), left.ty, right.ty);
                let size = self.tys.size_of(ty);
                let signed = ty.is_signed();

                if let Loc::Ct { value } = right.loc
                    && let Some(oper) = Self::imm_math_op(op, signed, size)
                {
                    let mut imm = u64::from_ne_bytes(value);
                    if matches!(op, T::Add | T::Sub)
                        && let ty::Kind::Ptr(ty) = ty::Kind::from_ty(ty)
                    {
                        let size = self.tys.size_of(self.tys.ptrs[ty as usize].base);
                        imm *= size as u64;
                    }

                    self.output.emit(oper(lhs.get(), lhs.get(), imm));
                    break 'ops Some(Value::new(ty, lhs));
                }

                let rhs = self.loc_to_reg(right.loc, rsize);

                if matches!(op, T::Add | T::Sub) {
                    let min_size = lsize.min(rsize);
                    if ty.is_signed() && min_size < size {
                        let operand = if lsize < rsize { lhs.get() } else { rhs.get() };
                        let op = [sxt8, sxt16, sxt32][min_size.ilog2() as usize];
                        self.output.emit(op(operand, operand));
                    }

                    if left.ty.is_pointer() ^ right.ty.is_pointer() {
                        let (offset, ty) = if left.ty.is_pointer() {
                            (rhs.get(), left.ty)
                        } else {
                            (lhs.get(), right.ty)
                        };

                        let ty::Kind::Ptr(ty) = ty.expand() else {
                            unreachable!()
                        };

                        let size = self.tys.size_of(self.tys.ptrs[ty as usize].base);
                        self.output.emit(muli64(offset, offset, size as _));
                    }
                }

                if let Some(op) = Self::math_op(op, signed, size) {
                    self.output.emit(op(lhs.get(), lhs.get(), rhs.get()));
                    self.ci.regs.free(rhs);
                    break 'ops Some(Value::new(ty, lhs));
                }

                'cmp: {
                    let against = match op {
                        T::Le | T::Gt => 1,
                        T::Ne | T::Eq => 0,
                        T::Ge | T::Lt => (-1i64) as _,
                        _ => break 'cmp,
                    };

                    let op_fn = if signed { cmps } else { cmpu };
                    self.output.emit(op_fn(lhs.get(), lhs.get(), rhs.get()));
                    self.output.emit(cmpui(lhs.get(), lhs.get(), against));
                    if matches!(op, T::Eq | T::Lt | T::Gt) {
                        self.output.emit(not(lhs.get(), lhs.get()));
                    }

                    self.ci.regs.free(rhs);
                    break 'ops Some(Value::new(ty::BOOL, lhs));
                }

                unimplemented!("{:#?}", op)
            }
            ast => unimplemented!("{:#?}", ast),
        }?;

        if let Some(ty) = ctx.ty {
            _ = self.assert_ty(expr.pos(), value.ty, ty);
        }

        Some(match ctx.loc {
            Some(dest) => {
                let ty = ctx.ty.unwrap_or(value.ty);
                self.store_typed(value.loc, dest, ty);
                Value {
                    ty,
                    loc: Loc::imm(0),
                }
            }
            None => value,
        })
    }

    fn struct_op(
        &mut self,
        op: TokenKind,
        ty: ty::Id,
        ctx: Ctx,
        left: Loc,
        mut right: Loc,
    ) -> Option<Value> {
        if let ty::Kind::Struct(stuct) = ty.expand() {
            let loc = ctx
                .loc
                .or_else(|| right.take_owned())
                .unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(self.tys.size_of(ty))));
            let mut offset = 0;
            for &Field { ty, .. } in self.tys.structs[stuct as usize].fields.clone().iter() {
                offset = Types::align_up(offset, self.tys.align_of(ty));
                let size = self.tys.size_of(ty);
                let ctx = Ctx::from(Value {
                    ty,
                    loc: loc.as_ref().offset(offset),
                });
                let left = left.as_ref().offset(offset);
                let right = right.as_ref().offset(offset);
                let value = self.struct_op(op, ty, ctx, left, right)?;
                self.ci.free_loc(value.loc);
                offset += size;
            }

            self.ci.free_loc(left);
            self.ci.free_loc(right);

            return Some(Value { ty, loc });
        }

        let size = self.tys.size_of(ty);
        let signed = ty.is_signed();
        let lhs = self.loc_to_reg(left, size);

        if let Loc::Ct { value } = right
            && let Some(op) = Self::imm_math_op(op, signed, size)
        {
            self.output
                .emit(op(lhs.get(), lhs.get(), u64::from_ne_bytes(value)));
            return Some(if let Some(value) = ctx.into_value() {
                self.store_typed(Loc::reg(lhs.as_ref()), value.loc, value.ty);
                Value::void()
            } else {
                Value {
                    ty,
                    loc: Loc::reg(lhs),
                }
            });
        }

        let rhs = self.loc_to_reg(right, size);

        if let Some(op) = Self::math_op(op, signed, size) {
            self.output.emit(op(lhs.get(), lhs.get(), rhs.get()));
            self.ci.regs.free(rhs);
            return if let Some(value) = ctx.into_value() {
                self.store_typed(Loc::reg(lhs), value.loc, value.ty);
                Some(Value::void())
            } else {
                Some(Value {
                    ty,
                    loc: Loc::reg(lhs),
                })
            };
        }

        unimplemented!("{:#?}", op)
    }

    #[allow(clippy::type_complexity)]
    fn math_op(
        op: TokenKind,
        signed: bool,
        size: u32,
    ) -> Option<fn(u8, u8, u8) -> (usize, [u8; instrs::MAX_SIZE])> {
        use TokenKind as T;

        macro_rules! div { ($($op:ident),*) => {[$(|a, b, c| $op(a, ZERO, b, c)),*]}; }
        macro_rules! rem { ($($op:ident),*) => {[$(|a, b, c| $op(ZERO, a, b, c)),*]}; }

        let ops = match op {
            T::Add => [add8, add16, add32, add64],
            T::Sub => [sub8, sub16, sub32, sub64],
            T::Mul => [mul8, mul16, mul32, mul64],
            T::Div if signed => div!(dirs8, dirs16, dirs32, dirs64),
            T::Div => div!(diru8, diru16, diru32, diru64),
            T::Mod if signed => rem!(dirs8, dirs16, dirs32, dirs64),
            T::Mod => rem!(diru8, diru16, diru32, diru64),
            T::Band => return Some(and),
            T::Bor => return Some(or),
            T::Xor => return Some(xor),
            T::Shl => [slu8, slu16, slu32, slu64],
            T::Shr if signed => [srs8, srs16, srs32, srs64],
            T::Shr => [sru8, sru16, sru32, sru64],
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
    }

    #[allow(clippy::type_complexity)]
    fn imm_math_op(
        op: TokenKind,
        signed: bool,
        size: u32,
    ) -> Option<fn(u8, u8, u64) -> (usize, [u8; instrs::MAX_SIZE])> {
        use TokenKind as T;

        macro_rules! def_op {
            ($name:ident |$a:ident, $b:ident, $c:ident| $($tt:tt)*) => {
                macro_rules! $name {
                    ($$($$op:ident),*) => {
                        [$$(
                            |$a, $b, $c: u64| $$op($($tt)*),
                        )*]
                    }
                }
            };
        }

        def_op!(basic_op | a, b, c | a, b, c as _);
        def_op!(sub_op | a, b, c | b, a, c.wrapping_neg() as _);

        let ops = match op {
            T::Add => basic_op!(addi8, addi16, addi32, addi64),
            T::Sub => sub_op!(addi8, addi16, addi32, addi64),
            T::Mul => basic_op!(muli8, muli16, muli32, muli64),
            T::Band => return Some(andi),
            T::Bor => return Some(ori),
            T::Xor => return Some(xori),
            T::Shr if signed => basic_op!(srui8, srui16, srui32, srui64),
            T::Shr => basic_op!(srui8, srui16, srui32, srui64),
            T::Shl => basic_op!(slui8, slui16, slui32, slui64),
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
    }

    fn handle_global(&mut self, id: ty::Global) -> Option<Value> {
        let ptr = self.ci.regs.allocate();

        let global = &mut self.tys.globals[id as usize];

        let reloc = Reloc::new(self.output.code.len() as u32, 3, 4);
        self.output.globals.push((id, reloc));
        self.output.emit(instrs::lra(ptr.get(), 0, 0));

        Some(Value {
            ty:  global.ty,
            loc: Loc::reg(ptr).into_derefed(),
        })
    }

    fn spill(&mut self, loc: Loc, size: Size) -> Loc {
        let stack = Loc::stack(self.ci.stack.allocate(size));
        self.store_sized(loc, &stack, size);
        stack
    }

    fn make_loc_owned(&mut self, loc: Loc, ty: ty::Id) -> Loc {
        let size = self.tys.size_of(ty);
        match size {
            0 => Loc::default(),
            1..=8 => Loc::reg(self.loc_to_reg(loc, size)),
            _ if loc.is_ref() => {
                let new_loc = Loc::stack(self.ci.stack.allocate(size));
                self.store_sized(loc, &new_loc, size);
                new_loc
            }
            _ => loc,
        }
    }

    #[must_use]
    fn complete_call_graph(&mut self) -> Output {
        let stash = self.pop_stash();
        self.complete_call_graph_low();
        stash
    }

    fn complete_call_graph_low(&mut self) {
        while self.ci.task_base < self.tasks.len()
            && let Some(task_slot) = self.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.handle_task(task);
        }
        self.ci.snap = self.output.snap();
    }

    fn handle_task(&mut self, FTask { file, id }: FTask) {
        let func = self.tys.funcs[id as usize];
        let sig = func.sig.unwrap();
        let ast = self.files[file as usize].clone();
        let expr = func.expr.get(&ast).unwrap();

        let repl = ItemCtx {
            file,
            id: ty::Kind::Func(id),
            ret: sig.ret,
            ..self.pool.cis.pop().unwrap_or_default()
        };
        let prev_ci = std::mem::replace(&mut self.ci, repl);
        self.ci.regs.init();
        self.ci.snap = self.output.snap();

        let Expr::BinOp {
            left: Expr::Ident { name, .. },
            op: TokenKind::Decl,
            right: &Expr::Closure { body, args, .. },
        } = expr
        else {
            unreachable!("{expr}")
        };

        log::dbg!("fn: {}", name);

        self.output.emit_prelude();

        log::dbg!("fn-args");
        let mut parama = self.tys.parama(self.ci.ret);
        let mut sig_args = sig.args.range();
        for arg in args.iter() {
            let ty = self.tys.args[sig_args.next().unwrap()];
            let sym = parser::find_symbol(&ast.symbols, arg.id);
            let loc = match sym.flags & idfl::COMPTIME != 0 {
                true => Loc::ty(self.tys.args[sig_args.next().unwrap()]),
                false => self.load_arg(sym.flags, ty, &mut parama),
            };
            self.ci.vars.push(Variable {
                id:    arg.id,
                value: Value { ty, loc },
            });
        }

        if self.tys.size_of(self.ci.ret) > 16 {
            let reg = self.ci.regs.allocate();
            self.output.emit(instrs::cp(reg.get(), 1));
            self.ci.ret_reg = reg;
        } else {
            self.ci.ret_reg = reg::Id::RET;
        }

        log::dbg!("fn-body");
        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        for vars in self.ci.vars.drain(..).collect::<Vec<_>>() {
            self.ci.free_loc(vars.value.loc);
        }

        log::dbg!("fn-prelude, stack: {:x}", self.ci.stack.max_height);

        log::dbg!("fn-relocs");
        self.ci.finalize(&mut self.output);
        self.output.emit(jala(ZERO, RET_ADDR, 0));
        self.ci.regs.free(std::mem::take(&mut self.ci.ret_reg));
        self.tys.funcs[id as usize].offset = self.ci.snap.code as Offset;
        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
    }

    fn load_arg(&mut self, flags: parser::IdentFlags, ty: ty::Id, parama: &mut ParamAlloc) -> Loc {
        let size = self.tys.size_of(ty) as Size;
        let (src, dst) = match size {
            0 => (Loc::default(), Loc::default()),
            ..=8 if flags & idfl::REFERENCED == 0 => {
                (Loc::reg(parama.next()), Loc::reg(self.ci.regs.allocate()))
            }
            1..=8 => (
                Loc::reg(parama.next()),
                Loc::stack(self.ci.stack.allocate(size)),
            ),
            9..=16 => (
                Loc::reg(parama.next_wide()),
                Loc::stack(self.ci.stack.allocate(size)),
            ),
            _ if flags & (idfl::MUTABLE | idfl::REFERENCED) == 0 => {
                let ptr = parama.next();
                let reg = self.ci.regs.allocate();
                self.output.emit(instrs::cp(reg.get(), ptr));
                return Loc::reg(reg).into_derefed();
            }
            _ => (
                Loc::reg(parama.next()).into_derefed(),
                Loc::stack(self.ci.stack.allocate(size)),
            ),
        };

        self.store_sized(src, &dst, size);
        dst
    }

    fn eca(&mut self, trap: Trap, ret: impl Into<ty::Id>) -> Value {
        self.output.emit(eca());
        self.output.write_trap(trap);
        Value {
            ty:  ret.into(),
            loc: Loc::reg(1),
        }
    }

    fn alloc_ret(&mut self, ret: ty::Id, ctx: Ctx) -> Loc {
        let size = self.tys.size_of(ret);
        match size {
            0 => Loc::default(),
            1..=8 => Loc::reg(1),
            9..=16 => Loc::stack(self.ci.stack.allocate(size)),
            _ => {
                let loc = ctx
                    .loc
                    .unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(size)));
                let Loc::Rt {
                    reg, stack, offset, ..
                } = &loc
                else {
                    todo!("old man with the beard looks at the sky scared");
                };
                self.stack_offset(1, reg.get(), stack.as_ref(), *offset);
                loc
            }
        }
    }

    fn loc_to_reg(&mut self, loc: impl Into<LocCow>, size: Size) -> reg::Id {
        match loc.into() {
            LocCow::Owned(Loc::Rt {
                derefed: false,
                mut reg,
                offset,
                stack,
            }) => {
                debug_assert!(stack.is_none(), "TODO");
                assert_eq!(offset, 0, "TODO");
                if reg.is_ref() {
                    let new_reg = self.ci.regs.allocate();
                    self.output.emit(cp(new_reg.get(), reg.get()));
                    reg = new_reg;
                }
                reg
            }
            LocCow::Ref(&Loc::Rt {
                derefed: false,
                ref reg,
                offset,
                ref stack,
            }) => {
                debug_assert!(stack.is_none(), "TODO");
                assert_eq!(offset, 0, "TODO");
                reg.as_ref()
            }
            loc => {
                let reg = self.ci.regs.allocate();
                self.store_sized(loc, Loc::reg(reg.as_ref()), size);
                reg
            }
        }
    }

    fn load_ret(&mut self, ty: ty::Id, loc: &Loc) {
        let size = self.tys.size_of(ty);
        if let 1..=16 = size {
            self.store_sized(Loc::reg(1), loc, size);
        }
    }

    fn pass_arg(&mut self, value: &Value, parama: &mut ParamAlloc) {
        self.pass_arg_low(&value.loc, self.tys.size_of(value.ty), parama)
    }

    fn pass_arg_low(&mut self, loc: &Loc, size: Size, parama: &mut ParamAlloc) {
        if size > 16 {
            let Loc::Rt {
                reg, stack, offset, ..
            } = loc
            else {
                unreachable!()
            };
            self.stack_offset(parama.next(), reg.get(), stack.as_ref(), *offset as _);
            return;
        }

        let dst = match size {
            0 => return,
            9..=16 => Loc::reg(parama.next_wide()),
            _ => Loc::reg(parama.next()),
        };

        self.store_sized(loc, dst, size);
    }

    fn store_typed(&mut self, src: impl Into<LocCow>, dst: impl Into<LocCow>, ty: ty::Id) {
        self.store_sized(src, dst, self.tys.size_of(ty) as _)
    }

    fn store_sized(&mut self, src: impl Into<LocCow>, dst: impl Into<LocCow>, size: Size) {
        self.store_sized_low(src.into(), dst.into(), size);
    }

    fn store_sized_low(&mut self, src: LocCow, dst: LocCow, size: Size) {
        macro_rules! lpat {
            ($der:literal, $reg:ident, $off:pat, $sta:pat) => {
                &Loc::Rt { derefed: $der, reg: ref $reg, offset: $off, stack: $sta }
            };
        }

        src.as_ref().assert_valid();
        dst.as_ref().assert_valid();

        match (src.as_ref(), dst.as_ref()) {
            (&Loc::Ct { value }, lpat!(true, reg, off, ref sta)) => {
                let ct = self.ci.regs.allocate();
                self.output.emit(li64(ct.get(), u64::from_ne_bytes(value)));
                let off = self.opt_stack_reloc(sta.as_ref(), off, 3);
                self.output.emit(st(ct.get(), reg.get(), off, size as _));
                self.ci.regs.free(ct);
            }
            (&Loc::Ct { value }, lpat!(false, reg, 0, None)) => {
                self.output.emit(li64(reg.get(), u64::from_ne_bytes(value)))
            }
            (lpat!(true, src, soff, ref ssta), lpat!(true, dst, doff, ref dsta)) => {
                // TODO: some oportuinies to ellit more optimal code
                let src_off = self.ci.regs.allocate();
                let dst_off = self.ci.regs.allocate();
                self.stack_offset(src_off.get(), src.get(), ssta.as_ref(), soff);
                self.stack_offset(dst_off.get(), dst.get(), dsta.as_ref(), doff);
                self.output
                    .emit(bmc(src_off.get(), dst_off.get(), size as _));
                self.ci.regs.free(src_off);
                self.ci.regs.free(dst_off);
            }
            (lpat!(false, src, 0, None), lpat!(false, dst, 0, None)) => {
                if src != dst {
                    self.output.emit(cp(dst.get(), src.get()));
                }
            }
            (lpat!(true, src, soff, ref ssta), lpat!(false, dst, 0, None)) => {
                if size < 8 {
                    self.output.emit(cp(dst.get(), 0));
                }
                let off = self.opt_stack_reloc(ssta.as_ref(), soff, 3);
                self.output.emit(ld(dst.get(), src.get(), off, size as _));
            }
            (lpat!(false, src, 0, None), lpat!(true, dst, doff, ref dsta)) => {
                let off = self.opt_stack_reloc(dsta.as_ref(), doff, 3);
                self.output.emit(st(src.get(), dst.get(), off, size as _))
            }
            (a, b) => unreachable!("{a:?} {b:?}"),
        }

        self.ci.free_loc(src);
        self.ci.free_loc(dst);
    }

    fn stack_offset(&mut self, dst: u8, op: u8, stack: Option<&stack::Id>, off: Offset) {
        let Some(stack) = stack else {
            self.output.emit_addi(dst, op, off as _);
            return;
        };

        let off = self.stack_reloc(stack, off, 3);
        self.output.emit(addi64(dst, op, off));
    }

    fn opt_stack_reloc(&mut self, stack: Option<&stack::Id>, off: Offset, sub_offset: u8) -> u64 {
        stack
            .map(|s| self.stack_reloc(s, off, sub_offset))
            .unwrap_or(off as _)
    }

    fn stack_reloc(&mut self, stack: &stack::Id, off: Offset, sub_offset: u8) -> u64 {
        log::dbg!("whaaaaatahack: {:b}", stack.repr());
        let offset = self.local_offset();
        self.ci.stack_relocs.push(Reloc::new(offset, sub_offset, 8));
        Reloc::pack_srel(stack, off)
    }

    fn link(&mut self) {
        // FIXME: this will cause problems relating snapshots

        self.output.funcs.retain(|&(f, rel)| {
            _ = task::unpack(self.tys.funcs[f as usize].offset)
                .map(|off| rel.apply_jump(&mut self.output.code, off));
            true
        });

        self.output.globals.retain(|&(g, rel)| {
            _ = task::unpack(self.tys.globals[g as usize].offset)
                .map(|off| rel.apply_jump(&mut self.output.code, off));
            true
        })
    }

    // TODO: sometimes its better to do this in bulk
    fn ty(&mut self, expr: &Expr) -> ty::Id {
        let mut ci = ItemCtx {
            file: self.ci.file,
            id: self.ci.id,
            ..self.pool.cis.pop().unwrap_or_default()
        };
        ci.vars.append(&mut self.ci.vars);

        let loc = self.ct_eval(ci, |s, prev| {
            s.output.emit_prelude();

            let ctx = Ctx::default().with_ty(ty::TYPE);
            let Some(ret) = s.expr_ctx(expr, ctx) else {
                s.report(expr.pos(), "type cannot be unreachable");
            };

            let stash = s.complete_call_graph();
            s.push_stash(stash);

            prev.vars.append(&mut s.ci.vars);
            s.ci.finalize(&mut s.output);
            s.output.emit(tx());

            let loc = match ret.loc {
                Loc::Rt { ref reg, .. } => Ok(reg.get()),
                Loc::Ct { value } => Err(value),
            };
            s.ci.free_loc(ret.loc);

            loc
        });

        ty::Id::from(match loc {
            Ok(reg) => self.ct.vm.read_reg(reg).cast::<u64>().to_ne_bytes(),
            Err(ct) => ct,
        })
    }

    fn handle_ecall(&mut self) {
        let arr = self.ct.vm.pc.get() as *const Trap;
        let trap = unsafe { std::ptr::read_unaligned(arr) };
        self.ct.vm.pc = self.ct.vm.pc.wrapping_add(std::mem::size_of::<Trap>());

        match trap {
            Trap::MakeStruct { file, struct_expr } => {
                let cfile = self.files[file as usize].clone();
                let &Expr::Struct {
                    fields, captured, ..
                } = struct_expr.get(&cfile).unwrap()
                else {
                    unreachable!()
                };

                let prev_len = self.ci.vars.len();

                let mut values = self.ct.vm.read_reg(2).0 as *const u8;
                for &id in captured {
                    let ty: ty::Id = unsafe { std::ptr::read_unaligned(values.cast()) };
                    unsafe { values = values.add(4) };
                    let size = self.tys.size_of(ty) as usize;
                    let mut imm = [0u8; 8];
                    assert!(size <= imm.len(), "TODO");
                    unsafe { std::ptr::copy_nonoverlapping(values, imm.as_mut_ptr(), size) };
                    self.ci.vars.push(Variable {
                        id,
                        value: Value::new(ty, Loc::imm(u64::from_ne_bytes(imm))),
                    });
                }

                let stru = ty::Kind::Struct(self.build_struct(fields)).compress();
                self.ci.vars.truncate(prev_len);
                self.ct.vm.write_reg(1, stru.repr() as u64);
            }
        }
    }

    fn find_or_declare(
        &mut self,
        pos: Pos,
        file: FileId,
        name: Result<Ident, &str>,
        lit_name: &str,
    ) -> ty::Kind {
        log::dbg!("find_or_declare: {lit_name}");
        let f = self.files[file as usize].clone();
        let Some((expr, ident)) = f.find_decl(name) else {
            match name {
                Ok(_) => self.report(pos, format_args!("undefined indentifier: {lit_name}")),
                Err("main") => self.report(pos, format_args!("missing main function: {f}")),
                Err(name) => unimplemented!("somehow we did not handle: {name:?}"),
            }
        };

        if let Some(existing) = self.tys.syms.get(&SymKey { file, ident }) {
            if let ty::Kind::Func(id) = existing.expand()
                && let func = &mut self.tys.funcs[id as usize]
                && let Err(idx) = task::unpack(func.offset)
            {
                func.offset = task::id(self.tasks.len());
                let task = self.tasks[idx].take();
                self.tasks.push(task);
            }
            return existing.expand();
        }

        let sym = match expr {
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: &Expr::Closure { pos, args, ret, .. },
            } => {
                let id = self.tys.funcs.len() as _;
                let func = Func {
                    offset: task::id(self.tasks.len()),
                    sig:    'b: {
                        let arg_base = self.tys.args.len();
                        for arg in args {
                            let sym = find_symbol(&self.files[file as usize].symbols, arg.id);
                            if sym.flags & idfl::COMPTIME != 0 {
                                self.tys.args.truncate(arg_base);
                                break 'b None;
                            }
                            let ty = self.ty(&arg.ty);
                            self.tys.args.push(ty);
                        }

                        self.tasks.push(Some(FTask { file, id }));

                        let args = self.pack_args(pos, arg_base);
                        let ret = self.ty(ret);
                        Some(Sig { args, ret })
                    },
                    expr:   ExprRef::new(expr),
                };
                self.tys.funcs.push(func);

                ty::Kind::Func(id)
            }
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: Expr::Struct { fields, .. },
            } => ty::Kind::Struct(self.build_struct(fields)),
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right,
            } => {
                let gid = self.tys.globals.len() as ty::Global;
                self.tys.globals.push(Global {
                    offset: u32::MAX,
                    ty:     Default::default(),
                });

                let ci = ItemCtx {
                    file,
                    id: ty::Kind::Global(gid),
                    ..self.pool.cis.pop().unwrap_or_default()
                };

                self.tys.globals[gid as usize] = self
                    .ct_eval(ci, |s, _| Ok::<_, !>(s.generate_global(right)))
                    .into_ok();

                ty::Kind::Global(gid)
            }
            e => unimplemented!("{e:#?}"),
        };
        self.tys.syms.insert(SymKey { ident, file }, sym.compress());
        sym
    }

    fn generate_global(&mut self, expr: &Expr) -> Global {
        self.output.emit_prelude();

        let ret = self.ci.regs.allocate();
        self.output.emit(instrs::cp(ret.get(), 1));
        self.ci.task_base = self.tasks.len();

        let ctx = Ctx::default().with_loc(Loc::reg(ret).into_derefed());
        let Some(ret) = self.expr_ctx(expr, ctx) else {
            self.report(expr.pos(), "expression is not reachable");
        };

        let stash = self.complete_call_graph();

        let offset = self.ci.snap.code;
        self.ci.snap.code += self.tys.size_of(ret.ty) as usize;
        self.output.code.resize(self.ci.snap.code, 0);

        self.push_stash(stash);

        self.ci.finalize(&mut self.output);
        self.output.emit(tx());

        let ret_loc = unsafe { self.output.code.as_mut_ptr().add(offset) };
        self.ct.vm.write_reg(1, ret_loc as u64);

        Global {
            ty:     ret.ty,
            offset: offset as _,
        }
    }

    fn pop_stash(&mut self) -> Output {
        let mut stash = self.pool.outputs.pop().unwrap_or_default();
        self.output.pop(&mut stash, &self.ci.snap);
        stash
    }

    fn push_stash(&mut self, mut stash: Output) {
        self.output.append(&mut stash);
        self.pool.outputs.push(stash);
    }

    fn ct_eval<T, E>(
        &mut self,
        ci: ItemCtx,
        compile: impl FnOnce(&mut Self, &mut ItemCtx) -> Result<T, E>,
    ) -> Result<T, E> {
        let stash = self.pop_stash();

        let mut prev_ci = std::mem::replace(&mut self.ci, ci);
        self.ci.snap = self.output.snap();
        self.ci.task_base = self.tasks.len();
        self.ci.regs.init();

        let ret = compile(self, &mut prev_ci);

        if ret.is_ok() {
            self.link();
            let entry = &mut self.output.code[self.ci.snap.code] as *mut _ as _;
            self.ct.vm.pc = hbvm::mem::Address::new(entry);
            loop {
                match self.ct.vm.run().unwrap() {
                    hbvm::VmRunOk::End => break,
                    hbvm::VmRunOk::Timer => unreachable!(),
                    hbvm::VmRunOk::Ecall => self.handle_ecall(),
                    hbvm::VmRunOk::Breakpoint => unreachable!(),
                }
            }
        }

        self.output.trunc(&self.ci.snap);
        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
        self.ci.snap = self.output.snap();
        self.push_stash(stash);

        ret
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, &self.files, ty)
    }

    #[must_use]
    fn assert_ty(&self, pos: Pos, ty: ty::Id, expected: ty::Id) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected) {
            res
        } else {
            let ty = self.ty_display(ty);
            let expected = self.ty_display(expected);
            self.report(pos, format_args!("expected {expected}, got {ty}"));
        }
    }

    fn report(&self, pos: Pos, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.cfile().nlines.line_col(pos);
        println!("{}:{}:{}: {}", self.cfile().path, line, col, msg);
        unreachable!();
    }

    fn cfile(&self) -> &parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn local_code(&mut self) -> &mut [u8] {
        &mut self.output.code[self.ci.snap.code..]
    }

    fn local_offset(&self) -> u32 {
        (self.output.code.len() - self.ci.snap.code) as u32
    }

    fn pack_args(&mut self, pos: Pos, arg_base: usize) -> ty::Tuple {
        let needle = &self.tys.args[arg_base..];
        if needle.is_empty() {
            return ty::Tuple::empty();
        }
        let len = needle.len();
        // FIXME: maybe later when this becomes a bottleneck we use more
        // efficient search (SIMD?, indexing?)
        let sp = self
            .tys
            .args
            .windows(needle.len())
            .position(|val| val == needle)
            .unwrap();
        self.tys.args.truncate((sp + needle.len()).max(arg_base));
        ty::Tuple::new(sp, len)
            .unwrap_or_else(|| self.report(pos, "amount of arguments not supported"))
    }
}

#[cfg(test)]
mod tests {
    use crate::{codegen::LoggedMem, log};

    use super::parser;

    const README: &str = include_str!("../README.md");

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        fn find_block(mut input: &'static str, test_name: &'static str) -> &'static str {
            const CASE_PREFIX: &str = "#### ";
            const CASE_SUFFIX: &str = "\n```hb";
            loop {
                let Some(pos) = input.find(CASE_PREFIX) else {
                    unreachable!("test {test_name} not found");
                };

                input = unsafe { input.get_unchecked(pos + CASE_PREFIX.len()..) };
                if !input.starts_with(test_name) {
                    continue;
                }
                input = unsafe { input.get_unchecked(test_name.len()..) };
                if !input.starts_with(CASE_SUFFIX) {
                    continue;
                }
                input = unsafe { input.get_unchecked(CASE_SUFFIX.len()..) };

                let end = input.find("```").unwrap_or(input.len());
                break unsafe { input.get_unchecked(..end) };
            }
        }

        let input = find_block(input, ident);

        let path = "test";
        let mut codegen = super::Codegen {
            files: vec![parser::Ast::new(path, input, &parser::no_loader)],
            ..Default::default()
        };
        codegen.generate();
        let mut out = Vec::new();
        codegen.dump(&mut out).unwrap();

        use std::fmt::Write;

        let mut stack = [0_u64; 128];

        let mut vm = unsafe {
            hbvm::Vm::<_, 0>::new(
                LoggedMem::default(),
                hbvm::mem::Address::new(out.as_ptr() as u64),
            )
        };

        vm.write_reg(
            super::STACK_PTR,
            unsafe { stack.as_mut_ptr().add(stack.len()) } as u64,
        );

        let stat = loop {
            match vm.run() {
                Ok(hbvm::VmRunOk::End) => break Ok(()),
                Ok(hbvm::VmRunOk::Ecall) => match vm.read_reg(2).0 {
                    1 => writeln!(output, "ev: Ecall").unwrap(), // compatibility with a test
                    69 => {
                        let [size, align] = [vm.read_reg(3).0 as usize, vm.read_reg(4).0 as usize];
                        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
                        let ptr = unsafe { std::alloc::alloc(layout) };
                        vm.write_reg(1, ptr as u64);
                    }
                    96 => {
                        let [ptr, size, align] = [
                            vm.read_reg(3).0 as usize,
                            vm.read_reg(4).0 as usize,
                            vm.read_reg(5).0 as usize,
                        ];

                        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
                        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
                    }
                    unknown => unreachable!("unknown ecall: {unknown:?}"),
                },
                Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
                Err(e) => break Err(e),
            }
        };

        writeln!(output, "code size: {}", out.len()).unwrap();
        writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
        writeln!(output, "status: {:?}", stat).unwrap();
        log::inf!("input lenght: {}", input.len());
    }

    crate::run_tests! { generate:
        arithmetic => README;
        variables => README;
        functions => README;
        if_statements => README;
        loops => README;
        fb_driver => README;
        pointers => README;
        structs => README;
        different_types => README;
        struct_operators => README;
        directives => README;
        global_variables => README;
        generic_types => README;
        generic_functions => README;
    }
}
