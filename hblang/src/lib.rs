#![feature(
    let_chains,
    if_let_guard,
    macro_metavar_expr,
    anonymous_lifetime_in_impl_trait,
    core_intrinsics,
    never_type,
    unwrap_infallible,
    slice_partition_dedup,
    hash_raw_entry,
    portable_simd,
    iter_collect_into,
    new_uninit,
    ptr_metadata,
    slice_ptr_get,
    slice_take,
    map_try_insert,
    extract_if,
    ptr_internals
)]
#![allow(stable_features, internal_features, clippy::format_collect)]

use {
    self::{
        ident::Ident,
        parser::{Expr, ExprRef, FileId},
        son::reg,
        ty::ArrayLen,
    },
    hbbytecode as instrs,
    parser::Ast,
    std::{
        collections::{hash_map, BTreeMap, VecDeque},
        io,
        ops::Range,
        path::{Path, PathBuf},
        rc::Rc,
        sync::Mutex,
    },
};

#[macro_export]
macro_rules! run_tests {
    ($runner:path: $($name:ident => $input:expr;)*) => {$(
        #[test]
        fn $name() {
            $crate::run_test(std::any::type_name_of_val(&$name), stringify!($name), $input, $runner);
        }
    )*};
}

pub mod codegen;
pub mod parser;
pub mod son;

mod lexer;

mod task {
    use super::Offset;

    pub fn unpack(offset: Offset) -> Result<Offset, usize> {
        if offset >> 31 != 0 {
            Err((offset & !(1 << 31)) as usize)
        } else {
            Ok(offset)
        }
    }

    pub fn is_done(offset: Offset) -> bool {
        unpack(offset).is_ok()
    }

    pub fn id(index: usize) -> Offset {
        1 << 31 | index as u32
    }
}

mod ident {
    pub type Ident = u32;

    const LEN_BITS: u32 = 6;

    pub fn len(ident: u32) -> u32 {
        ident & ((1 << LEN_BITS) - 1)
    }

    pub fn is_null(ident: u32) -> bool {
        (ident >> LEN_BITS) == 0
    }

    pub fn pos(ident: u32) -> u32 {
        (ident >> LEN_BITS).saturating_sub(1)
    }

    pub fn new(pos: u32, len: u32) -> u32 {
        debug_assert!(len < (1 << LEN_BITS));
        ((pos + 1) << LEN_BITS) | len
    }

    pub fn range(ident: u32) -> std::ops::Range<usize> {
        let (len, pos) = (len(ident) as usize, pos(ident) as usize);
        pos..pos + len
    }
}

mod log {
    #![allow(unused_macros)]

    #[derive(PartialOrd, PartialEq, Ord, Eq, Debug)]
    pub enum Level {
        Err,
        Wrn,
        Inf,
        Dbg,
        Trc,
    }

    pub const LOG_LEVEL: Level = match option_env!("LOG_LEVEL") {
        Some(val) => match val.as_bytes()[0] {
            b'e' => Level::Err,
            b'w' => Level::Wrn,
            b'i' => Level::Inf,
            b'd' => Level::Dbg,
            b't' => Level::Trc,
            _ => panic!("Invalid log level."),
        },
        None => {
            if cfg!(debug_assertions) {
                Level::Dbg
            } else {
                Level::Err
            }
        }
    };

    macro_rules! log {
        ($level:expr, $fmt:literal $($expr:tt)*) => {
            if $level <= $crate::log::LOG_LEVEL {
                eprintln!("{:?}: {}", $level, format_args!($fmt $($expr)*));
            }
        };

        ($level:expr, $($arg:expr),+) => {
            if $level <= $crate::log::LOG_LEVEL {
                $(eprintln!("[{}:{}:{}][{:?}]: {} = {:?}", line!(), column!(), file!(), $level, stringify!($arg), $arg);)*
            }
        };
    }

    macro_rules! err { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Err, $($arg)*) }; }
    macro_rules! wrn { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Wrn, $($arg)*) }; }
    macro_rules! inf { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Inf, $($arg)*) }; }
    macro_rules! dbg { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Dbg, $($arg)*) }; }
    macro_rules! trc { ($($arg:tt)*) => { $crate::log::log!($crate::log::Level::Trc, $($arg)*) }; }

    #[allow(unused_imports)]
    pub(crate) use {dbg, err, inf, log, trc, wrn};
}

mod ty {
    use {
        crate::{
            lexer::TokenKind,
            parser::{self, Expr},
        },
        std::{num::NonZeroU32, ops::Range},
    };

    pub type ArrayLen = u32;

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;
    pub type Slice = u32;

    #[derive(Clone, Copy)]
    pub struct Tuple(pub u32);

    impl Tuple {
        const LEN_BITS: u32 = 5;
        const LEN_MASK: usize = Self::MAX_LEN - 1;
        const MAX_LEN: usize = 1 << Self::LEN_BITS;

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

        pub fn empty() -> Self {
            Self(0)
        }

        pub fn repr(&self) -> u32 {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    pub struct Id(NonZeroU32);

    impl Default for Id {
        fn default() -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(UNDECLARED) })
        }
    }

    impl Id {
        pub fn is_signed(self) -> bool {
            (I8..=INT).contains(&self.repr())
        }

        pub fn is_unsigned(self) -> bool {
            (U8..=UINT).contains(&self.repr())
        }

        pub fn is_integer(self) -> bool {
            (U8..=INT).contains(&self.repr())
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
                _ if oa == Self::from(NEVER) => ob,
                _ if ob == Self::from(NEVER) => oa,
                _ if oa == ob => oa,
                _ if oa.is_pointer() && ob.is_pointer() => return None,
                _ if a.is_signed() && b.is_signed() || a.is_unsigned() && b.is_unsigned() => ob,
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 < b.repr() - I8 => ob,
                _ if oa.is_integer() && ob.is_pointer() => ob,
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

    impl From<u64> for Id {
        fn from(id: u64) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(id as _) })
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
        LEFT_UNREACHABLE;
        RIGHT_UNREACHABLE;
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

                $vis fn from_ty(ty: Id) -> Self {
                    let (flag, index) = (ty.repr() >> Self::FLAG_OFFSET, ty.repr() & Self::INDEX_MASK);
                    match flag {
                        $(${index(0)} => Self::$variant(index),)*
                        i => unreachable!("{i}"),
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
            Slice,
        }
    }

    impl Default for Kind {
        fn default() -> Self {
            Self::Builtin(UNDECLARED)
        }
    }

    pub struct Display<'a> {
        tys: &'a super::Types,
        files: &'a [parser::Ast],
        ty: Id,
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
                    .find(|(sym, &ty)| sym.file < self.files.len() as u32 && ty == self.ty)
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
                TK::Slice(idx) => {
                    let array = self.tys.arrays[idx as usize];
                    match array.len {
                        ArrayLen::MAX => write!(f, "[{}]", self.rety(array.ty)),
                        len => write!(f, "[{}; {len}]", self.rety(array.ty)),
                    }
                }
            }
        }
    }

    pub fn bin_ret(ty: Id, op: TokenKind) -> Id {
        use TokenKind as T;
        match op {
            T::Lt | T::Gt | T::Le | T::Ge | T::Ne | T::Eq => BOOL.into(),
            _ => ty,
        }
    }
}

type EncodedInstr = (usize, [u8; instrs::MAX_SIZE]);
type Offset = u32;
type Size = u32;

fn emit(out: &mut Vec<u8>, (len, instr): EncodedInstr) {
    out.extend_from_slice(&instr[..len]);
}

#[derive(PartialEq, Eq, Hash)]
struct SymKey {
    file: u32,
    ident: u32,
}

impl SymKey {
    pub fn pointer_to(ty: ty::Id) -> Self {
        Self { file: u32::MAX, ident: ty.repr() }
    }
}

#[derive(Clone, Copy)]
struct Sig {
    args: ty::Tuple,
    ret: ty::Id,
}

struct Func {
    file: FileId,
    expr: ExprRef,
    sig: Option<Sig>,
    offset: Offset,
    // TODO: change to indices into common vec
    relocs: Vec<TypedReloc>,
    code: Vec<u8>,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            file: u32::MAX,
            expr: Default::default(),
            sig: None,
            offset: u32::MAX,
            relocs: Default::default(),
            code: Default::default(),
        }
    }
}

struct TypedReloc {
    target: ty::Id,
    reloc: Reloc,
}

struct Global {
    file: FileId,
    name: Ident,
    ty: ty::Id,
    offset: Offset,
    data: Vec<u8>,
}

impl Default for Global {
    fn default() -> Self {
        Self {
            ty: Default::default(),
            offset: u32::MAX,
            data: Default::default(),
            file: u32::MAX,
            name: u32::MAX,
        }
    }
}

// TODO: make into bit struct (width: u2, sub_offset: u3, offset: u27)
#[derive(Clone, Copy, Debug)]
struct Reloc {
    offset: Offset,
    sub_offset: u8,
    width: u8,
}

impl Reloc {
    fn new(offset: usize, sub_offset: u8, width: u8) -> Self {
        Self { offset: offset as u32, sub_offset, width }
    }

    fn apply_jump(mut self, code: &mut [u8], to: u32, from: u32) -> i64 {
        self.offset += from;
        let offset = to as i64 - self.offset as i64;
        self.write_offset(code, offset);
        offset
    }

    fn write_offset(&self, code: &mut [u8], offset: i64) {
        let bytes = offset.to_ne_bytes();
        let slice = &mut code[self.offset as usize + self.sub_offset as usize..];
        slice[..self.width as usize].copy_from_slice(&bytes[..self.width as usize]);
    }
}

struct Field {
    name: Rc<str>,
    ty: ty::Id,
}

struct Struct {
    fields: Rc<[Field]>,
}

struct Ptr {
    base: ty::Id,
}

#[derive(Clone, Copy)]
struct Array {
    ty: ty::Id,
    len: ArrayLen,
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

#[repr(packed)]
#[allow(dead_code)]
struct AbleOsExecutableHeader {
    magic_number: [u8; 3],
    executable_version: u32,

    code_length: u64,
    data_length: u64,
    debug_length: u64,
    config_length: u64,
    metadata_length: u64,
}

#[derive(Default)]
struct Types {
    syms: HashMap<SymKey, ty::Id>,

    funcs: Vec<Func>,
    args: Vec<ty::Id>,
    globals: Vec<Global>,
    structs: Vec<Struct>,
    ptrs: Vec<Ptr>,
    arrays: Vec<Array>,
}

const HEADER_SIZE: usize = std::mem::size_of::<AbleOsExecutableHeader>();

impl Types {
    fn assemble(&mut self, to: &mut Vec<u8>) {
        to.extend([0u8; HEADER_SIZE]);

        emit(to, instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
        emit(to, instrs::tx());
        let exe = self.dump_reachable(0, to);
        Reloc::new(HEADER_SIZE, 3, 4).apply_jump(to, self.funcs[0].offset, 0);

        unsafe { *to.as_mut_ptr().cast::<AbleOsExecutableHeader>() = exe }
    }

    fn dump_reachable(&mut self, from: ty::Func, to: &mut Vec<u8>) -> AbleOsExecutableHeader {
        let mut used_funcs = vec![];
        let mut used_globals = vec![];

        let mut frontier = vec![ty::Kind::Func(from).compress()];

        while let Some(itm) = frontier.pop() {
            match itm.expand() {
                ty::Kind::Func(func) => {
                    let fuc = &mut self.funcs[func as usize];
                    if task::is_done(fuc.offset) {
                        continue;
                    }
                    fuc.offset = 0;
                    used_funcs.push(func);
                    frontier.extend(fuc.relocs.iter().map(|r| r.target));
                }
                ty::Kind::Global(glob) => {
                    let glb = &mut self.globals[glob as usize];
                    if task::is_done(glb.offset) {
                        continue;
                    }
                    glb.offset = 0;
                    used_globals.push(glob);
                }
                _ => unreachable!(),
            }
        }

        for &func in &used_funcs {
            let fuc = &mut self.funcs[func as usize];
            fuc.offset = to.len() as _;
            to.extend(&fuc.code);
        }

        let code_length = to.len();

        for &global in &used_globals {
            let global = &mut self.globals[global as usize];
            global.offset = to.len() as _;
            to.extend(&global.data);
        }

        let data_length = to.len() - code_length;

        for func in used_funcs {
            let fuc = &self.funcs[func as usize];
            for rel in &fuc.relocs {
                let offset = match rel.target.expand() {
                    ty::Kind::Func(fun) => self.funcs[fun as usize].offset,
                    ty::Kind::Global(glo) => self.globals[glo as usize].offset,
                    _ => unreachable!(),
                };
                rel.reloc.apply_jump(to, offset, fuc.offset);
            }
        }

        AbleOsExecutableHeader {
            magic_number: [0x15, 0x91, 0xD2],
            executable_version: 0,
            code_length: (code_length - HEADER_SIZE) as _,
            data_length: data_length as _,
            debug_length: 0,
            config_length: 0,
            metadata_length: 0,
        }
    }

    pub fn disasm(
        &self,
        mut sluce: &[u8],
        files: &[parser::Ast],
        output: &mut impl std::io::Write,
        eca_handler: impl FnMut(&mut &[u8]),
    ) -> std::io::Result<()> {
        use instrs::DisasmItem;
        let functions = self
            .funcs
            .iter()
            .filter(|f| task::is_done(f.offset))
            .map(|f| {
                let name = if f.file != u32::MAX {
                    let file = &files[f.file as usize];
                    let Expr::BinOp { left: &Expr::Ident { name, .. }, .. } =
                        f.expr.get(file).unwrap()
                    else {
                        unreachable!()
                    };
                    name
                } else {
                    "target_fn"
                };
                (f.offset, (name, f.code.len() as u32, DisasmItem::Func))
            })
            .chain(self.globals.iter().filter(|g| task::is_done(g.offset)).map(|g| {
                let name = if g.file == u32::MAX {
                    std::str::from_utf8(&g.data).unwrap()
                } else {
                    let file = &files[g.file as usize];
                    file.ident_str(g.name)
                };
                (g.offset, (name, g.data.len() as Size, DisasmItem::Global))
            }))
            .collect::<BTreeMap<_, _>>();
        instrs::disasm(&mut sluce, &functions, output, eca_handler)
    }

    fn parama(&self, ret: impl Into<ty::Id>) -> ParamAlloc {
        ParamAlloc(2 + (9..=16).contains(&self.size_of(ret.into())) as u8..12)
    }

    fn offset_of(&self, idx: ty::Struct, field: &str) -> Option<(Offset, ty::Id)> {
        let record = &self.structs[idx as usize];
        let until = record.fields.iter().position(|f| f.name.as_ref() == field)?;
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

    fn make_array(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Id {
        ty::Kind::Slice(self.make_array_low(ty, len)).compress()
    }

    fn make_array_low(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Slice {
        let id = SymKey {
            file: match len {
                ArrayLen::MAX => ArrayLen::MAX - 1,
                len => ArrayLen::MAX - len - 2,
            },
            ident: ty.repr(),
        };

        self.syms
            .entry(id)
            .or_insert_with(|| {
                self.arrays.push(Array { ty, len });
                ty::Kind::Slice(self.arrays.len() as u32 - 1).compress()
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
            ty::Kind::Slice(arr) => {
                let arr = &self.arrays[arr as usize];
                match arr.len {
                    0 => 0,
                    ArrayLen::MAX => 16,
                    len => self.size_of(arr.ty) * len,
                }
            }
            ty::Kind::Struct(stru) => {
                let mut offset = 0u32;
                let record = &self.structs[stru as usize];
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
            ty::Kind::Struct(stru) => self.structs[stru as usize]
                .fields
                .iter()
                .map(|&Field { ty, .. }| self.align_of(ty))
                .max()
                .unwrap(),
            ty::Kind::Slice(arr) => {
                let arr = &self.arrays[arr as usize];
                match arr.len {
                    ArrayLen::MAX => 8,
                    _ => self.align_of(arr.ty),
                }
            }
            _ => self.size_of(ty).max(1),
        }
    }
}

struct TaskQueue<T> {
    inner: Mutex<TaskQueueInner<T>>,
}

impl<T> TaskQueue<T> {
    fn new(max_waiters: usize) -> Self {
        Self { inner: Mutex::new(TaskQueueInner::new(max_waiters)) }
    }

    pub fn push(&self, message: T) {
        self.extend([message]);
    }

    pub fn extend(&self, messages: impl IntoIterator<Item = T>) {
        self.inner.lock().unwrap().push(messages);
    }

    pub fn pop(&self) -> Option<T> {
        TaskQueueInner::pop(&self.inner)
    }
}

enum TaskSlot<T> {
    Waiting,
    Delivered(T),
    Closed,
}

struct TaskQueueInner<T> {
    max_waiters: usize,
    messages: VecDeque<T>,
    parked: VecDeque<(*mut TaskSlot<T>, std::thread::Thread)>,
}

unsafe impl<T: Send> Send for TaskQueueInner<T> {}
unsafe impl<T: Send + Sync> Sync for TaskQueueInner<T> {}

impl<T> TaskQueueInner<T> {
    fn new(max_waiters: usize) -> Self {
        Self { max_waiters, messages: Default::default(), parked: Default::default() }
    }

    fn push(&mut self, messages: impl IntoIterator<Item = T>) {
        for msg in messages {
            if let Some((dest, thread)) = self.parked.pop_front() {
                unsafe { *dest = TaskSlot::Delivered(msg) };
                thread.unpark();
            } else {
                self.messages.push_back(msg);
            }
        }
    }

    fn pop(s: &Mutex<Self>) -> Option<T> {
        let mut res = TaskSlot::Waiting;
        {
            let mut s = s.lock().unwrap();
            if let Some(msg) = s.messages.pop_front() {
                return Some(msg);
            }

            if s.max_waiters == s.parked.len() + 1 {
                for (dest, thread) in s.parked.drain(..) {
                    unsafe { *dest = TaskSlot::Closed };
                    thread.unpark();
                }
                return None;
            }

            s.parked.push_back((&mut res, std::thread::current()));
        }

        loop {
            std::thread::park();

            let _s = s.lock().unwrap();
            match std::mem::replace(&mut res, TaskSlot::Waiting) {
                TaskSlot::Delivered(msg) => return Some(msg),
                TaskSlot::Closed => return None,
                TaskSlot::Waiting => {}
            }
        }
    }
}

pub fn parse_from_fs(extra_threads: usize, root: &str) -> io::Result<Vec<Ast>> {
    fn resolve(path: &str, from: &str) -> Result<PathBuf, CantLoadFile> {
        let path = match Path::new(from).parent() {
            Some(parent) => PathBuf::from_iter([parent, Path::new(path)]),
            None => PathBuf::from(path),
        };

        path.canonicalize().map_err(|source| CantLoadFile { path, source })
    }

    #[derive(Debug)]
    struct CantLoadFile {
        path: PathBuf,
        source: io::Error,
    }

    impl std::fmt::Display for CantLoadFile {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "can't load file: {}", parser::display_rel_path(&self.path),)
        }
    }

    impl std::error::Error for CantLoadFile {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.source)
        }
    }

    impl From<CantLoadFile> for io::Error {
        fn from(e: CantLoadFile) -> Self {
            io::Error::new(io::ErrorKind::InvalidData, e)
        }
    }

    type Task = (u32, PathBuf);

    let seen = Mutex::new(HashMap::<PathBuf, u32>::default());
    let tasks = TaskQueue::<Task>::new(extra_threads + 1);
    let ast = Mutex::new(Vec::<io::Result<Ast>>::new());

    let loader = |path: &str, from: &str| {
        if path.starts_with("rel:") {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "`rel:` prefix was removed and is now equivalent to no prefix (remove it)"
                    .to_string(),
            ));
        }

        let physiscal_path = resolve(path, from)?;

        let id = {
            let mut seen = seen.lock().unwrap();
            let len = seen.len();
            match seen.entry(physiscal_path.clone()) {
                hash_map::Entry::Occupied(entry) => {
                    return Ok(*entry.get());
                }
                hash_map::Entry::Vacant(entry) => {
                    entry.insert(len as _);
                    len as u32
                }
            }
        };

        if !physiscal_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("can't find file: {}", parser::display_rel_path(&physiscal_path)),
            ));
        }

        tasks.push((id, physiscal_path));
        Ok(id)
    };

    let execute_task = |(_, path): Task| {
        let path = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("path contains invalid characters: {}", parser::display_rel_path(&path)),
            )
        })?;
        Ok(Ast::new(path, std::fs::read_to_string(path)?, &loader))
    };

    let thread = || {
        while let Some(task @ (indx, ..)) = tasks.pop() {
            let res = execute_task(task);
            let mut ast = ast.lock().unwrap();
            let len = ast.len().max(indx as usize + 1);
            ast.resize_with(len, || Err(io::ErrorKind::InvalidData.into()));
            ast[indx as usize] = res;
        }
    };

    let path = Path::new(root).canonicalize()?;
    seen.lock().unwrap().insert(path.clone(), 0);
    tasks.push((0, path));

    if extra_threads == 0 {
        thread();
    } else {
        std::thread::scope(|s| (0..extra_threads + 1).for_each(|_| _ = s.spawn(thread)));
    }

    ast.into_inner().unwrap().into_iter().collect::<io::Result<Vec<_>>>()
}

type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FnvHasher>>;
type _HashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<FnvHasher>>;

struct FnvHasher(u64);

impl std::hash::Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0 = bytes.iter().fold(self.0, |hash, &byte| {
            let mut hash = hash;
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001B3);
            hash
        });
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self(0xCBF29CE484222325)
    }
}

#[cfg(test)]
pub fn run_test(
    name: &'static str,
    ident: &'static str,
    input: &'static str,
    test: fn(&'static str, &'static str, &mut String),
) {
    use std::{io::Write, path::PathBuf};

    let filter = std::env::var("PT_FILTER").unwrap_or_default();
    if !filter.is_empty() && !name.contains(&filter) {
        return;
    }

    let mut output = String::new();
    {
        struct DumpOut<'a>(&'a mut String);
        impl Drop for DumpOut<'_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    println!("{}", self.0);
                }
            }
        }

        let dump = DumpOut(&mut output);
        test(ident, input, dump.0);
    }

    let mut root = PathBuf::from(
        std::env::var("PT_TEST_ROOT")
            .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/tests").to_string()),
    );
    root.push(name.replace("::", "_").replace(concat!(env!("CARGO_PKG_NAME"), "_"), ""));
    root.set_extension("txt");

    let expected = std::fs::read_to_string(&root).unwrap_or_default();

    if output == expected {
        return;
    }

    if std::env::var("PT_UPDATE").is_ok() {
        std::fs::write(&root, output).unwrap();
        return;
    }

    if !root.exists() {
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::write(&root, vec![]).unwrap();
    }

    let mut proc = std::process::Command::new("diff")
        .arg("-u")
        .arg("--color")
        .arg(&root)
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::inherit())
        .spawn()
        .unwrap();

    proc.stdin.as_mut().unwrap().write_all(output.as_bytes()).unwrap();

    proc.wait().unwrap();

    panic!("test failed");
}

#[cfg(test)]
fn test_parse_files(ident: &'static str, input: &'static str) -> Vec<parser::Ast> {
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

    let mut module_map = Vec::new();
    let mut last_start = 0;
    let mut last_module_name = "test";
    for (i, m) in input.match_indices("// in module: ") {
        parser::test::format(ident, input[last_start..i].trim());
        module_map.push((last_module_name, &input[last_start..i]));
        let (module_name, _) = input[i + m.len()..].split_once('\n').unwrap();
        last_module_name = module_name;
        last_start = i + m.len() + module_name.len() + 1;
    }
    parser::test::format(ident, input[last_start..].trim());
    module_map.push((last_module_name, input[last_start..].trim()));

    let loader = |path: &str, _: &str| {
        module_map
            .iter()
            .position(|&(name, _)| name == path)
            .map(|i| i as parser::FileId)
            .ok_or(io::Error::from(io::ErrorKind::NotFound))
    };

    module_map
        .iter()
        .map(|&(path, content)| parser::Ast::new(path, content.to_owned(), &loader))
        .collect()
}

#[cfg(test)]
fn test_run_vm(out: &[u8], output: &mut String) {
    use std::fmt::Write;

    let mut stack = [0_u64; 1024 * 20];

    let mut vm = unsafe {
        hbvm::Vm::<_, { 1024 * 100 }>::new(
            LoggedMem::default(),
            hbvm::mem::Address::new(out.as_ptr() as u64).wrapping_add(HEADER_SIZE),
        )
    };

    vm.write_reg(codegen::STACK_PTR, unsafe { stack.as_mut_ptr().add(stack.len()) } as u64);

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
                3 => vm.write_reg(1, 42),
                unknown => unreachable!("unknown ecall: {unknown:?}"),
            },
            Ok(hbvm::VmRunOk::Timer) => {
                writeln!(output, "timed out").unwrap();
                break Ok(());
            }
            Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
            Err(e) => break Err(e),
        }
    };

    writeln!(output, "code size: {}", out.len() - HEADER_SIZE).unwrap();
    writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
    writeln!(output, "status: {:?}", stat).unwrap();
}

#[derive(Default)]
pub struct Options {
    pub fmt: bool,
    pub fmt_current: bool,
    pub dump_asm: bool,
    pub extra_threads: usize,
}

fn format_to(
    ast: &parser::Ast,
    source: &str,
    out: &mut impl std::io::Write,
) -> std::io::Result<()> {
    parser::with_fmt_source(source, || {
        for (i, expr) in ast.exprs().iter().enumerate() {
            write!(out, "{expr}")?;
            if let Some(expr) = ast.exprs().get(i + 1)
                && let Some(rest) = source.get(expr.pos() as usize..)
            {
                if parser::insert_needed_semicolon(rest) {
                    write!(out, ";")?;
                }
                if parser::preserve_newlines(&source[..expr.pos() as usize]) > 1 {
                    writeln!(out)?;
                }
            }

            if i + 1 != ast.exprs().len() {
                writeln!(out)?;
            }
        }
        std::io::Result::Ok(())
    })
}

pub fn run_compiler(
    root_file: &str,
    options: Options,
    out: &mut impl std::io::Write,
) -> io::Result<()> {
    let parsed = parse_from_fs(options.extra_threads, root_file)?;

    fn format_ast(ast: parser::Ast) -> std::io::Result<()> {
        let mut output = Vec::new();
        let source = std::fs::read_to_string(&*ast.path)?;
        format_to(&ast, &source, &mut output)?;
        std::fs::write(&*ast.path, output)?;
        Ok(())
    }

    if options.fmt {
        for parsed in parsed {
            format_ast(parsed)?;
        }
    } else if options.fmt_current {
        let ast = parsed.into_iter().next().unwrap();
        let source = std::fs::read_to_string(&*ast.path)?;
        format_to(&ast, &source, out)?;
    } else {
        let mut codegen = codegen::Codegen::default();
        codegen.files = parsed;

        codegen.generate();
        if options.dump_asm {
            codegen.disasm(out)?;
        } else {
            let mut buf = Vec::new();
            codegen.assemble(&mut buf);
            out.write_all(&buf)?;
        }
    }

    Ok(())
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
        log::trc!(
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
        log::trc!(
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
        log::trc!(
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

#[cfg(test)]
mod test {
    use std::sync::Arc;

    #[test]
    fn task_queue_sanity() {
        let queue = Arc::new(super::TaskQueue::new(1000));

        let threads = (0..10)
            .map(|_| {
                let queue = queue.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        queue.extend([queue.pop().unwrap()]);
                    }
                })
            })
            .collect::<Vec<_>>();

        queue.extend(0..5);

        for t in threads {
            t.join().unwrap();
        }
    }
}
