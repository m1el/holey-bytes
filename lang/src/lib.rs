#![feature(
    iter_array_chunks,
    assert_matches,
    let_chains,
    if_let_guard,
    macro_metavar_expr,
    anonymous_lifetime_in_impl_trait,
    core_intrinsics,
    never_type,
    unwrap_infallible,
    slice_partition_dedup,
    portable_simd,
    iter_collect_into,
    new_uninit,
    ptr_metadata,
    slice_ptr_get,
    slice_take,
    map_try_insert,
    extract_if,
    ptr_internals,
    iter_intersperse,
    str_from_raw_parts,
    ptr_sub_ptr,
    slice_from_ptr_range,
    is_sorted,
    iter_next_chunk,
    pointer_is_aligned_to,
    maybe_uninit_fill
)]
#![warn(clippy::dbg_macro)]
#![expect(stable_features, internal_features)]
#![no_std]

#[cfg(feature = "std")]
pub use fs::*;
pub use utils::Ent;
use {
    self::{
        lexer::TokenKind,
        parser::{idfl, CommentOr, Expr, ExprRef, Pos},
        ty::{ArrayLen, Builtin, Module},
        utils::EntVec,
    },
    alloc::{string::String, vec::Vec},
    core::{cell::Cell, fmt::Display, ops::Range},
    hashbrown::hash_map,
};

#[macro_use]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(test)]
const README: &str = include_str!("../README.md");

#[macro_export]
macro_rules! run_tests {
    ($runner:path: $($name:ident;)*) => {$(
        #[test]
        fn $name() {
            $crate::run_test(core::any::type_name_of_val(&$name), stringify!($name), $crate::README, $runner);
        }
    )*};
}

pub mod fmt;
#[cfg(any(feature = "std", test))]
pub mod fs;
pub mod fuzz;
pub mod lexer;
pub mod parser;
pub mod regalloc;
pub mod son;

mod utils;

mod debug {
    pub fn panicking() -> bool {
        #[cfg(feature = "std")]
        {
            std::thread::panicking()
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    #[cfg(all(debug_assertions, feature = "std"))]
    pub type Trace = std::rc::Rc<std::backtrace::Backtrace>;
    #[cfg(not(all(debug_assertions, feature = "std")))]
    pub type Trace = ();

    pub fn trace() -> Trace {
        #[cfg(all(debug_assertions, feature = "std"))]
        {
            std::rc::Rc::new(std::backtrace::Backtrace::capture())
        }
        #[cfg(not(all(debug_assertions, feature = "std")))]
        {}
    }
}

pub mod reg {
    pub const STACK_PTR: Reg = 254;
    pub const ZERO: Reg = 0;
    pub const RET: Reg = 1;
    pub const RET_ADDR: Reg = 31;

    pub type Reg = u8;
}

mod ctx_map {
    use core::hash::BuildHasher;

    pub type Hash = u64;
    pub type HashBuilder = core::hash::BuildHasherDefault<IdentityHasher>;

    #[derive(Default)]
    pub struct IdentityHasher(u64);

    impl core::hash::Hasher for IdentityHasher {
        fn finish(&self) -> u64 {
            self.0
        }

        fn write(&mut self, _: &[u8]) {
            unimplemented!()
        }

        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    #[derive(Clone)]
    pub struct Key<T> {
        pub value: T,
        pub hash: Hash,
    }

    impl<T> core::hash::Hash for Key<T> {
        fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
            state.write_u64(self.hash);
        }
    }

    pub trait CtxEntry {
        type Ctx: ?Sized;
        type Key<'a>: Eq + core::hash::Hash;

        fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a>;
    }

    #[derive(Clone)]
    pub struct CtxMap<T> {
        inner: hashbrown::HashMap<Key<T>, (), HashBuilder>,
    }

    impl<T> Default for CtxMap<T> {
        fn default() -> Self {
            Self { inner: Default::default() }
        }
    }

    impl<T: CtxEntry> CtxMap<T> {
        pub fn entry<'a, 'b>(
            &'a mut self,
            key: T::Key<'b>,
            ctx: &'b T::Ctx,
        ) -> (hashbrown::hash_map::RawEntryMut<'a, Key<T>, (), HashBuilder>, Hash) {
            let hash = crate::FnvBuildHasher::default().hash_one(&key);
            (self.inner.raw_entry_mut().from_hash(hash, |k| k.value.key(ctx) == key), hash)
        }

        pub fn get<'a>(&self, key: T::Key<'a>, ctx: &'a T::Ctx) -> Option<&T> {
            let hash = crate::FnvBuildHasher::default().hash_one(&key);
            self.inner
                .raw_entry()
                .from_hash(hash, |k| k.value.key(ctx) == key)
                .map(|(k, _)| &k.value)
        }

        pub fn clear(&mut self) {
            self.inner.clear();
        }

        pub fn remove(&mut self, value: &T, ctx: &T::Ctx) -> Option<T> {
            let (entry, _) = self.entry(value.key(ctx), ctx);
            match entry {
                hashbrown::hash_map::RawEntryMut::Occupied(o) => Some(o.remove_entry().0.value),
                hashbrown::hash_map::RawEntryMut::Vacant(_) => None,
            }
        }

        pub fn insert<'a>(&mut self, key: T::Key<'a>, value: T, ctx: &'a T::Ctx) {
            let (entry, hash) = self.entry(key, ctx);
            match entry {
                hashbrown::hash_map::RawEntryMut::Occupied(_) => unreachable!(),
                hashbrown::hash_map::RawEntryMut::Vacant(v) => {
                    _ = v.insert(Key { hash, value }, ())
                }
            }
        }

        pub fn get_or_insert<'a>(
            &mut self,
            key: T::Key<'a>,
            ctx: &'a mut T::Ctx,
            with: impl FnOnce(&'a mut T::Ctx) -> T,
        ) -> &mut T {
            let (entry, hash) = self.entry(key, unsafe { &mut *(&mut *ctx as *mut _) });
            match entry {
                hashbrown::hash_map::RawEntryMut::Occupied(o) => &mut o.into_key_value().0.value,
                hashbrown::hash_map::RawEntryMut::Vacant(v) => {
                    &mut v.insert(Key { hash, value: with(ctx) }, ()).0.value
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Ident(u32);

impl Ident {
    pub const INVALID: Self = Self(u32::MAX);
    const LEN_BITS: u32 = 6;

    pub fn len(self) -> u32 {
        self.0 & ((1 << Self::LEN_BITS) - 1)
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn is_null(self) -> bool {
        (self.0 >> Self::LEN_BITS) == 0
    }

    pub fn pos(self) -> u32 {
        (self.0 >> Self::LEN_BITS).saturating_sub(1)
    }

    pub fn new(pos: u32, len: u32) -> Option<Self> {
        (len < (1 << Self::LEN_BITS)).then_some(((pos + 1) << Self::LEN_BITS) | len).map(Self)
    }

    pub fn range(self) -> core::ops::Range<usize> {
        let (len, pos) = (self.len() as usize, self.pos() as usize);
        pos..pos + len
    }

    fn builtin(builtin: Builtin) -> Ident {
        Self(builtin.index() as _)
    }
}

pub mod ty {
    use {
        crate::{
            lexer::TokenKind,
            parser::{self, Pos},
            utils::Ent,
            Ident, Size, Types,
        },
        core::{num::NonZeroU32, ops::Range},
    };

    pub type ArrayLen = u32;

    impl Func {
        pub const ECA: Func = Func(u32::MAX);
        pub const MAIN: Func = Func(u32::MIN);
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord)]
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

        pub fn range(self) -> Range<usize> {
            let start = self.0 as usize >> Self::LEN_BITS;
            start..start + self.len()
        }

        pub fn len(self) -> usize {
            self.0 as usize & Self::LEN_MASK
        }

        pub fn is_empty(self) -> bool {
            self.len() == 0
        }

        pub fn empty() -> Self {
            Self(0)
        }

        pub fn args(self) -> ArgIter {
            ArgIter(self.range())
        }
    }

    pub struct ArgIter(Range<usize>);

    pub enum Arg {
        Type(Id),
        Value(Id),
    }

    impl ArgIter {
        pub(crate) fn next(&mut self, tys: &Types) -> Option<Arg> {
            let ty = tys.ins.args[self.0.next()?];
            if ty == Id::TYPE {
                return Some(Arg::Type(tys.ins.args[self.0.next().unwrap()]));
            }
            Some(Arg::Value(ty))
        }

        pub(crate) fn next_value(&mut self, tys: &Types) -> Option<Id> {
            loop {
                match self.next(tys)? {
                    Arg::Type(_) => continue,
                    Arg::Value(id) => break Some(id),
                }
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    pub struct Id(NonZeroU32);

    impl crate::ctx_map::CtxEntry for Id {
        type Ctx = crate::TypeIns;
        type Key<'a> = crate::SymKey<'a>;

        fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
            match self.expand() {
                Kind::Struct(s) => {
                    let st = &ctx.structs[s];
                    debug_assert_ne!(st.pos, Pos::MAX);
                    crate::SymKey::Struct(st.file, st.pos, st.captures)
                }
                Kind::Ptr(p) => crate::SymKey::Pointer(&ctx.ptrs[p]),
                Kind::Opt(p) => crate::SymKey::Optional(&ctx.opts[p]),
                Kind::Func(f) => {
                    let fc = &ctx.funcs[f];
                    if let Some(base) = fc.base {
                        // TODO: merge base and sig
                        crate::SymKey::FuncInst(base, fc.sig.unwrap().args)
                    } else {
                        crate::SymKey::Decl(fc.file, fc.name)
                    }
                }
                Kind::Global(g) => {
                    let gb = &ctx.globals[g];
                    crate::SymKey::Decl(gb.file, gb.name)
                }
                Kind::Slice(s) => crate::SymKey::Array(&ctx.slices[s]),
                Kind::Module(_) | Kind::Builtin(_) => {
                    crate::SymKey::Decl(Module::default(), Ident::INVALID)
                }
                Kind::Const(c) => crate::SymKey::Constant(&ctx.consts[c]),
            }
        }
    }

    impl Default for Id {
        fn default() -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(UNDECLARED) })
        }
    }

    impl Id {
        pub const DINT: Self = Self::UINT;

        pub fn bin_ret(self, op: TokenKind) -> Id {
            use TokenKind as T;
            match op {
                T::Lt | T::Gt | T::Le | T::Ge | T::Ne | T::Eq => Id::BOOL,
                _ => self,
            }
        }

        pub fn is_float(self) -> bool {
            matches!(self.repr(), F32 | F64) || self.is_never()
        }

        pub fn is_signed(self) -> bool {
            matches!(self.repr(), I8..=INT) || self.is_never()
        }

        pub fn is_unsigned(self) -> bool {
            matches!(self.repr(), U8..=UINT) || self.is_never()
        }

        pub fn is_integer(self) -> bool {
            matches!(self.repr(), U8..=INT) || self.is_never()
        }

        pub fn is_never(self) -> bool {
            self == Self::NEVER
        }

        pub fn strip_pointer(self) -> Self {
            match self.expand() {
                Kind::Ptr(_) => Id::UINT,
                _ => self,
            }
        }

        pub fn is_pointer(self) -> bool {
            matches!(self.expand(), Kind::Ptr(_)) || self.is_never()
        }

        pub fn is_optional(self) -> bool {
            matches!(self.expand(), Kind::Opt(_)) || self.is_never()
        }

        pub fn try_upcast(self, ob: Self) -> Option<Self> {
            let (oa, ob) = (Self(self.0.min(ob.0)), Self(self.0.max(ob.0)));
            let (a, b) = (oa.strip_pointer(), ob.strip_pointer());
            Some(match () {
                _ if oa == Id::NEVER => ob,
                _ if ob == Id::NEVER => oa,
                _ if oa == ob => oa,
                _ if ob.is_optional() => ob,
                _ if oa.is_pointer() && ob.is_pointer() => return None,
                _ if a.is_signed() && b.is_signed() || a.is_unsigned() && b.is_unsigned() => ob,
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 < b.repr() - I8 => ob,
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 > b.repr() - I8 => oa,
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

        pub(crate) fn simple_size(&self) -> Option<Size> {
            Some(match self.expand() {
                Kind::Ptr(_) => 8,
                Kind::Builtin(Builtin(VOID)) => 0,
                Kind::Builtin(Builtin(NEVER)) => 0,
                Kind::Builtin(Builtin(INT | UINT | F64)) => 8,
                Kind::Builtin(Builtin(I32 | U32 | TYPE | F32)) => 4,
                Kind::Builtin(Builtin(I16 | U16)) => 2,
                Kind::Builtin(Builtin(I8 | U8 | BOOL)) => 1,
                _ => return None,
            })
        }

        pub(crate) fn extend(self) -> Self {
            if self.is_signed() {
                Self::INT
            } else if self.is_pointer() {
                self
            } else {
                Self::UINT
            }
        }

        pub(crate) fn loc(&self, tys: &Types) -> Loc {
            match self.expand() {
                Kind::Opt(o)
                    if let ty = tys.ins.opts[o].base
                        && ty.loc(tys) == Loc::Reg
                        && (ty.is_pointer() || tys.size_of(ty) < 8) =>
                {
                    Loc::Reg
                }
                Kind::Ptr(_) | Kind::Builtin(_) => Loc::Reg,
                Kind::Struct(_) if tys.size_of(*self) == 0 => Loc::Reg,
                Kind::Struct(_) | Kind::Slice(_) | Kind::Opt(_) => Loc::Stack,
                Kind::Func(_) | Kind::Global(_) | Kind::Module(_) | Kind::Const(_) => {
                    unreachable!()
                }
            }
        }

        pub(crate) fn has_pointers(&self, tys: &Types) -> bool {
            match self.expand() {
                Kind::Struct(s) => tys.struct_fields(s).iter().any(|f| f.ty.has_pointers(tys)),
                Kind::Ptr(_) => true,
                Kind::Slice(s) => tys.ins.slices[s].len == ArrayLen::MAX,
                _ => false,
            }
        }
    }

    #[derive(PartialEq, Eq, Clone, Copy)]
    pub enum Loc {
        Reg,
        Stack,
    }

    impl From<u64> for Id {
        fn from(id: u64) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(id as _) })
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
            $(const $name: u32 = ${index(0)} + 1;)*

            mod __lc_names {
                use super::*;
                $(pub const $name: &str = unsafe {
                    const LCL: &[u8] = unsafe {
                        &array_to_lower_case(
                            *(stringify!($name).as_ptr() as *const [u8; stringify!($name).len()])
                        )
                    };
                    core::str::from_utf8_unchecked(LCL)
                };)*
            }

            impl Id {
                $(pub const $name: Self = Kind::Builtin(Builtin($name)).compress();)*
            }

            impl Kind {
                $(pub const $name: Self = Kind::Builtin(Builtin($name));)*
            }

            pub fn from_str(name: &str) -> Option<Builtin> {
                match name {
                    $(__lc_names::$name => Some(Builtin($name)),)*
                    _ => None,
                }
            }

            pub fn to_str(ty: Builtin) -> &'static str {
                match ty.0 {
                    $($name => __lc_names::$name,)*
                    v => unreachable!("invalid type: {}", v),
                }
            }
        };
    }

    builtin_type! {
        UNDECLARED;
        LEFT_UNREACHABLE;
        RIGHT_UNREACHABLE;
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
        F32;
        F64;
    }

    macro_rules! type_kind {
        ($(#[$meta:meta])* $vis:vis enum $name:ident {$( $variant:ident, )*}) => {
            crate::utils::decl_ent! {
                $(pub struct $variant(u32);)*
            }

            $(#[$meta])*
            $vis enum $name {
                $($variant($variant),)*
            }

            impl $name {
                const FLAG_BITS: u32 = (${count($variant)} as u32).next_power_of_two().ilog2();
                const FLAG_OFFSET: u32 = core::mem::size_of::<Id>() as u32 * 8 - Self::FLAG_BITS;
                const INDEX_MASK: u32 = (1 << (32 - Self::FLAG_BITS)) - 1;

                $vis fn from_ty(ty: Id) -> Self {
                    let (flag, index) = (ty.repr() >> Self::FLAG_OFFSET, ty.repr() & Self::INDEX_MASK);
                    match flag {
                        $(${index(0)} => Self::$variant($variant(index)),)*
                        i => unreachable!("{i}"),
                    }
                }

                $vis const fn compress(self) -> Id {
                    let (index, flag) = match self {
                        $(Self::$variant(index) => (index.0, ${index(0)}),)*
                    };
                   Id(unsafe { NonZeroU32::new_unchecked((flag << Self::FLAG_OFFSET) | index) })
                }
            }

            $(
                impl From<$variant> for $name {
                    fn from(value: $variant) -> Self {
                        Self::$variant(value)
                    }
                }

                impl From<$variant> for Id {
                    fn from(value: $variant) -> Self {
                        $name::$variant(value).compress()
                    }
                }
            )*
        };
    }

    type_kind! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Kind {
            Builtin,
            Struct,
            Ptr,
            Slice,
            Opt,
            Func,
            Global,
            Module,
            Const,
        }
    }

    impl Module {
        pub const MAIN: Self = Self(0);
    }

    impl Default for Module {
        fn default() -> Self {
            Self(u32::MAX)
        }
    }

    impl TryFrom<Ident> for Builtin {
        type Error = ();

        fn try_from(value: Ident) -> Result<Self, Self::Error> {
            if value.is_null() {
                Ok(Self(value.len()))
            } else {
                Err(())
            }
        }
    }

    impl Default for Kind {
        fn default() -> Self {
            Id::UNDECLARED.expand()
        }
    }

    pub struct Display<'a> {
        tys: &'a super::Types,
        files: &'a [parser::Ast],
        ty: Id,
    }

    impl<'a> Display<'a> {
        pub fn new(tys: &'a super::Types, files: &'a [parser::Ast], ty: Id) -> Self {
            Self { tys, files, ty }
        }

        pub fn rety(&self, ty: Id) -> Self {
            Self::new(self.tys, self.files, ty)
        }
    }

    impl core::fmt::Display for Display<'_> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            use Kind as TK;
            match TK::from_ty(self.ty) {
                TK::Module(idx) => {
                    f.write_str("@use(\"")?;
                    self.files[idx.index()].path.fmt(f)?;
                    f.write_str(")[")?;
                    idx.fmt(f)?;
                    f.write_str("]")
                }
                TK::Builtin(ty) => f.write_str(to_str(ty)),
                TK::Opt(ty) => {
                    f.write_str("?")?;
                    self.rety(self.tys.ins.opts[ty].base).fmt(f)
                }
                TK::Ptr(ty) => {
                    f.write_str("^")?;
                    self.rety(self.tys.ins.ptrs[ty].base).fmt(f)
                }
                TK::Struct(idx) => {
                    let record = &self.tys.ins.structs[idx];
                    if record.name.is_null() {
                        f.write_str("[")?;
                        idx.fmt(f)?;
                        f.write_str("]{")?;
                        for (i, &super::Field { name, ty }) in
                            self.tys.struct_fields(idx).iter().enumerate()
                        {
                            if i != 0 {
                                f.write_str(", ")?;
                            }
                            f.write_str(self.tys.names.ident_str(name))?;
                            f.write_str(": ")?;
                            self.rety(ty).fmt(f)?;
                        }
                        f.write_str("}")
                    } else {
                        let file = &self.files[record.file.index()];
                        f.write_str(file.ident_str(record.name))
                    }
                }
                TK::Func(idx) => {
                    f.write_str("fn")?;
                    idx.fmt(f)
                }
                TK::Global(idx) => {
                    let global = &self.tys.ins.globals[idx];
                    let file = &self.files[global.file.index()];
                    f.write_str(file.ident_str(global.name))
                }
                TK::Slice(idx) => {
                    let array = self.tys.ins.slices[idx];
                    f.write_str("[")?;
                    self.rety(array.elem).fmt(f)?;
                    if array.len != ArrayLen::MAX {
                        f.write_str("; ")?;
                        array.len.fmt(f)?;
                    }
                    f.write_str("]")
                }
                TK::Const(idx) => {
                    let cnst = &self.tys.ins.consts[idx];
                    let file = &self.files[cnst.file.index()];
                    f.write_str(file.ident_str(cnst.name))
                }
            }
        }
    }
}

type Offset = u32;
type Size = u32;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum SymKey<'a> {
    Pointer(&'a Ptr),
    Optional(&'a Opt),
    Struct(Module, Pos, ty::Tuple),
    FuncInst(ty::Func, ty::Tuple),
    Decl(Module, Ident),
    Array(&'a Array),
    Constant(&'a Const),
}

#[derive(Clone, Copy)]
pub struct Sig {
    args: ty::Tuple,
    ret: ty::Id,
}

#[derive(Default, Clone, Copy)]
struct Func {
    file: Module,
    name: Ident,
    base: Option<ty::Func>,
    expr: ExprRef,
    sig: Option<Sig>,
    is_inline: bool,
    comp_state: [CompState; 2],
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum CompState {
    #[default]
    Dead,
    Queued(usize),
    Compiled,
}

#[derive(Clone, Copy)]
struct TypedReloc {
    target: ty::Id,
    reloc: Reloc,
}

#[derive(Clone, Default)]
struct Global {
    file: Module,
    name: Ident,
    ty: ty::Id,
    data: Vec<u8>,
}

#[derive(PartialEq, Eq, Hash)]
pub struct Const {
    ast: ExprRef,
    name: Ident,
    file: Module,
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
    name: Ident,
    ty: ty::Id,
}

#[derive(Default)]
struct Struct {
    name: Ident,
    pos: Pos,
    file: Module,
    size: Cell<Size>,
    align: Cell<u8>,
    captures: ty::Tuple,
    explicit_alignment: Option<u8>,
    field_start: u32,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Opt {
    base: ty::Id,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Ptr {
    base: ty::Id,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Array {
    elem: ty::Id,
    len: ArrayLen,
}
impl Array {
    fn len(&self) -> Option<usize> {
        (self.len != ArrayLen::MAX).then_some(self.len as usize)
    }
}

#[derive(Clone, Copy)]
enum PLoc {
    Reg(u8, u16),
    WideReg(u8, u16),
    Ref(u8, u32),
}

struct ParamAlloc(Range<u8>);

impl ParamAlloc {
    pub fn next(&mut self, ty: ty::Id, tys: &Types) -> Option<PLoc> {
        Some(match tys.size_of(ty) {
            0 => return None,
            size @ 1..=8 => PLoc::Reg(self.0.next().unwrap(), size as _),
            size @ 9..=16 => PLoc::WideReg(self.0.next_chunk::<2>().unwrap()[0], size as _),
            size @ 17.. => PLoc::Ref(self.0.next().unwrap(), size),
        })
    }
}

impl ctx_map::CtxEntry for Ident {
    type Ctx = str;
    type Key<'a> = &'a str;

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        unsafe { ctx.get_unchecked(self.range()) }
    }
}

#[derive(Default)]
struct IdentInterner {
    lookup: ctx_map::CtxMap<Ident>,
    strings: String,
}

impl IdentInterner {
    fn intern(&mut self, ident: &str) -> Ident {
        let (entry, hash) = self.lookup.entry(ident, &self.strings);
        match entry {
            hash_map::RawEntryMut::Occupied(o) => o.get_key_value().0.value,
            hash_map::RawEntryMut::Vacant(v) => {
                let id = Ident::new(self.strings.len() as _, ident.len() as _).unwrap();
                self.strings.push_str(ident);
                v.insert(ctx_map::Key { hash, value: id }, ());
                id
            }
        }
    }

    fn ident_str(&self, ident: Ident) -> &str {
        &self.strings[ident.range()]
    }

    fn project(&self, ident: &str) -> Option<Ident> {
        self.lookup.get(ident, &self.strings).copied()
    }

    fn clear(&mut self) {
        self.lookup.clear();
        self.strings.clear()
    }
}

#[derive(Default)]
struct TypesTmp {
    fields: Vec<Field>,
    args: Vec<ty::Id>,
}

#[derive(Default)]
pub struct TypeIns {
    args: Vec<ty::Id>,
    fields: Vec<Field>,
    funcs: EntVec<ty::Func, Func>,
    globals: EntVec<ty::Global, Global>,
    consts: EntVec<ty::Const, Const>,
    structs: EntVec<ty::Struct, Struct>,
    ptrs: EntVec<ty::Ptr, Ptr>,
    opts: EntVec<ty::Opt, Opt>,
    slices: EntVec<ty::Slice, Array>,
}

struct FTask {
    file: Module,
    id: ty::Func,
    ct: bool,
}

struct StringRef(ty::Global);

impl ctx_map::CtxEntry for StringRef {
    type Ctx = EntVec<ty::Global, Global>;
    type Key<'a> = &'a [u8];

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        &ctx[self.0].data
    }
}

#[derive(Default)]
pub struct Types {
    syms: ctx_map::CtxMap<ty::Id>,
    names: IdentInterner,
    strings: ctx_map::CtxMap<StringRef>,
    ins: TypeIns,
    tmp: TypesTmp,
    tasks: Vec<Option<FTask>>,
}

trait TypeParser {
    fn tys(&mut self) -> &mut Types;
    fn on_reuse(&mut self, existing: ty::Id);
    fn find_local_ty(&mut self, name: Ident) -> Option<ty::Id>;
    fn eval_const(&mut self, file: Module, expr: &Expr, ty: ty::Id) -> u64;
    fn eval_global(&mut self, file: Module, name: Ident, expr: &Expr) -> ty::Id;
    fn infer_type(&mut self, expr: &Expr) -> ty::Id;
    fn report(&self, file: Module, pos: Pos, msg: impl Display) -> ty::Id;

    fn find_type(
        &mut self,
        pos: Pos,
        from_file: Module,
        file: Module,
        id: Result<Ident, &str>,
        files: &[parser::Ast],
    ) -> ty::Id {
        let ty = if let Ok(id) = id
            && let Some(ty) = self.find_local_ty(id)
        {
            ty
        } else if let Ok(id) = id
            && let tys = self.tys()
            && let Some(&ty) = tys.syms.get(SymKey::Decl(file, id), &tys.ins)
        {
            self.on_reuse(ty);
            ty
        } else {
            let f = &files[file.index()];

            let Some((expr @ Expr::BinOp { left, right, .. }, name)) = f.find_decl(id) else {
                return match id {
                    Ok(_) => {
                        debug_assert_eq!(from_file, file);
                        self.report(file, pos, "somehow this was not found")
                    }
                    Err("main") => self.report(
                        from_file,
                        pos,
                        format_args!(
                            "missing main function in '{}', compiler can't \
                            emmit libraries since such concept is not defined \
                            (minimal main function: `main := fn(): void {{}}`)",
                            f.path
                        ),
                    ),
                    Err(name) => {
                        self.report(from_file, pos, format_args!("undefined indentifier: {name}"))
                    }
                };
            };

            let tys = self.tys();
            if let Some(&ty) = tys.syms.get(SymKey::Decl(file, name), &tys.ins) {
                ty
            } else {
                let (is_ct, ty) = left
                    .find_pattern_path(name, right, |right, is_ct| {
                        (
                            is_ct,
                            if is_ct && !matches!(right, Expr::Closure { .. }) {
                                self.tys()
                                    .ins
                                    .consts
                                    .push(Const { ast: ExprRef::new(expr), name, file })
                                    .into()
                            } else {
                                self.parse_ty(file, right, Some(name), files)
                            },
                        )
                    })
                    .unwrap_or_else(|_| unreachable!());
                let tys = self.tys();
                if let ty::Kind::Func(f) = ty.expand()
                    && is_ct
                {
                    tys.ins.funcs[f].is_inline = true;
                }
                tys.syms.insert(SymKey::Decl(file, name), ty, &tys.ins);
                ty
            }
        };

        let tys = self.tys();
        if let ty::Kind::Global(g) = ty.expand() {
            let g = &tys.ins.globals[g];
            if g.ty == ty::Id::TYPE {
                return ty::Id::from(
                    u32::from_ne_bytes(g.data.as_slice().try_into().unwrap()) as u64
                );
            }
        }
        ty
    }

    /// returns none if comptime eval is required
    fn parse_ty(
        &mut self,
        file: Module,
        expr: &Expr,
        name: Option<Ident>,
        files: &[parser::Ast],
    ) -> ty::Id {
        match *expr {
            Expr::Mod { id, .. } => ty::Kind::Module(id).compress(),
            Expr::UnOp { op: TokenKind::Xor, val, .. } => {
                let base = self.parse_ty(file, val, None, files);
                self.tys().make_ptr(base)
            }
            Expr::UnOp { op: TokenKind::Que, val, .. } => {
                let base = self.parse_ty(file, val, None, files);
                self.tys().make_opt(base)
            }
            Expr::Ident { id, .. } if let Ok(bt) = ty::Builtin::try_from(id) => bt.into(),
            Expr::Ident { id, pos, .. } => self.find_type(pos, file, file, Ok(id), files),
            Expr::Field { target, pos, name }
                if let ty::Kind::Module(inside) =
                    self.parse_ty(file, target, None, files).expand() =>
            {
                self.find_type(pos, file, inside, Err(name), files)
            }
            Expr::Directive { name: "TypeOf", args: [expr], .. } => self.infer_type(expr),
            Expr::Slice { size: None, item, .. } => {
                let ty = self.parse_ty(file, item, None, files);
                self.tys().make_array(ty, ArrayLen::MAX)
            }
            Expr::Slice { size: Some(&Expr::Number { value, .. }), item, .. } => {
                let ty = self.parse_ty(file, item, None, files);
                self.tys().make_array(ty, value as _)
            }
            Expr::Slice { size, item, .. } => {
                let ty = self.parse_ty(file, item, None, files);
                let len = size
                    .map_or(ArrayLen::MAX, |expr| self.eval_const(file, expr, ty::Id::U32) as _);
                self.tys().make_array(ty, len)
            }
            Expr::Struct { pos, fields, packed, captured, .. } => {
                let captures_start = self.tys().tmp.args.len();
                for &cp in captured {
                    let ty = self.find_local_ty(cp).expect("TODO");
                    self.tys().tmp.args.push(ty);
                }
                let captured = self.tys().pack_args(captures_start).expect("TODO");

                let sym = SymKey::Struct(file, pos, captured);
                let tys = self.tys();
                if let Some(&ty) = tys.syms.get(sym, &tys.ins) {
                    return ty;
                }

                let prev_tmp = self.tys().tmp.fields.len();
                for field in fields.iter().filter_map(CommentOr::or) {
                    let ty = self.parse_ty(file, &field.ty, None, files);
                    let field = Field { name: self.tys().names.intern(field.name), ty };
                    self.tys().tmp.fields.push(field);
                }

                let tys = self.tys();
                let ty = tys
                    .ins
                    .structs
                    .push(Struct {
                        file,
                        pos,
                        name: name.unwrap_or_default(),
                        field_start: tys.ins.fields.len() as _,
                        explicit_alignment: packed.then_some(1),
                        ..Default::default()
                    })
                    .into();

                tys.ins.fields.extend(tys.tmp.fields.drain(prev_tmp..));

                tys.syms.insert(sym, ty, &tys.ins);
                ty
            }
            Expr::Closure { pos, args, ret, .. } if let Some(name) = name => {
                let func = Func {
                    file,
                    name,
                    sig: 'b: {
                        let arg_base = self.tys().tmp.args.len();
                        for arg in args {
                            let sym = parser::find_symbol(&files[file.index()].symbols, arg.id);
                            if sym.flags & idfl::COMPTIME != 0 {
                                self.tys().tmp.args.truncate(arg_base);
                                break 'b None;
                            }
                            let ty = self.parse_ty(file, &arg.ty, None, files);
                            self.tys().tmp.args.push(ty);
                        }

                        let Some(args) = self.tys().pack_args(arg_base) else {
                            return self.report(file, pos, "function has too many argumnets");
                        };
                        let ret = self.parse_ty(file, ret, None, files);

                        Some(Sig { args, ret })
                    },
                    expr: ExprRef::new(expr),
                    ..Default::default()
                };

                self.tys().ins.funcs.push(func).into()
            }
            _ if let Some(name) = name => self.eval_global(file, name, expr),
            _ => ty::Id::from(self.eval_const(file, expr, ty::Id::TYPE)),
        }
    }
}

impl Types {
    fn struct_field_range(&self, strct: ty::Struct) -> Range<usize> {
        let start = self.ins.structs[strct].field_start as usize;
        let end =
            self.ins.structs.next(strct).map_or(self.ins.fields.len(), |s| s.field_start as usize);
        start..end
    }

    fn pack_args(&mut self, arg_base: usize) -> Option<ty::Tuple> {
        let base = self.ins.args.len();
        self.ins.args.extend(self.tmp.args.drain(arg_base..));
        let needle = &self.ins.args[base..];
        if needle.is_empty() {
            return Some(ty::Tuple::empty());
        }
        let len = needle.len();
        // FIXME: maybe later when this becomes a bottleneck we use more
        // efficient search (SIMD?, indexing?)
        let sp = self.ins.args.windows(needle.len()).position(|val| val == needle).unwrap();
        self.ins.args.truncate((sp + needle.len()).max(base));
        ty::Tuple::new(sp, len)
    }

    fn struct_fields(&self, strct: ty::Struct) -> &[Field] {
        &self.ins.fields[self.struct_field_range(strct)]
    }

    fn parama(&self, ret: impl Into<ty::Id>) -> (Option<PLoc>, ParamAlloc) {
        let mut iter = ParamAlloc(1..12);
        let ret = iter.next(ret.into(), self);
        iter.0.start += ret.is_none() as u8;
        (ret, iter)
    }

    fn make_opt(&mut self, base: ty::Id) -> ty::Id {
        self.make_generic_ty(Opt { base }, |ins| &mut ins.opts, |e| SymKey::Optional(e))
    }

    fn make_ptr(&mut self, base: ty::Id) -> ty::Id {
        self.make_generic_ty(Ptr { base }, |ins| &mut ins.ptrs, |e| SymKey::Pointer(e))
    }

    fn make_array(&mut self, elem: ty::Id, len: ArrayLen) -> ty::Id {
        self.make_generic_ty(Array { elem, len }, |ins| &mut ins.slices, |e| SymKey::Array(e))
    }

    fn make_generic_ty<K: Ent + Into<ty::Id>, T: Copy>(
        &mut self,
        ty: T,
        get_col: fn(&mut TypeIns) -> &mut EntVec<K, T>,
        key: fn(&T) -> SymKey,
    ) -> ty::Id {
        *self.syms.get_or_insert(key(&{ ty }), &mut self.ins, |ins| get_col(ins).push(ty).into())
    }

    fn size_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Slice(arr) => {
                let arr = &self.ins.slices[arr];
                match arr.len {
                    0 => 0,
                    ArrayLen::MAX => 16,
                    len => self.size_of(arr.elem) * len,
                }
            }
            ty::Kind::Struct(stru) => {
                if self.ins.structs[stru].size.get() != 0 {
                    return self.ins.structs[stru].size.get();
                }

                let mut oiter = OffsetIter::new(stru, self);
                while oiter.next(self).is_some() {}
                self.ins.structs[stru].size.set(oiter.offset);
                oiter.offset
            }
            ty::Kind::Opt(opt) => {
                let base = self.ins.opts[opt].base;
                if self.nieche_of(base).is_some() {
                    self.size_of(base)
                } else {
                    self.size_of(base) + self.align_of(base)
                }
            }
            _ if let Some(size) = ty.simple_size() => size,
            ty => unimplemented!("size_of: {:?}", ty),
        }
    }

    fn align_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Struct(stru) => {
                if self.ins.structs[stru].align.get() != 0 {
                    return self.ins.structs[stru].align.get() as _;
                }
                let align = self.ins.structs[stru].explicit_alignment.map_or_else(
                    || {
                        self.struct_fields(stru)
                            .iter()
                            .map(|&Field { ty, .. }| self.align_of(ty))
                            .max()
                            .unwrap_or(1)
                    },
                    |a| a as _,
                );
                self.ins.structs[stru].align.set(align.try_into().unwrap());
                align
            }
            ty::Kind::Slice(arr) => {
                let arr = &self.ins.slices[arr];
                match arr.len {
                    ArrayLen::MAX => 8,
                    _ => self.align_of(arr.elem),
                }
            }
            _ => self.size_of(ty).max(1),
        }
    }

    fn base_of(&self, ty: ty::Id) -> Option<ty::Id> {
        match ty.expand() {
            ty::Kind::Ptr(p) => Some(self.ins.ptrs[p].base),
            _ => None,
        }
    }

    fn inner_of(&self, ty: ty::Id) -> Option<ty::Id> {
        match ty.expand() {
            ty::Kind::Opt(o) => Some(self.ins.opts[o].base),
            _ => None,
        }
    }

    fn opt_layout(&self, inner_ty: ty::Id) -> OptLayout {
        match self.nieche_of(inner_ty) {
            Some((_, flag_offset, flag_ty)) => {
                OptLayout { flag_ty, flag_offset, payload_offset: 0 }
            }
            None => OptLayout {
                flag_ty: ty::Id::BOOL,
                flag_offset: 0,
                payload_offset: self.align_of(inner_ty),
            },
        }
    }

    fn nieche_of(&self, ty: ty::Id) -> Option<(bool, Offset, ty::Id)> {
        match ty.expand() {
            ty::Kind::Ptr(_) => Some((false, 0, ty::Id::UINT)),
            // TODO: cache this
            ty::Kind::Struct(s) => OffsetIter::new(s, self).into_iter(self).find_map(|(f, off)| {
                self.nieche_of(f.ty).map(|(uninit, o, ty)| (uninit, o + off, ty))
            }),
            _ => None,
        }
    }

    fn find_struct_field(&self, s: ty::Struct, name: &str) -> Option<usize> {
        let name = self.names.project(name)?;
        self.struct_fields(s).iter().position(|f| f.name == name)
    }

    fn clear(&mut self) {
        self.syms.clear();
        self.names.clear();
        self.strings.clear();

        self.ins.funcs.clear();
        self.ins.args.clear();
        self.ins.globals.clear();
        self.ins.structs.clear();
        self.ins.fields.clear();
        self.ins.ptrs.clear();
        self.ins.slices.clear();

        debug_assert_eq!(self.tmp.fields.len(), 0);
        debug_assert_eq!(self.tmp.args.len(), 0);

        debug_assert_eq!(self.tasks.len(), 0);
    }
}

struct OptLayout {
    flag_ty: ty::Id,
    flag_offset: Offset,
    payload_offset: Offset,
}

struct OffsetIter {
    strct: ty::Struct,
    offset: Offset,
    fields: Range<usize>,
}

impl OffsetIter {
    fn new(strct: ty::Struct, tys: &Types) -> Self {
        Self { strct, offset: 0, fields: tys.struct_field_range(strct) }
    }

    fn offset_of(tys: &Types, idx: ty::Struct, field: &str) -> Option<(Offset, ty::Id)> {
        let field_id = tys.names.project(field)?;
        OffsetIter::new(idx, tys)
            .into_iter(tys)
            .find(|(f, _)| f.name == field_id)
            .map(|(f, off)| (off, f.ty))
    }

    fn next<'a>(&mut self, tys: &'a Types) -> Option<(&'a Field, Offset)> {
        let stru = &tys.ins.structs[self.strct];
        let field = &tys.ins.fields[self.fields.next()?];

        let align = stru.explicit_alignment.map_or_else(|| tys.align_of(field.ty), |a| a as u32);
        self.offset = (self.offset + align - 1) & !(align - 1);

        let off = self.offset;
        self.offset += tys.size_of(field.ty);
        Some((field, off))
    }

    fn next_ty(&mut self, tys: &Types) -> Option<(ty::Id, Offset)> {
        let (field, off) = self.next(tys)?;
        Some((field.ty, off))
    }

    fn into_iter(mut self, tys: &Types) -> impl Iterator<Item = (&Field, Offset)> {
        core::iter::from_fn(move || self.next(tys))
    }
}

type HashMap<K, V> = hashbrown::HashMap<K, V, FnvBuildHasher>;
type FnvBuildHasher = core::hash::BuildHasherDefault<FnvHasher>;

struct FnvHasher(u64);

impl core::hash::Hasher for FnvHasher {
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
    use std::{io::Write, path::PathBuf, string::ToString};

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
                    std::println!("{}", self.0);
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
fn test_parse_files(
    ident: &str,
    input: &str,
    ctx: &mut parser::Ctx,
) -> (Vec<parser::Ast>, Vec<Vec<u8>>) {
    use {
        self::parser::FileKind,
        std::{borrow::ToOwned, string::ToString},
    };

    fn find_block<'a>(mut input: &'a str, test_name: &str) -> &'a str {
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
    let mut embed_map = Vec::new();
    let mut last_start = 0;
    let mut last_module_name = "test.hb";
    for (i, m) in input.match_indices("// in module: ") {
        if last_module_name.ends_with(".hb") {
            fmt::test::format(ident, input[last_start..i].trim());
            module_map.push((last_module_name, &input[last_start..i]));
        } else {
            embed_map.push((last_module_name, &input[last_start..i]));
        }
        let (module_name, _) = input[i + m.len()..].split_once('\n').unwrap();
        last_module_name = module_name;
        last_start = i + m.len() + module_name.len() + 1;
    }
    if last_module_name.ends_with(".hb") {
        fmt::test::format(ident, input[last_start..].trim());
        module_map.push((last_module_name, &input[last_start..]));
    } else {
        embed_map.push((last_module_name, &input[last_start..]));
    }

    let mut loader = |path: &str, _: &str, kind| match kind {
        FileKind::Module => module_map
            .iter()
            .position(|&(name, _)| name == path)
            .ok_or("Module Not Found".to_string()),
        FileKind::Embed => embed_map
            .iter()
            .position(|&(name, _)| name == path)
            .ok_or("Embed Not Found".to_string()),
    };

    (
        module_map
            .iter()
            .map(|&(path, content)| parser::Ast::new(path, content.to_owned(), ctx, &mut loader))
            .collect(),
        embed_map.iter().map(|&(_, content)| content.to_owned().into_bytes()).collect(),
    )
}

fn endoce_string(
    literal: &str,
    str: &mut Vec<u8>,
    report: impl Fn(&core::str::Bytes, &str),
) -> Option<()> {
    let report = |bytes: &core::str::Bytes, msg: &_| {
        report(bytes, msg);
        None::<u8>
    };

    let decode_braces = |str: &mut Vec<u8>, bytes: &mut core::str::Bytes| {
        while let Some(b) = bytes.next()
            && b != b'}'
        {
            let c = bytes.next().or_else(|| report(bytes, "incomplete escape sequence"))?;
            let decode = |b: u8| {
                Some(match b {
                    b'0'..=b'9' => b - b'0',
                    b'a'..=b'f' => b - b'a' + 10,
                    b'A'..=b'F' => b - b'A' + 10,
                    _ => report(bytes, "expected hex digit or '}'")?,
                })
            };
            str.push(decode(b)? << 4 | decode(c)?);
        }

        Some(())
    };

    let mut bytes = literal.bytes();
    while let Some(b) = bytes.next() {
        if b != b'\\' {
            str.push(b);
            continue;
        }
        let b = match bytes.next().or_else(|| report(&bytes, "incomplete escape sequence"))? {
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'\\' => b'\\',
            b'\'' => b'\'',
            b'"' => b'"',
            b'0' => b'\0',
            b'{' => {
                decode_braces(str, &mut bytes);
                continue;
            }
            _ => report(&bytes, "unknown escape sequence, expected [nrt\\\"'{0]")?,
        };
        str.push(b);
    }

    if str.last() != Some(&0) {
        report(&bytes, "string literal must end with null byte (for now)");
    }

    Some(())
}

pub fn quad_sort<T>(mut slice: &mut [T], mut cmp: impl FnMut(&T, &T) -> core::cmp::Ordering) {
    while let Some(it) = slice.take_first_mut() {
        for ot in &mut *slice {
            if cmp(it, ot) == core::cmp::Ordering::Greater {
                core::mem::swap(it, ot);
            }
        }
    }
    debug_assert!(slice.is_sorted_by(|a, b| cmp(a, b) != core::cmp::Ordering::Greater));
}
