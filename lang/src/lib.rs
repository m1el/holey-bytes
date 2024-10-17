#![feature(
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
    is_sorted
)]
#![feature(pointer_is_aligned_to)]
#![warn(clippy::dbg_macro)]
#![allow(stable_features, internal_features)]
#![no_std]

#[cfg(feature = "std")]
pub use fs::*;
use {
    self::{
        ident::Ident,
        lexer::TokenKind,
        parser::{CommentOr, Expr, ExprRef, FileId, Pos},
        ty::ArrayLen,
    },
    alloc::{collections::BTreeMap, string::String, vec::Vec},
    core::{cell::Cell, ops::Range},
    hashbrown::hash_map,
    hbbytecode as instrs,
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

pub mod codegen;
pub mod fmt;
#[cfg(any(feature = "std", test))]
pub mod fs;
pub mod parser;
#[cfg(feature = "opts")]
pub mod son;

pub mod lexer;
#[cfg(feature = "opts")]
mod vc;

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

        #[cfg_attr(not(feature = "opts"), expect(dead_code))]
        pub fn clear(&mut self) {
            self.inner.clear();
        }

        #[cfg_attr(not(feature = "opts"), expect(dead_code))]
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

    #[cfg_attr(not(feature = "opts"), expect(dead_code))]
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

    pub fn new(pos: u32, len: u32) -> Option<u32> {
        (len < (1 << LEN_BITS)).then_some(((pos + 1) << LEN_BITS) | len)
    }

    pub fn range(ident: u32) -> core::ops::Range<usize> {
        let (len, pos) = (len(ident) as usize, pos(ident) as usize);
        pos..pos + len
    }
}

mod ty {
    use {
        crate::{
            ident,
            lexer::TokenKind,
            parser::{self, Pos},
        },
        core::{num::NonZeroU32, ops::Range},
    };

    pub type ArrayLen = u32;

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;
    pub type Slice = u32;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

        pub fn empty() -> Self {
            Self(0)
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
                    let st = &ctx.structs[s as usize];
                    debug_assert_ne!(st.pos, Pos::MAX);
                    crate::SymKey::Struct(st.file, st.pos)
                }
                Kind::Ptr(p) => crate::SymKey::Pointer(&ctx.ptrs[p as usize]),
                Kind::Func(f) => {
                    let fc = &ctx.funcs[f as usize];
                    if let Some(base) = fc.base {
                        crate::SymKey::FuncInst(base, fc.sig.unwrap().args)
                    } else {
                        crate::SymKey::Decl(fc.file, fc.name)
                    }
                }
                Kind::Global(g) => {
                    let gb = &ctx.globals[g as usize];
                    crate::SymKey::Decl(gb.file, gb.name)
                }
                Kind::Slice(s) => crate::SymKey::Array(&ctx.arrays[s as usize]),
                Kind::Module(_) | Kind::Builtin(_) => crate::SymKey::Decl(u32::MAX, u32::MAX),
            }
        }
    }

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

        pub fn try_upcast(self, ob: Self, kind: TyCheck) -> Option<Self> {
            let (oa, ob) = (Self(self.0.min(ob.0)), Self(self.0.max(ob.0)));
            let (a, b) = (oa.strip_pointer(), ob.strip_pointer());
            Some(match () {
                _ if oa == Self::from(NEVER) => ob,
                _ if ob == Self::from(NEVER) => oa,
                _ if oa == ob => oa,
                _ if oa.is_pointer() && ob.is_pointer() => return None,
                _ if a.is_signed() && b.is_signed() || a.is_unsigned() && b.is_unsigned() => ob,
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 < b.repr() - I8 => ob,
                _ if oa.is_integer() && ob.is_pointer() && kind == TyCheck::BinOp => ob,
                _ => return None,
            })
        }

        pub fn expand(self) -> Kind {
            Kind::from_ty(self)
        }

        pub const fn repr(self) -> u32 {
            self.0.get()
        }

        #[allow(unused)]
        pub fn is_struct(&self) -> bool {
            matches!(self.expand(), Kind::Struct(_))
        }
    }

    #[derive(PartialEq, Eq, Default, Debug, Clone, Copy)]
    pub enum TyCheck {
        BinOp,
        #[default]
        Assign,
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
                $(pub const $name: &str = unsafe {
                    const LCL: &[u8] = unsafe {
                        &array_to_lower_case(
                            *(stringify!($name).as_ptr() as *const [u8; stringify!($name).len()])
                        )
                    };
                    core::str::from_utf8_unchecked(LCL)
                };)*
            }

            #[allow(dead_code)]
            impl Id {
                $(pub const $name: Self = Kind::Builtin($name).compress();)*
            }

            pub fn from_str(name: &str) -> Option<Builtin> {
                match name {
                    $(__lc_names::$name => Some($name),)*
                    _ => None,
                }
            }

            pub fn to_str(ty: Builtin) -> &'static str {
                match ty {
                    $($name => __lc_names::$name,)*
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
                const FLAG_OFFSET: u32 = core::mem::size_of::<Id>() as u32 * 8 - Self::FLAG_BITS;
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

    impl core::fmt::Display for Display<'_> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            use Kind as TK;
            match TK::from_ty(self.ty) {
                TK::Module(idx) => {
                    f.write_str("@use(\"")?;
                    self.files[idx as usize].path.fmt(f)?;
                    f.write_str(")[")?;
                    idx.fmt(f)?;
                    f.write_str("]")
                }
                TK::Builtin(ty) => f.write_str(to_str(ty)),
                TK::Ptr(ty) => {
                    f.write_str("^")?;
                    self.rety(self.tys.ins.ptrs[ty as usize].base).fmt(f)
                }
                TK::Struct(idx) => {
                    let record = &self.tys.ins.structs[idx as usize];
                    if ident::is_null(record.name) {
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
                        let file = &self.files[record.file as usize];
                        f.write_str(file.ident_str(record.name))
                    }
                }
                TK::Func(idx) => {
                    f.write_str("fn")?;
                    idx.fmt(f)
                }
                TK::Global(idx) => {
                    f.write_str("global")?;
                    idx.fmt(f)
                }
                TK::Slice(idx) => {
                    let array = self.tys.ins.arrays[idx as usize];
                    f.write_str("[")?;
                    self.rety(array.ty).fmt(f)?;
                    if array.len != ArrayLen::MAX {
                        f.write_str("; ")?;
                        array.len.fmt(f)?;
                    }
                    f.write_str("]")
                }
            }
        }
    }

    #[cfg_attr(not(feature = "opts"), expect(dead_code))]
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

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum SymKey<'a> {
    Pointer(&'a Ptr),
    Struct(FileId, Pos),
    FuncInst(ty::Func, ty::Tuple),
    Decl(FileId, Ident),
    Array(&'a Array),
}

#[derive(Clone, Copy)]
struct Sig {
    args: ty::Tuple,
    ret: ty::Id,
}

struct Func {
    file: FileId,
    name: Ident,
    base: Option<ty::Func>,
    computed: Option<ty::Id>,
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
            name: 0,
            base: None,
            computed: None,
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

#[derive(Clone)]
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
    name: Ident,
    ty: ty::Id,
}

#[derive(Default)]
struct Struct {
    name: Ident,
    pos: Pos,
    file: FileId,
    size: Cell<Size>,
    align: Cell<u8>,
    explicit_alignment: Option<u8>,
    field_start: u32,
}

#[derive(PartialEq, Eq, Hash)]
pub struct Ptr {
    base: ty::Id,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Array {
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

impl ctx_map::CtxEntry for Ident {
    type Ctx = str;
    type Key<'a> = &'a str;

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        unsafe { ctx.get_unchecked(ident::range(*self)) }
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
                let id = ident::new(self.strings.len() as _, ident.len() as _).unwrap();
                self.strings.push_str(ident);
                v.insert(ctx_map::Key { hash, value: id }, ());
                id
            }
        }
    }

    fn ident_str(&self, ident: Ident) -> &str {
        &self.strings[ident::range(ident)]
    }

    fn project(&self, ident: &str) -> Option<Ident> {
        self.lookup.get(ident, &self.strings).copied()
    }
}

#[derive(Default)]
struct TypesTmp {
    fields: Vec<Field>,
    frontier: Vec<ty::Id>,
    globals: Vec<ty::Global>,
    funcs: Vec<ty::Func>,
    args: Vec<ty::Id>,
}

#[derive(Default)]
pub struct TypeIns {
    funcs: Vec<Func>,
    args: Vec<ty::Id>,
    globals: Vec<Global>,
    structs: Vec<Struct>,
    fields: Vec<Field>,
    ptrs: Vec<Ptr>,
    arrays: Vec<Array>,
}

#[derive(Default)]
struct Types {
    syms: ctx_map::CtxMap<ty::Id>,
    names: IdentInterner,
    ins: TypeIns,
    tmp: TypesTmp,
}

const HEADER_SIZE: usize = core::mem::size_of::<AbleOsExecutableHeader>();

impl Types {
    fn struct_field_range(&self, strct: ty::Struct) -> Range<usize> {
        let start = self.ins.structs[strct as usize].field_start as usize;
        let end = self
            .ins
            .structs
            .get(strct as usize + 1)
            .map_or(self.ins.fields.len(), |s| s.field_start as usize);
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

    fn find_type(
        &mut self,
        file: FileId,
        id: Result<Ident, &str>,
        files: &[parser::Ast],
    ) -> Option<ty::Id> {
        if let Ok(id) = id
            && let Some(&ty) = self.syms.get(SymKey::Decl(file, id), &self.ins)
        {
            if let ty::Kind::Global(g) = ty.expand() {
                let g = &self.ins.globals[g as usize];
                if g.ty == ty::Id::TYPE {
                    return Some(ty::Id::from(
                        u32::from_ne_bytes(*g.data.first_chunk().unwrap()) as u64
                    ));
                }
            }

            return Some(ty);
        }

        let f = &files[file as usize];
        let (Expr::BinOp { left, right, .. }, name) = f.find_decl(id)? else { unreachable!() };

        let ty = left
            .find_pattern_path(name, right, |right| self.ty(file, right, files))
            .unwrap_or_else(|_| unreachable!())?;
        if let ty::Kind::Struct(s) = ty.expand() {
            self.ins.structs[s as usize].name = name;
        }

        self.syms.insert(SymKey::Decl(file, name), ty, &self.ins);
        Some(ty)
    }

    /// returns none if comptime eval is required
    fn ty(&mut self, file: FileId, expr: &Expr, files: &[parser::Ast]) -> Option<ty::Id> {
        Some(match *expr {
            Expr::Mod { id, .. } => ty::Kind::Module(id).compress(),
            Expr::UnOp { op: TokenKind::Xor, val, .. } => {
                let base = self.ty(file, val, files)?;
                self.make_ptr(base)
            }
            Expr::Ident { id, .. } if ident::is_null(id) => id.into(),
            Expr::Ident { id, .. } => self.find_type(file, Ok(id), files)?,
            Expr::Field { target, name, .. } => {
                let ty::Kind::Module(file) = self.ty(file, target, files)?.expand() else {
                    return None;
                };

                self.find_type(file, Err(name), files)?
            }
            Expr::Slice { size: None, item, .. } => {
                let ty = self.ty(file, item, files)?;
                self.make_array(ty, ArrayLen::MAX)
            }
            Expr::Slice { size: Some(&Expr::Number { value, .. }), item, .. } => {
                let ty = self.ty(file, item, files)?;
                self.make_array(ty, value as _)
            }
            Expr::Struct { pos, fields, packed, .. } => {
                let sym = SymKey::Struct(file, pos);
                if let Some(&ty) = self.syms.get(sym, &self.ins) {
                    return Some(ty);
                }

                let prev_tmp = self.tmp.fields.len();
                for field in fields.iter().filter_map(CommentOr::or) {
                    let Some(ty) = self.ty(file, &field.ty, files) else {
                        self.tmp.fields.truncate(prev_tmp);
                        return None;
                    };
                    self.tmp.fields.push(Field { name: self.names.intern(field.name), ty });
                }

                self.ins.structs.push(Struct {
                    file,
                    pos,
                    field_start: self.ins.fields.len() as _,
                    explicit_alignment: packed.then_some(1),
                    ..Default::default()
                });
                self.ins.fields.extend(self.tmp.fields.drain(prev_tmp..));

                let ty = ty::Kind::Struct(self.ins.structs.len() as u32 - 1).compress();
                self.syms.insert(sym, ty, &self.ins);
                ty
            }
            _ => return None,
        })
    }

    fn reassemble(&mut self, buf: &mut Vec<u8>) {
        self.ins.funcs.iter_mut().for_each(|f| f.offset = u32::MAX);
        self.ins.globals.iter_mut().for_each(|g| g.offset = u32::MAX);
        self.assemble(buf)
    }

    fn assemble(&mut self, to: &mut Vec<u8>) {
        to.extend([0u8; HEADER_SIZE]);

        emit(to, instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
        emit(to, instrs::tx());
        let exe = self.dump_reachable(0, to);
        Reloc::new(HEADER_SIZE, 3, 4).apply_jump(to, self.ins.funcs[0].offset, 0);

        unsafe { *to.as_mut_ptr().cast::<AbleOsExecutableHeader>() = exe }
    }

    fn dump_reachable(&mut self, from: ty::Func, to: &mut Vec<u8>) -> AbleOsExecutableHeader {
        debug_assert!(self.tmp.frontier.is_empty());
        debug_assert!(self.tmp.funcs.is_empty());
        debug_assert!(self.tmp.globals.is_empty());

        self.tmp.frontier.push(ty::Kind::Func(from).compress());
        while let Some(itm) = self.tmp.frontier.pop() {
            match itm.expand() {
                ty::Kind::Func(func) => {
                    let fuc = &mut self.ins.funcs[func as usize];
                    if task::is_done(fuc.offset) {
                        continue;
                    }
                    fuc.offset = 0;
                    self.tmp.funcs.push(func);
                    self.tmp.frontier.extend(fuc.relocs.iter().map(|r| r.target));
                }
                ty::Kind::Global(glob) => {
                    let glb = &mut self.ins.globals[glob as usize];
                    if task::is_done(glb.offset) {
                        continue;
                    }
                    glb.offset = 0;
                    self.tmp.globals.push(glob);
                }
                _ => unreachable!(),
            }
        }

        for &func in &self.tmp.funcs {
            let fuc = &mut self.ins.funcs[func as usize];
            fuc.offset = to.len() as _;
            to.extend(&fuc.code);
        }

        let code_length = to.len();

        for global in self.tmp.globals.drain(..) {
            let global = &mut self.ins.globals[global as usize];
            global.offset = to.len() as _;
            to.extend(&global.data);
        }

        let data_length = to.len() - code_length;

        for func in self.tmp.funcs.drain(..) {
            let fuc = &self.ins.funcs[func as usize];
            for rel in &fuc.relocs {
                let offset = match rel.target.expand() {
                    ty::Kind::Func(fun) => self.ins.funcs[fun as usize].offset,
                    ty::Kind::Global(glo) => self.ins.globals[glo as usize].offset,
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

    pub fn disasm<'a>(
        &'a self,
        mut sluce: &[u8],
        files: &'a [parser::Ast],
        output: &mut String,
        eca_handler: impl FnMut(&mut &[u8]),
    ) -> Result<(), hbbytecode::DisasmError<'a>> {
        use instrs::DisasmItem;
        let functions = self
            .ins
            .funcs
            .iter()
            .filter(|f| task::is_done(f.offset))
            .map(|f| {
                let name = if f.file != u32::MAX {
                    let file = &files[f.file as usize];
                    let Expr::BinOp { left: &Expr::Ident { id, .. }, .. } =
                        f.expr.get(file).unwrap()
                    else {
                        unreachable!()
                    };
                    file.ident_str(id)
                } else {
                    "target_fn"
                };
                (f.offset, (name, f.code.len() as u32, DisasmItem::Func))
            })
            .chain(self.ins.globals.iter().filter(|g| task::is_done(g.offset)).map(|g| {
                let name = if g.file == u32::MAX {
                    core::str::from_utf8(&g.data).unwrap()
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

    fn make_ptr(&mut self, base: ty::Id) -> ty::Id {
        ty::Kind::Ptr(self.make_ptr_low(base)).compress()
    }

    fn make_ptr_low(&mut self, base: ty::Id) -> ty::Ptr {
        let ptr = Ptr { base };
        let (entry, hash) = self.syms.entry(SymKey::Pointer(&ptr), &self.ins);
        match entry {
            hash_map::RawEntryMut::Occupied(o) => o.get_key_value().0.value,
            hash_map::RawEntryMut::Vacant(v) => {
                self.ins.ptrs.push(ptr);
                v.insert(
                    ctx_map::Key {
                        value: ty::Kind::Ptr(self.ins.ptrs.len() as u32 - 1).compress(),
                        hash,
                    },
                    (),
                )
                .0
                .value
            }
        }
        .expand()
        .inner()
    }

    fn make_array(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Id {
        ty::Kind::Slice(self.make_array_low(ty, len)).compress()
    }

    fn make_array_low(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Slice {
        self.syms
            .get_or_insert(SymKey::Array(&Array { ty, len }), &mut self.ins, |ins| {
                ins.arrays.push(Array { ty, len });
                ty::Kind::Slice(ins.arrays.len() as u32 - 1).compress()
            })
            .expand()
            .inner()

        //let array = Array { ty, len };
        //let (entry, hash) = self.syms.entry(SymKey::Array(&array), &self.ins);
        //match entry {
        //    hash_map::RawEntryMut::Occupied(o) => o.get_key_value().0.value,
        //    hash_map::RawEntryMut::Vacant(v) => {
        //        self.ins.arrays.push(array);
        //        v.insert(
        //            ctx_map::Key {
        //                value: ty::Kind::Slice(self.ins.ptrs.len() as u32 - 1).compress(),
        //                hash,
        //            },
        //            (),
        //        )
        //        .0
        //        .value
        //    }
        //}
        //.expand()
        //.inner()
    }

    fn size_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Ptr(_) => 8,
            ty::Kind::Builtin(ty::VOID) => 0,
            ty::Kind::Builtin(ty::NEVER) => 0,
            ty::Kind::Builtin(ty::INT | ty::UINT) => 8,
            ty::Kind::Builtin(ty::I32 | ty::U32 | ty::TYPE) => 4,
            ty::Kind::Builtin(ty::I16 | ty::U16) => 2,
            ty::Kind::Builtin(ty::I8 | ty::U8 | ty::BOOL) => 1,
            ty::Kind::Slice(arr) => {
                let arr = &self.ins.arrays[arr as usize];
                match arr.len {
                    0 => 0,
                    ArrayLen::MAX => 16,
                    len => self.size_of(arr.ty) * len,
                }
            }
            ty::Kind::Struct(stru) => {
                if self.ins.structs[stru as usize].size.get() != 0 {
                    return self.ins.structs[stru as usize].size.get();
                }

                let mut oiter = OffsetIter::new(stru, self);
                while oiter.next(self).is_some() {}
                self.ins.structs[stru as usize].size.set(oiter.offset);
                oiter.offset
            }
            ty => unimplemented!("size_of: {:?}", ty),
        }
    }

    fn align_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Struct(stru) => {
                if self.ins.structs[stru as usize].align.get() != 0 {
                    return self.ins.structs[stru as usize].align.get() as _;
                }
                let align = self.ins.structs[stru as usize].explicit_alignment.map_or_else(
                    || {
                        self.struct_fields(stru)
                            .iter()
                            .map(|&Field { ty, .. }| self.align_of(ty))
                            .max()
                            .unwrap_or(1)
                    },
                    |a| a as _,
                );
                self.ins.structs[stru as usize].align.set(align.try_into().unwrap());
                align
            }
            ty::Kind::Slice(arr) => {
                let arr = &self.ins.arrays[arr as usize];
                match arr.len {
                    ArrayLen::MAX => 8,
                    _ => self.align_of(arr.ty),
                }
            }
            _ => self.size_of(ty).max(1),
        }
    }

    fn base_of(&self, ty: ty::Id) -> Option<ty::Id> {
        match ty.expand() {
            ty::Kind::Ptr(p) => Some(self.ins.ptrs[p as usize].base),
            _ => None,
        }
    }

    #[cfg_attr(not(feature = "opts"), expect(dead_code))]
    #[expect(dead_code)]
    fn find_struct_field(&self, s: ty::Struct, name: &str) -> Option<usize> {
        let name = self.names.project(name)?;
        self.struct_fields(s).iter().position(|f| f.name == name)
    }
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
        let stru = &tys.ins.structs[self.strct as usize];
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

#[cfg(any(feature = "opts", feature = "std"))]
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
fn test_parse_files(ident: &'static str, input: &'static str) -> (Vec<parser::Ast>, Vec<Vec<u8>>) {
    use {
        self::parser::FileKind,
        std::{borrow::ToOwned, string::ToString},
    };

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
            .map(|i| i as parser::FileId)
            .ok_or("Module Not Found".to_string()),
        FileKind::Embed => embed_map
            .iter()
            .position(|&(name, _)| name == path)
            .map(|i| i as parser::FileId)
            .ok_or("Embed Not Found".to_string()),
    };

    let mut ctx = parser::ParserCtx::default();
    (
        module_map
            .iter()
            .map(|&(path, content)| {
                parser::Ast::new(path, content.to_owned(), &mut ctx, &mut loader)
            })
            .collect(),
        embed_map.iter().map(|&(_, content)| content.to_owned().into_bytes()).collect(),
    )
}

#[cfg(test)]
fn test_run_vm(out: &[u8], output: &mut String) {
    use core::fmt::Write;

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
                    let layout = core::alloc::Layout::from_size_align(size, align).unwrap();
                    let ptr = unsafe { alloc::alloc::alloc(layout) };
                    vm.write_reg(1, ptr as u64);
                }
                96 => {
                    let [ptr, size, align] = [
                        vm.read_reg(3).0 as usize,
                        vm.read_reg(4).0 as usize,
                        vm.read_reg(5).0 as usize,
                    ];

                    let layout = core::alloc::Layout::from_size_align(size, align).unwrap();
                    unsafe { alloc::alloc::dealloc(ptr as *mut u8, layout) };
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
pub struct LoggedMem {
    pub mem: hbvm::mem::HostMemory,
    op_buf: Vec<hbbytecode::Oper>,
    disp_buf: String,
    prev_instr: Option<hbbytecode::Instr>,
}

impl LoggedMem {
    unsafe fn display_instr<T>(&mut self, instr: hbbytecode::Instr, addr: hbvm::mem::Address) {
        let novm: *const hbvm::Vm<Self, 0> = core::ptr::null();
        let offset = core::ptr::addr_of!((*novm).memory) as usize;
        let regs = unsafe {
            &*core::ptr::addr_of!(
                (*(((self as *mut _ as *mut u8).sub(offset)) as *const hbvm::Vm<Self, 0>))
                    .registers
            )
        };

        let mut bytes = core::slice::from_raw_parts(
            (addr.get() - 1) as *const u8,
            core::mem::size_of::<T>() + 1,
        );
        use core::fmt::Write;
        hbbytecode::parse_args(&mut bytes, instr, &mut self.op_buf).unwrap();
        debug_assert!(bytes.is_empty());
        self.disp_buf.clear();
        write!(self.disp_buf, "{:<10}", format!("{instr:?}")).unwrap();
        for (i, op) in self.op_buf.drain(..).enumerate() {
            if i != 0 {
                write!(self.disp_buf, ", ").unwrap();
            }
            write!(self.disp_buf, "{op:?}").unwrap();
            if let hbbytecode::Oper::R(r) = op {
                write!(self.disp_buf, "({})", regs[r as usize].0).unwrap()
            }
        }
        log::trace!("read-typed: {:x}: {}", addr.get(), self.disp_buf);
    }
}

impl hbvm::mem::Memory for LoggedMem {
    unsafe fn load(
        &mut self,
        addr: hbvm::mem::Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), hbvm::mem::LoadError> {
        log::trace!(
            "load: {:x} {}",
            addr.get(),
            AsHex(core::slice::from_raw_parts(addr.get() as *const u8, count))
        );
        self.mem.load(addr, target, count)
    }

    unsafe fn store(
        &mut self,
        addr: hbvm::mem::Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), hbvm::mem::StoreError> {
        log::trace!(
            "store: {:x} {}",
            addr.get(),
            AsHex(core::slice::from_raw_parts(source, count))
        );
        self.mem.store(addr, source, count)
    }

    unsafe fn prog_read<T: Copy + 'static>(&mut self, addr: hbvm::mem::Address) -> T {
        if log::log_enabled!(log::Level::Trace) {
            if core::any::TypeId::of::<u8>() == core::any::TypeId::of::<T>() {
                if let Some(instr) = self.prev_instr {
                    self.display_instr::<()>(instr, addr);
                }
                self.prev_instr = hbbytecode::Instr::try_from(*(addr.get() as *const u8)).ok();
            } else {
                let instr = self.prev_instr.take().unwrap();
                self.display_instr::<T>(instr, addr);
            }
        }

        self.mem.prog_read(addr)
    }
}

struct AsHex<'a>(&'a [u8]);

impl core::fmt::Display for AsHex<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for &b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
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
