#![allow(dead_code)]
use {
    crate::{
        ident::{self, Ident},
        instrs::{self, *},
        lexer::{self, TokenKind},
        log,
        parser::{self, find_symbol, idfl, CtorField, Expr, ExprRef, FileId, Pos},
        HashMap,
    },
    std::{
        mem,
        ops::{self, Range},
        rc::Rc,
    },
};

type Nid = u32;

pub struct PoolVec<T> {
    values: Vec<PoolSlot<T>>,
    free: u32,
}

impl<T> Default for PoolVec<T> {
    fn default() -> Self {
        Self { values: Default::default(), free: u32::MAX }
    }
}

impl<T> PoolVec<T> {
    pub fn add(&mut self, value: T) -> u32 {
        if self.free == u32::MAX {
            self.free = self.values.len() as _;
            self.values.push(PoolSlot::Next(u32::MAX));
        }

        let free = self.free;
        self.free = match mem::replace(&mut self.values[free as usize], PoolSlot::Value(value)) {
            PoolSlot::Value(_) => unreachable!(),
            PoolSlot::Next(free) => free,
        };
        free
    }

    pub fn remove(&mut self, id: u32) -> T {
        let value = match mem::replace(&mut self.values[id as usize], PoolSlot::Next(self.free)) {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        };
        self.free = id;
        value
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.free = u32::MAX;
    }
}

impl<T> ops::Index<u32> for PoolVec<T> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        match &self.values[index as usize] {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        }
    }
}

impl<T> ops::IndexMut<u32> for PoolVec<T> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match &mut self.values[index as usize] {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        }
    }
}

enum PoolSlot<T> {
    Value(T),
    Next(u32),
}

pub enum Inputs {
    Start,
    End,
    BinOp { op: lexer::TokenKind, lhs: Nid, rhs: Nid },
    Return { value: Nid, cfg: Nid },
    Tuple { on: Nid, index: Nid },
    ConstInt { value: i64 },
}

pub struct Node {
    pub inputs: Inputs,
    pub depth: u32,
    pub ty: ty::Id,
    pub outputs: Vec<Nid>,
}

pub type Nodes = PoolVec<Node>;

type Offset = u32;
type Size = u32;
type ArrayLen = u32;

mod ty {
    use {
        crate::{
            lexer::TokenKind,
            parser::{self, Expr},
            son::ArrayLen,
        },
        std::{num::NonZeroU32, ops::Range},
    };

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;
    pub type Param = u32;
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
}

#[derive(Clone, Copy, Debug)]
struct Loop {
    var_count: u32,
    offset: u32,
    reloc_base: u32,
}

struct Variable {
    id: Ident,
    value: Nid,
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    id: ty::Kind,
    ret: Option<ty::Id>,

    task_base: usize,
    snap: Snapshot,

    loops: Vec<Loop>,
    vars: Vec<Variable>,
}

impl ItemCtx {}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
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

#[derive(Clone, Copy)]
struct Func {
    file: FileId,
    expr: ExprRef,
    sig: Option<Sig>,
    offset: Offset,
}

struct Global {
    offset: Offset,
    ty: ty::Id,
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

struct ParamAlloc(Range<u8>);

impl ParamAlloc {
    pub fn next(&mut self) -> u8 {
        self.0.next().expect("too many paramteters")
    }

    fn next_wide(&mut self) -> u8 {
        (self.next(), self.next()).0
    }
}

#[derive(Clone, Copy)]
struct Array {
    ty: ty::Id,
    len: ArrayLen,
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

impl Types {
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
    id: ty::Func,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Snapshot {
    code: usize,
    string_data: usize,
    funcs: usize,
    globals: usize,
    strings: usize,
}

impl Snapshot {
    fn _sub(&mut self, other: &Self) {
        self.code -= other.code;
        self.string_data -= other.string_data;
        self.funcs -= other.funcs;
        self.globals -= other.globals;
        self.strings -= other.strings;
    }

    fn _add(&mut self, other: &Self) {
        self.code += other.code;
        self.string_data += other.string_data;
        self.funcs += other.funcs;
        self.globals += other.globals;
        self.strings += other.strings;
    }
}

#[derive(Default, Debug)]
struct Ctx {
    ty: Option<ty::Id>,
}

impl Ctx {
    pub fn with_ty(self, ty: impl Into<ty::Id>) -> Self {
        Self { ty: Some(ty.into()), ..self }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
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

#[derive(Default)]
pub struct Codegen {
    pub files: Vec<parser::Ast>,
    tasks: Vec<Option<FTask>>,

    tys: Types,
    ci: ItemCtx,
    pool: Pool,
}

impl Codegen {
    pub fn generate(&mut self) {
        self.find_or_declare(0, 0, Err("main"), "");
        self.complete_call_graph_low();
    }

    fn expr(&mut self, expr: &Expr) -> Option<Nid> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn build_struct(&mut self, fields: &[(&str, Expr)]) -> ty::Struct {
        let fields = fields
            .iter()
            .map(|&(name, ty)| Field { name: name.into(), ty: self.ty(&ty) })
            .collect();
        self.tys.structs.push(Struct { fields });
        self.tys.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Nid> {
        todo!()
    }

    #[must_use]
    fn complete_call_graph(&mut self) {
        self.complete_call_graph_low();
    }

    fn complete_call_graph_low(&mut self) {
        while self.ci.task_base < self.tasks.len()
            && let Some(task_slot) = self.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.handle_task(task);
        }
    }

    fn handle_task(&mut self, FTask { file, id }: FTask) {
        todo!();
    }

    // TODO: sometimes its better to do this in bulk
    fn ty(&mut self, expr: &Expr) -> ty::Id {
        todo!()
    }

    fn find_or_declare(
        &mut self,
        pos: Pos,
        file: FileId,
        name: Result<Ident, &str>,
        lit_name: &str,
    ) -> ty::Kind {
        todo!()
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, &self.files, ty)
    }

    #[must_use]
    #[track_caller]
    fn assert_ty(&self, pos: Pos, ty: ty::Id, expected: ty::Id) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected) {
            res
        } else {
            let ty = self.ty_display(ty);
            let expected = self.ty_display(expected);
            self.report(pos, format_args!("expected {expected}, got {ty}"));
        }
    }

    fn report_log(&self, pos: Pos, msg: impl std::fmt::Display) {
        let (line, col) = lexer::line_col(self.cfile().file.as_bytes(), pos);
        println!("{}:{}:{}: {}", self.cfile().path, line, col, msg);
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl std::fmt::Display) -> ! {
        self.report_log(pos, msg);
        unreachable!();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: &str) -> ! {
        self.report(
            ast.pos(),
            format_args!(
                "compiler does not (yet) know how to handle ({hint}):\n\
                {ast:}\n\
                info for weak people:\n\
                {ast:#?}"
            ),
        )
    }

    fn cfile(&self) -> &parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn pack_args(&mut self, pos: Pos, arg_base: usize) -> ty::Tuple {
        let needle = &self.tys.args[arg_base..];
        if needle.is_empty() {
            return ty::Tuple::empty();
        }
        let len = needle.len();
        // FIXME: maybe later when this becomes a bottleneck we use more
        // efficient search (SIMD?, indexing?)
        let sp = self.tys.args.windows(needle.len()).position(|val| val == needle).unwrap();
        self.tys.args.truncate((sp + needle.len()).max(arg_base));
        ty::Tuple::new(sp, len)
            .unwrap_or_else(|| self.report(pos, "amount of arguments not supported"))
    }
}

#[cfg(test)]
mod tests {
    use {
        super::parser,
        crate::{codegen::LoggedMem, log, parser::FileId},
        std::io,
    };

    const README: &str = include_str!("../README.md");

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        todo!()
    }

    crate::run_tests! { generate:
        arithmetic => README;
        variables => README;
        functions => README;
        comments => README;
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
        c_strings => README;
        struct_patterns => README;
        arrays => README;
        struct_return_from_module_function => README;
        //comptime_pointers => README;
        sort_something_viredly => README;
        hex_octal_binary_literals => README;
        comptime_min_reg_leak => README;
       // structs_in_registers => README;
        comptime_function_from_another_file => README;
        inline => README;
        inline_test => README;
    }
}
