pub use self::reg::{RET_ADDR, STACK_PTR, ZERO};
use {
    crate::{
        ident::{self, Ident},
        instrs::{self, *},
        lexer::TokenKind,
        log,
        parser::{
            self, find_symbol, idfl, CommentOr, CtorField, Expr, ExprRef, FileId, Pos, StructField,
        },
        ty, Field, Func, Global, LoggedMem, ParamAlloc, Reloc, Sig, Struct, SymKey, TypedReloc,
        Types,
    },
    std::fmt::Display,
};

type Offset = u32;
type Size = u32;
type ArrayLen = u32;

fn load_value(ptr: *const u8, size: u32) -> u64 {
    let mut dst = [0u8; 8];
    dst[..size as usize].copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, size as usize) });
    u64::from_ne_bytes(dst)
}

fn ensure_loaded(value: CtValue, derefed: bool, size: u32) -> u64 {
    if derefed {
        load_value(value.0 as *const u8, size)
    } else {
        value.0
    }
}

mod stack {
    use {
        super::{Offset, Size},
        std::num::NonZeroU32,
    };

    impl crate::Reloc {
        pub fn pack_srel(id: &Id, off: u32) -> u64 {
            ((id.repr() as u64) << 32) | (off as u64)
        }

        pub fn apply_stack_offset(&self, code: &mut [u8], stack: &Alloc) {
            let bytes =
                &code[self.offset as usize + self.sub_offset as usize..][..self.width as usize];
            let (id, off) = Self::unpack_srel(u64::from_ne_bytes(bytes.try_into().unwrap()));
            self.write_offset(code, stack.final_offset(id, off) as i64);
        }

        pub fn unpack_srel(id: u64) -> (u32, u32) {
            ((id >> 32) as u32, id as u32)
        }
    }

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
        size: Size,
        offset: Offset,
        rc: u32,
    }

    #[derive(Default)]
    pub struct Alloc {
        height: Size,
        pub max_height: Size,
        meta: Vec<Meta>,
    }

    impl Alloc {
        pub fn allocate(&mut self, size: Size) -> Id {
            self.meta.push(Meta { size, offset: 0, rc: 1 });

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

        pub fn dup_id(&mut self, id: &Id) -> Id {
            if id.is_ref() {
                return id.as_ref();
            }

            self.meta[id.index()].rc += 1;
            Id(id.0)
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

    #[cfg(debug_assertions)]
    type Bt = std::backtrace::Backtrace;
    #[cfg(not(debug_assertions))]
    type Bt = ();

    #[derive(Default, Debug)]
    pub struct Id(Reg, Option<Bt>);

    impl PartialEq for Id {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl Eq for Id {}

    impl Id {
        pub const RET: Self = Id(RET, None);

        pub fn get(&self) -> Reg {
            self.0
        }

        pub fn as_ref(&self) -> Self {
            Self(self.0, None)
        }

        pub fn is_ref(&self) -> bool {
            self.1.is_none()
        }
    }

    impl From<u8> for Id {
        fn from(value: u8) -> Self {
            Self(value, None)
        }
    }

    #[cfg(debug_assertions)]
    impl Drop for Id {
        fn drop(&mut self) {
            if !std::thread::panicking()
                && let Some(bt) = self.1.take()
            {
                unreachable!("reg id leaked: {:?} {bt}", self.0);
            }
        }
    }

    #[derive(Default, PartialEq, Eq)]
    pub struct Alloc {
        free: Vec<Reg>,
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
            Id(
                reg,
                #[cfg(debug_assertions)]
                Some(std::backtrace::Backtrace::capture()),
                #[cfg(not(debug_assertions))]
                Some(()),
            )
        }

        pub fn free(&mut self, reg: Id) {
            if reg.1.is_some() {
                self.free.push(reg.0);
                std::mem::forget(reg);
            }
        }

        pub fn pushed_size(&self) -> usize {
            ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
        }
    }
}

struct Value {
    ty: ty::Id,
    loc: Loc,
}

impl Value {
    fn new(ty: impl Into<ty::Id>, loc: impl Into<Loc>) -> Self {
        Self { ty: ty.into(), loc: loc.into() }
    }

    fn void() -> Self {
        Self { ty: ty::VOID.into(), loc: Loc::ct(0) }
    }

    fn imm(value: u64) -> Self {
        Self { ty: ty::UINT.into(), loc: Loc::ct(value) }
    }

    fn ty(ty: ty::Id) -> Self {
        Self { ty: ty::TYPE.into(), loc: Loc::ct(ty.repr() as u64) }
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

#[repr(packed)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct CtValue(u64);

#[derive(Debug, PartialEq, Eq)]
enum Loc {
    Rt { derefed: bool, reg: reg::Id, stack: Option<stack::Id>, offset: Offset },
    Ct { derefed: bool, value: CtValue },
}

impl Loc {
    fn stack(stack: stack::Id) -> Self {
        Self::Rt { stack: Some(stack), reg: reg::STACK_PTR.into(), derefed: true, offset: 0 }
    }

    fn reg(reg: impl Into<reg::Id>) -> Self {
        let reg = reg.into();
        assert!(reg.get() != 0);
        Self::Rt { derefed: false, reg, stack: None, offset: 0 }
    }

    fn ct(value: u64) -> Self {
        Self::Ct { value: CtValue(value), derefed: false }
    }

    fn ct_ptr(value: u64) -> Self {
        Self::Ct { value: CtValue(value), derefed: true }
    }

    fn ty(ty: ty::Id) -> Self {
        Self::ct(ty.repr() as _)
    }

    fn offset(mut self, offset: u32) -> Self {
        match &mut self {
            Self::Rt { offset: off, .. } => *off += offset,
            Self::Ct { derefed: false, value } => value.0 += offset as u64,
            _ => unreachable!("offseting constant"),
        }
        self
    }

    fn as_ref(&self) -> Self {
        match *self {
            Loc::Rt { derefed, ref reg, ref stack, offset } => Loc::Rt {
                derefed,
                reg: reg.as_ref(),
                stack: stack.as_ref().map(stack::Id::as_ref),
                offset,
            },
            Loc::Ct { value, derefed } => Self::Ct { derefed, value },
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
            Self::Ct { derefed: false, value } => Some(ty::Id::from(value.0)),
            Self::Ct { derefed: true, value } => {
                Some(unsafe { std::ptr::read(value.0 as *const u8 as _) })
            }
            Self::Rt { .. } => None,
        }
    }

    fn is_stack(&self) -> bool {
        matches!(self, Self::Rt { derefed: true, reg, stack: Some(_), offset: 0 } if reg.get() == STACK_PTR)
    }

    fn is_reg(&self) -> bool {
        matches!(self, Self::Rt { derefed: false, reg: _, stack: None, offset: 0 })
    }
}

impl From<reg::Id> for Loc {
    fn from(reg: reg::Id) -> Self {
        Loc::reg(reg)
    }
}

impl Default for Loc {
    fn default() -> Self {
        Self::ct(0)
    }
}

#[derive(Clone, Copy)]
struct Loop {
    var_count: u32,
    offset: u32,
    reloc_base: u32,
}

struct Variable {
    id: Ident,
    uses_left: u32,
    value: Value,
}

struct ItemCtxSnap {
    stack_relocs: usize,
    ret_relocs: usize,
    loop_relocs: usize,
    code: usize,
    relocs: usize,
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    id: ty::Kind,
    ret: Option<ty::Id>,
    ret_reg: reg::Id,
    inline_ret_loc: Loc,

    task_base: usize,

    stack: stack::Alloc,
    regs: reg::Alloc,

    loops: Vec<Loop>,
    vars: Vec<Variable>,

    stack_relocs: Vec<Reloc>,
    ret_relocs: Vec<Reloc>,
    loop_relocs: Vec<Reloc>,
    code: Vec<u8>,
    relocs: Vec<TypedReloc>,
}

impl ItemCtx {
    fn write_trap(&mut self, kind: trap::Trap) {
        self.emit(eca());
        self.code.push(255);
        self.code.extend(kind.as_slice());
    }

    fn snap(&self) -> ItemCtxSnap {
        ItemCtxSnap {
            stack_relocs: self.stack_relocs.len(),
            ret_relocs: self.ret_relocs.len(),
            loop_relocs: self.loop_relocs.len(),
            code: self.code.len(),
            relocs: self.relocs.len(),
        }
    }

    fn revert(&mut self, snap: ItemCtxSnap) {
        self.stack_relocs.truncate(snap.stack_relocs);
        self.ret_relocs.truncate(snap.ret_relocs);
        self.loop_relocs.truncate(snap.loop_relocs);
        self.code.truncate(snap.code);
        self.relocs.truncate(snap.relocs);
    }

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
        let name = instrs::NAMES[instr[0] as usize];
        log::trc!(
            "{:08x}: {}: {}",
            self.code.len(),
            name,
            instr.iter().take(len).skip(1).map(|b| format!("{:02x}", b)).collect::<String>()
        );
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

    pub fn dup_loc(&mut self, loc: &Loc) -> Loc {
        match *loc {
            Loc::Rt { derefed, ref reg, ref stack, offset } => Loc::Rt {
                reg: reg.as_ref(),
                derefed,
                stack: stack.as_ref().map(|s| self.stack.dup_id(s)),
                offset,
            },
            ref loc => loc.as_ref(),
        }
    }

    fn finalize(&mut self) {
        if let Some(last_ret) = self.ret_relocs.last()
            && last_ret.offset as usize == self.code.len() - 5
        {
            self.code.truncate(self.code.len() - 5);
            self.ret_relocs.pop();
        }

        let len = self.code.len() as Offset;

        self.stack.finalize_leaked();
        for rel in self.stack_relocs.drain(..) {
            rel.apply_stack_offset(&mut self.code, &self.stack)
        }

        for rel in self.ret_relocs.drain(..) {
            let off = rel.apply_jump(&mut self.code, len, 0);
            debug_assert!(off > 0);
        }

        let pushed = self.regs.pushed_size() as i64;
        let stack = self.stack.max_height as i64;

        write_reloc(&mut self.code, 3, -(pushed + stack), 8);
        write_reloc(&mut self.code, 3 + 8 + 3, stack, 8);
        write_reloc(&mut self.code, 3 + 8 + 3 + 8, pushed, 2);

        self.emit(instrs::ld(reg::RET_ADDR, reg::STACK_PTR, stack as _, pushed as _));
        self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, (pushed + stack) as _));

        self.stack.clear();

        debug_assert!(self.loops.is_empty());
        debug_assert!(self.loop_relocs.is_empty());
        debug_assert!(self.vars.is_empty());
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

#[derive(Debug)]
struct FTask {
    file: FileId,
    id: ty::Func,
}

#[derive(Default, Debug)]
struct Ctx {
    loc: Option<Loc>,
    ty: Option<ty::Id>,
}

impl Ctx {
    pub fn with_loc(self, loc: Loc) -> Self {
        Self { loc: Some(loc), ..self }
    }

    pub fn with_ty(self, ty: impl Into<ty::Id>) -> Self {
        Self { ty: Some(ty.into()), ..self }
    }

    fn into_value(self) -> Option<Value> {
        Some(Value { ty: self.ty.unwrap(), loc: self.loc? })
    }
}

impl From<Value> for Ctx {
    fn from(value: Value) -> Self {
        Self { loc: Some(value.loc), ty: Some(value.ty) }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
    arg_locs: Vec<Loc>,
}

const VM_STACK_SIZE: usize = 1024 * 1024 * 2;

struct Comptime {
    vm: hbvm::Vm<LoggedMem, { 1024 * 10 }>,
    _stack: Box<[u8; VM_STACK_SIZE]>,
    code: Vec<u8>,
}

impl Default for Comptime {
    fn default() -> Self {
        let mut stack = Box::<[u8; VM_STACK_SIZE]>::new_uninit();
        let mut vm = hbvm::Vm::default();
        let ptr = unsafe { stack.as_mut_ptr().cast::<u8>().add(VM_STACK_SIZE) as u64 };
        vm.write_reg(STACK_PTR, ptr);
        Self { vm, _stack: unsafe { stack.assume_init() }, code: Default::default() }
    }
}

mod trap {
    use {
        super::ty,
        crate::parser::{ExprRef, FileId},
    };

    macro_rules! gen_trap {
        (
            #[derive(Trap)]
            $vis:vis enum $name:ident {
                $($variant:ident {
                    $($fname:ident: $fty:ty,)*
                },)*
            }
        ) => {
            #[repr(u8)]
            $vis enum $name {
                $($variant($variant),)*
            }

            impl $name {
                $vis fn size(&self) -> usize {
                    1 + match self {
                        $(Self::$variant(_) => std::mem::size_of::<$variant>(),)*
                    }
                }
            }

            $(
                #[repr(packed)]
                $vis struct $variant {
                    $($vis $fname: $fty,)*
                }
            )*
        };
    }

    gen_trap! {
        #[derive(Trap)]
        pub enum Trap {
            MakeStruct {
                file: FileId,
                struct_expr: ExprRef,
            },
            MomizedCall {
                func: ty::Func,
            },
        }
    }

    impl Trap {
        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self as *const _ as _, self.size()) }
        }
    }
}

#[derive(Default)]
pub struct Codegen {
    pub files: Vec<parser::Ast>,
    tasks: Vec<Option<FTask>>,

    tys: Types,
    ci: ItemCtx,
    pool: Pool,
    ct: Comptime,
}

impl Codegen {
    pub fn generate(&mut self) {
        self.ci.emit_entry_prelude();
        self.find_or_declare(0, 0, Err("main"), "");
        self.make_func_reachable(0);
        self.complete_call_graph();
    }

    fn expr(&mut self, expr: &Expr) -> Option<Value> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn build_struct(&mut self, fields: &[CommentOr<StructField>]) -> ty::Struct {
        let fields = fields
            .iter()
            .filter_map(CommentOr::or)
            .map(|sf| Field { name: sf.name.into(), ty: self.ty(&sf.ty) })
            .collect();
        self.tys.structs.push(Struct { fields });
        self.tys.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Value> {
        use {Expr as E, TokenKind as T};
        let value = match *expr {
            E::Mod { id, .. } => Some(Value::ty(ty::Kind::Module(id).compress())),
            E::Struct { fields, captured, .. } => {
                if captured.is_empty() {
                    Some(Value::ty(ty::Kind::Struct(self.build_struct(fields)).compress()))
                } else {
                    let values = captured
                        .iter()
                        .map(|&id| E::Ident {
                            pos: 0,
                            is_ct: false,
                            id,
                            name: "booodab",
                            index: u16::MAX,
                        })
                        .map(|expr| self.expr(&expr))
                        .collect::<Option<Vec<_>>>()?;
                    let values_size =
                        values.iter().map(|value| 4 + self.tys.size_of(value.ty)).sum::<Size>();

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
                        trap::Trap::MakeStruct(trap::MakeStruct {
                            file: self.ci.file,
                            struct_expr: ExprRef::new(expr),
                        }),
                        ty::TYPE,
                    );
                    self.ci.free_loc(Loc::stack(stack));
                    Some(val)
                }
            }
            E::Slice { size, item, .. } => {
                let ty = self.ty(item);
                let len = size.map_or(ArrayLen::MAX, |expr| self.eval_const(expr, ty::U32) as _);
                Some(Value::ty(self.tys.make_array(ty, len)))
            }
            E::Index { base, index } => {
                // TODO: we need to check if index is in bounds on debug builds

                let mut base_val = self.expr(base)?;
                if base_val.ty.is_pointer() {
                    base_val.loc = self.make_loc_owned(base_val.loc, base_val.ty);
                }
                let index_val = self.expr(index)?;
                _ = self.assert_ty(index.pos(), index_val.ty, ty::INT.into(), "subsctipt");

                if let ty::Kind::Ptr(ty) = base_val.ty.expand() {
                    base_val.ty = self.tys.ptrs[ty as usize].base;
                    base_val.loc = base_val.loc.into_derefed();
                }

                match base_val.ty.expand() {
                    ty::Kind::Slice(arr) => {
                        let ty = self.tys.arrays[arr as usize].ty;
                        let item_size = self.tys.size_of(ty);

                        let Loc::Rt { derefed: true, ref mut reg, ref stack, offset } =
                            base_val.loc
                        else {
                            unreachable!()
                        };

                        if reg.is_ref() {
                            let new_reg = self.ci.regs.allocate();
                            self.stack_offset(new_reg.get(), reg.get(), stack.as_ref(), offset);
                            *reg = new_reg;
                        } else {
                            self.stack_offset(reg.get(), reg.get(), stack.as_ref(), offset);
                        }

                        let idx = self.loc_to_reg(index_val.loc, 8);

                        if item_size != 1 {
                            self.ci.emit(muli64(idx.get(), idx.get(), item_size as _));
                        }
                        self.ci.emit(add64(reg.get(), reg.get(), idx.get()));
                        self.ci.regs.free(idx);

                        Some(Value::new(ty, base_val.loc))
                    }
                    _ => self.report(
                        base.pos(),
                        format_args!(
                            "compiler did not (yet) learn how to index into '{}'",
                            self.ty_display(base_val.ty)
                        ),
                    ),
                }
            }
            E::Directive { name: "inline", args: [func_ast, args @ ..], .. } => {
                let ty::Kind::Func(mut func) = self.ty(func_ast).expand() else {
                    self.report(func_ast.pos(), "first argument of inline needs to be a function");
                };

                let fuc = &self.tys.funcs[func as usize];
                let fast = self.files[fuc.file as usize].clone();
                let E::BinOp { right: &E::Closure { args: cargs, body, .. }, .. } =
                    fuc.expr.get(&fast).unwrap()
                else {
                    unreachable!();
                };

                let scope = self.ci.vars.len();
                let sig = self.compute_signature(&mut func, func_ast.pos(), args)?;

                self.assert_arg_count(expr.pos(), args.len(), cargs.len(), "inline function call");

                if scope == self.ci.vars.len() {
                    for ((arg, ty), carg) in
                        args.iter().zip(sig.args.view(&self.tys.args).to_owned()).zip(cargs)
                    {
                        let loc = self.expr_ctx(arg, Ctx::default().with_ty(ty))?.loc;

                        let sym = parser::find_symbol(&fast.symbols, carg.id).flags;
                        self.ci.vars.push(Variable {
                            id: carg.id,
                            value: Value { ty, loc },
                            uses_left: idfl::count(sym) as u32,
                        });
                    }
                }

                let ret_reloc_base = self.ci.ret_relocs.len();

                let loc = self.alloc_ret(sig.ret, ctx, true);
                let prev_ret_reg = std::mem::replace(&mut self.ci.inline_ret_loc, loc);
                let fuc = &self.tys.funcs[func as usize];
                let prev_file = std::mem::replace(&mut self.ci.file, fuc.file);
                let prev_ret = std::mem::replace(&mut self.ci.ret, Some(sig.ret));
                self.expr(body);
                let loc = std::mem::replace(&mut self.ci.inline_ret_loc, prev_ret_reg);
                self.ci.file = prev_file;
                self.ci.ret = prev_ret;

                for var in self.ci.vars.drain(scope..).collect::<Vec<_>>() {
                    self.ci.free_loc(var.value.loc);
                }

                if let Some(last_ret) = self.ci.ret_relocs.last()
                    && last_ret.offset as usize == self.ci.code.len() - 5
                {
                    self.ci.code.truncate(self.ci.code.len() - 5);
                    self.ci.ret_relocs.pop();
                }
                let len = self.ci.code.len() as u32;
                for rel in self.ci.ret_relocs.drain(ret_reloc_base..) {
                    rel.apply_jump(&mut self.ci.code, len, 0);
                }

                return Some(Value { ty: sig.ret, loc });
            }
            E::Directive { name: "TypeOf", args: [expr], .. } => {
                Some(Value::ty(self.infer_type(expr)))
            }
            E::Directive { name: "eca", args: [ret_ty, args @ ..], .. } => {
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

                let loc = self.alloc_ret(ty, ctx, false);

                self.ci.emit(eca());

                self.load_ret(ty, &loc);

                return Some(Value { ty, loc });
            }
            E::Directive { name: "sizeof", args: [ty], .. } => {
                let ty = self.ty(ty);
                return Some(Value::imm(self.tys.size_of(ty) as _));
            }
            E::Directive { name: "alignof", args: [ty], .. } => {
                let ty = self.ty(ty);
                return Some(Value::imm(self.tys.align_of(ty) as _));
            }
            E::Directive { name: "intcast", args: [val], .. } => {
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
                    self.ci.emit(op(reg.get(), reg.get()));
                    val.loc = Loc::reg(reg);
                }

                Some(Value { ty, loc: val.loc })
            }
            E::Directive { name: "bitcast", args: [val], .. } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, @bitcast(<expr>))`",
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
            E::Directive { name: "as", args: [ty, val], .. } => {
                let ty = self.ty(ty);
                ctx.ty = Some(ty);
                return self.expr_ctx(val, ctx);
            }
            E::Bool { value, .. } => {
                Some(Value { ty: ty::BOOL.into(), loc: Loc::ct(value as u64) })
            }
            E::Idk { pos } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "`idk` can be used only when type can be inferred, use @as(<type>, idk)",
                    );
                };

                if ctx.loc.is_some() {
                    // self.report(
                    //     pos,
                    //     format_args!(
                    //         "`idk` would be written to an existing memory location \
                    //         which at ths point does notthing so its prohibited. TODO: make debug \
                    //         builds write 0xAA instead. Info for weak people: {:?}",
                    //         ctx.loc
                    //     ),
                    // );
                }

                let loc = match self.tys.size_of(ty) {
                    0 => Loc::default(),
                    1..=8 => Loc::reg(self.ci.regs.allocate()),
                    size => Loc::stack(self.ci.stack.allocate(size)),
                };

                Some(Value { ty, loc })
            }
            E::String { pos, mut literal } => {
                literal = literal.trim_matches('"');

                if !literal.ends_with("\\0") {
                    self.report(pos, "string literal must end with null byte (for now)");
                }

                let report = |bytes: &std::str::Bytes, message| {
                    self.report(pos + (literal.len() - bytes.len()) as u32 - 1, message)
                };

                let mut str = Vec::<u8>::with_capacity(literal.len());

                let decode_braces = |str: &mut Vec<u8>, bytes: &mut std::str::Bytes| {
                    while let Some(b) = bytes.next()
                        && b != b'}'
                    {
                        let c = bytes
                            .next()
                            .unwrap_or_else(|| report(bytes, "incomplete escape sequence"));
                        let decode = |b: u8| match b {
                            b'0'..=b'9' => b - b'0',
                            b'a'..=b'f' => b - b'a' + 10,
                            b'A'..=b'F' => b - b'A' + 10,
                            _ => report(bytes, "expected hex digit or '}'"),
                        };
                        str.push(decode(b) << 4 | decode(c));
                    }
                };

                let mut bytes = literal.bytes();
                while let Some(b) = bytes.next() {
                    if b != b'\\' {
                        str.push(b);
                        continue;
                    }
                    let b = match bytes
                        .next()
                        .unwrap_or_else(|| report(&bytes, "incomplete escape sequence"))
                    {
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'\\' => b'\\',
                        b'\'' => b'\'',
                        b'"' => b'"',
                        b'0' => b'\0',
                        b'{' => {
                            decode_braces(&mut str, &mut bytes);
                            continue;
                        }
                        _ => report(&bytes, "unknown escape sequence, expected [nrt\\\"'{0]"),
                    };
                    str.push(b);
                }

                let reloc = Reloc::new(self.ci.code.len() as _, 3, 4);
                let glob = self.tys.globals.len() as ty::Global;
                self.tys.globals.push(Global { data: str, ..Default::default() });
                self.ci
                    .relocs
                    .push(TypedReloc { target: ty::Kind::Global(glob).compress(), reloc });
                let reg = self.ci.regs.allocate();
                self.ci.emit(instrs::lra(reg.get(), 0, 0));
                Some(Value::new(self.tys.make_ptr(ty::U8.into()), reg))
            }
            E::Ctor { pos, ty, fields, .. } => {
                let (ty, loc) = self.prepare_struct_ctor(pos, ctx, ty, fields.len());

                let ty::Kind::Struct(stru) = ty.expand() else {
                    self.report(
                        pos,
                        "our current technology does not (yet) allow\
                        us to construct '{}' with struct constructor",
                    );
                };
                for &CtorField { pos, name, ref value, .. } in fields {
                    let Some((offset, ty)) = self.tys.offset_of(stru, name) else {
                        self.report(pos, format_args!("field not found: {name:?}"));
                    };
                    let loc = loc.as_ref().offset(offset);
                    let value = self.expr_ctx(value, Ctx::default().with_loc(loc).with_ty(ty))?;
                    self.ci.free_loc(value.loc);
                }
                return Some(Value { ty, loc });
            }
            E::Tupl { pos, ty, fields, .. } => {
                let (ty, loc) = self.prepare_struct_ctor(pos, ctx, ty, fields.len());

                match ty.expand() {
                    ty::Kind::Struct(stru) => {
                        let mut offset = 0;
                        let sfields = self.tys.structs[stru as usize].fields.clone();
                        for (sfield, field) in sfields.iter().zip(fields) {
                            let loc = loc.as_ref().offset(offset);
                            let ctx = Ctx::default().with_loc(loc).with_ty(sfield.ty);
                            let value = self.expr_ctx(field, ctx)?;
                            self.ci.free_loc(value.loc);
                            offset += self.tys.size_of(sfield.ty);
                            offset = Types::align_up(offset, self.tys.align_of(sfield.ty));
                        }
                    }
                    ty::Kind::Slice(arr) => {
                        let arr = self.tys.arrays[arr as usize];
                        let item_size = self.tys.size_of(arr.ty);
                        for (i, value) in fields.iter().enumerate() {
                            let loc = loc.as_ref().offset(i as u32 * item_size);
                            let value =
                                self.expr_ctx(value, Ctx::default().with_loc(loc).with_ty(arr.ty))?;
                            self.ci.free_loc(value.loc);
                        }
                    }
                    _ => self.report(
                        pos,
                        format_args!(
                            "compiler does not (yet) know how to initialize\
                            '{}' with tuple constructor",
                            self.ty_display(ty)
                        ),
                    ),
                }

                return Some(Value { ty, loc });
            }
            E::Field { target, name: field } => {
                let checkpoint = self.ci.snap();
                let mut tal = self.expr(target)?;

                if let ty::Kind::Ptr(ty) = tal.ty.expand() {
                    tal.ty = self.tys.ptrs[ty as usize].base;
                    tal.loc = tal.loc.into_derefed();
                }

                match tal.ty.expand() {
                    ty::Kind::Struct(idx) => {
                        let Some((offset, ty)) = self.tys.offset_of(idx, field) else {
                            self.report(target.pos(), format_args!("field not found: {field:?}"));
                        };
                        Some(Value { ty, loc: tal.loc.offset(offset) })
                    }
                    ty::Kind::Builtin(ty::TYPE) => {
                        self.ci.free_loc(tal.loc);
                        self.ci.revert(checkpoint);
                        match ty::Kind::from_ty(self.ty(target)) {
                            ty::Kind::Module(idx) => {
                                match self.find_or_declare(target.pos(), idx, Err(field), "") {
                                    ty::Kind::Global(idx) => self.handle_global(idx),
                                    e => Some(Value::ty(e.compress())),
                                }
                            }
                            ty::Kind::Global(idx) => self.handle_global(idx),
                            e => unimplemented!("{e:?}"),
                        }
                    }
                    smh => self.report(
                        target.pos(),
                        format_args!("the field operation is not supported: {smh:?}"),
                    ),
                }
            }
            E::UnOp { op: T::Sub, val, pos } => {
                let value = self.expr(val)?;

                if !value.ty.is_integer() {
                    self.report(pos, format_args!("cant negate '{}'", self.ty_display(value.ty)));
                }

                let size = self.tys.size_of(value.ty);

                let (oper, dst, drop_loc) = if let Some(dst) = &ctx.loc
                    && dst.is_reg()
                    && let Some(dst) = ctx.loc.take()
                {
                    (
                        self.loc_to_reg(&value.loc, size),
                        if dst.is_ref() {
                            self.loc_to_reg(&dst, size)
                        } else {
                            self.loc_to_reg(dst, size)
                        },
                        value.loc,
                    )
                } else {
                    let oper = self.loc_to_reg(value.loc, size);
                    (oper.as_ref(), oper, Loc::default())
                };

                self.ci.emit(neg(dst.get(), oper.get()));
                self.ci.free_loc(drop_loc);

                Some(Value::new(value.ty, dst))
            }
            E::UnOp { op: T::Xor, val, .. } => {
                let val = self.ty(val);
                Some(Value::ty(self.tys.make_ptr(val)))
            }
            E::UnOp { op: T::Band, val, pos } => {
                let mut val = self.expr(val)?;
                let Loc::Rt { derefed: drfd @ true, reg, stack, offset } = &mut val.loc else {
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

                Some(Value { ty: self.tys.make_ptr(val.ty), loc: val.loc })
            }
            E::UnOp { op: T::Mul, val, pos } => {
                let val = self.expr(val)?;
                match val.ty.expand() {
                    ty::Kind::Ptr(ty) => Some(Value {
                        ty: self.tys.ptrs[ty as usize].base,
                        loc: Loc::reg(self.loc_to_reg(val.loc, self.tys.size_of(val.ty)))
                            .into_derefed(),
                    }),
                    _ => self.report(
                        pos,
                        format_args!("expected pointer, got {}", self.ty_display(val.ty)),
                    ),
                }
            }
            E::BinOp { left, op: T::Decl, right } if self.has_ct(left) => {
                let slot_base = self.ct.vm.read_reg(reg::STACK_PTR).0;
                let (cnt, ty) = self.eval_const_low(right, None);
                if self.assign_ct_pattern(left, ty, cnt as _) {
                    self.ct.vm.write_reg(reg::STACK_PTR, slot_base);
                }
                Some(Value::void())
            }
            E::BinOp { left, op: T::Decl, right } => {
                let value = self.expr(right)?;
                self.assign_pattern(left, value)
            }
            E::Call { func: fast, args, .. } => {
                log::trc!("call {fast}");
                let func_ty = self.ty(fast);
                let ty::Kind::Func(mut func) = func_ty.expand() else {
                    self.report(fast.pos(), "can't call this, maybe in the future");
                };

                // TODO: this will be usefull but not now
                let scope = self.ci.vars.len();
                //let mut snap = self.output.snap();
                //snap.sub(&self.ci.snap);
                //let prev_stack_rel = self.ci.stack_relocs.len();
                //let prev_ret_rel = self.ci.ret_relocs.len();
                let sig = self.compute_signature(&mut func, expr.pos(), args)?;
                //self.ci.ret_relocs.truncate(prev_ret_rel);
                //self.ci.stack_relocs.truncate(prev_stack_rel);
                //snap.add(&self.ci.snap);
                //self.output.trunc(&snap);
                self.ci.vars.truncate(scope);

                let fuc = &self.tys.funcs[func as usize];
                let ast = self.files[fuc.file as usize].clone();
                let E::BinOp { right: &E::Closure { args: cargs, .. }, .. } =
                    fuc.expr.get(&ast).unwrap()
                else {
                    unreachable!();
                };

                let mut parama = self.tys.parama(sig.ret);
                let mut values = Vec::with_capacity(args.len());
                let mut sig_args = sig.args.range();
                let mut should_momize = !args.is_empty() && sig.ret == ty::Id::from(ty::TYPE);

                self.assert_arg_count(expr.pos(), args.len(), cargs.len(), "function call");

                for (i, (arg, carg)) in args.iter().zip(cargs).enumerate() {
                    let ty = self.tys.args[sig_args.next().unwrap()];
                    let sym = parser::find_symbol(&ast.symbols, carg.id);
                    if sym.flags & idfl::COMPTIME != 0 {
                        sig_args.next().unwrap();
                        continue;
                    }

                    // TODO: pass the arg as dest
                    let varg = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    _ = self.assert_ty(arg.pos(), varg.ty, ty, format_args!("argument({i})"));
                    self.pass_arg(&varg, &mut parama);
                    values.push(varg.loc);
                    should_momize = false;
                }

                for value in values {
                    self.ci.free_loc(value);
                }

                let loc = self.alloc_ret(sig.ret, ctx, true);

                if should_momize {
                    self.ci.write_trap(trap::Trap::MomizedCall(trap::MomizedCall { func }));
                }

                let reloc = Reloc::new(self.ci.code.len(), 3, 4);
                self.ci.relocs.push(TypedReloc { target: ty::Kind::Func(func).compress(), reloc });
                self.ci.emit(jal(RET_ADDR, ZERO, 0));
                self.make_func_reachable(func);

                if should_momize {
                    self.ci.emit(tx());
                }

                self.load_ret(sig.ret, &loc);
                return Some(Value { ty: sig.ret, loc });
            }
            E::Ident { id, .. } if ident::is_null(id) => Some(Value::ty(id.into())),
            E::Ident { id, .. }
                if let Some((var_index, var)) =
                    self.ci.vars.iter_mut().enumerate().find(|(_, v)| v.id == id) =>
            {
                var.uses_left -= 1;
                let loc = match var.uses_left == 0
                    && !self.ci.loops.last().is_some_and(|l| l.var_count > var_index as u32)
                {
                    true => std::mem::take(&mut var.value.loc),
                    false => var.value.loc.as_ref(),
                };

                Some(Value { ty: self.ci.vars[var_index].value.ty, loc })
            }
            E::Ident { id, name, .. } => match self
                .tys
                .syms
                .get(&SymKey { ident: id, file: self.ci.file })
                .copied()
                .map(ty::Kind::from_ty)
                .unwrap_or_else(|| self.find_or_declare(ident::pos(id), self.ci.file, Ok(id), name))
            {
                ty::Kind::Global(id) => self.handle_global(id),
                tk => Some(Value::ty(tk.compress())),
            },
            E::Return { pos, val, .. } => {
                let ty = if let Some(val) = val {
                    let size = self.ci.ret.map_or(17, |ty| self.tys.size_of(ty));
                    let loc = match size {
                        _ if self.ci.inline_ret_loc != Loc::default() => {
                            Some(self.ci.inline_ret_loc.as_ref())
                        }
                        0 => None,
                        1..=16 => Some(Loc::reg(1)),
                        _ => Some(Loc::reg(self.ci.ret_reg.as_ref()).into_derefed()),
                    };
                    self.expr_ctx(val, Ctx { ty: self.ci.ret, loc })?.ty
                } else {
                    ty::VOID.into()
                };

                match self.ci.ret {
                    None => self.ci.ret = Some(ty),
                    Some(ret) => _ = self.assert_ty(pos, ty, ret, "return type"),
                }

                self.ci.ret_relocs.push(Reloc::new(self.ci.code.len(), 1, 4));
                self.ci.emit(jmp(0));
                None
            }
            E::Block { stmts, .. } => {
                for stmt in stmts {
                    let val = self.expr(stmt)?;
                    self.ci.free_loc(val.loc);
                }
                Some(Value::void())
            }
            E::Number { value, pos, .. } => Some(Value {
                ty: {
                    let ty = ctx.ty.map(ty::Id::strip_pointer).unwrap_or(ty::INT.into());
                    if !ty.is_integer() && !ty.is_pointer() {
                        self.report(
                            pos,
                            format_args!(
                                "this integer was inferred to be '{}' \
                                which does not make sense",
                                self.ty_display(ty)
                            ),
                        );
                    }
                    ty
                },
                loc: Loc::ct(value as u64),
            }),
            E::If { cond, then, mut else_, .. } => {
                let mut then = Some(then);
                let jump_offset;
                if let &E::BinOp { left, op, right } = cond
                    && let ty = self.infer_type(left)
                    && let Some((op, swapped)) = op.cond_op(ty.is_signed())
                {
                    let left = self.expr_ctx(left, Ctx::default())?;
                    let right = self.expr_ctx(right, Ctx::default())?;
                    let lsize = self.tys.size_of(left.ty);
                    let rsize = self.tys.size_of(right.ty);
                    let left_reg = self.loc_to_reg(&left.loc, lsize);
                    let right_reg = self.loc_to_reg(&right.loc, rsize);
                    jump_offset = self.ci.code.len();
                    self.ci.emit(op(left_reg.get(), right_reg.get(), 0));
                    self.ci.free_loc(left.loc);
                    self.ci.free_loc(right.loc);
                    self.ci.regs.free(left_reg);
                    self.ci.regs.free(right_reg);
                    if swapped {
                        std::mem::swap(&mut then, &mut else_);
                    }
                } else {
                    let cond = self.expr_ctx(cond, Ctx::default().with_ty(ty::BOOL))?;
                    let reg = self.loc_to_reg(&cond.loc, 1);
                    jump_offset = self.ci.code.len();
                    self.ci.emit(jeq(reg.get(), 0, 0));
                    self.ci.free_loc(cond.loc);
                    self.ci.regs.free(reg);
                }

                let then_unreachable =
                    if let Some(then) = then { self.expr(then).is_none() } else { false };
                let mut else_unreachable = false;

                let mut jump = self.ci.code.len() as i64 - jump_offset as i64;

                if let Some(else_) = else_ {
                    let else_jump_offset = self.ci.code.len();
                    if !then_unreachable {
                        self.ci.emit(jmp(0));
                        jump = self.ci.code.len() as i64 - jump_offset as i64;
                    }

                    else_unreachable = self.expr(else_).is_none();

                    if !then_unreachable {
                        let jump = self.ci.code.len() as i64 - else_jump_offset as i64;
                        write_reloc(&mut self.ci.code, else_jump_offset + 1, jump, 4);
                    }
                }

                write_reloc(&mut self.ci.code, jump_offset + 3, jump, 2);

                (!then_unreachable || !else_unreachable).then_some(Value::void())
            }
            E::Loop { body, .. } => 'a: {
                let loop_start = self.ci.code.len();
                self.ci.loops.push(Loop {
                    var_count: self.ci.vars.len() as _,
                    offset: loop_start as _,
                    reloc_base: self.ci.loop_relocs.len() as u32,
                });
                let body_unreachable = self.expr(body).is_none();

                if !body_unreachable {
                    let loop_end = self.ci.code.len();
                    self.ci.emit(jmp(loop_start as i32 - loop_end as i32));
                }

                let loop_end = self.ci.code.len() as u32;

                let loopa = self.ci.loops.pop().unwrap();
                let is_unreachable = loopa.reloc_base == self.ci.loop_relocs.len() as u32;
                for reloc in self.ci.loop_relocs.drain(loopa.reloc_base as usize..) {
                    let off = reloc.apply_jump(&mut self.ci.code, loop_end, 0);
                    debug_assert!(off > 0);
                }

                let mut vars = std::mem::take(&mut self.ci.vars);
                for var in vars.drain(loopa.var_count as usize..) {
                    self.ci.free_loc(var.value.loc);
                }
                self.ci.vars = vars;

                if is_unreachable {
                    break 'a None;
                }

                Some(Value::void())
            }
            E::Break { .. } => {
                self.ci.loop_relocs.push(Reloc::new(self.ci.code.len(), 1, 4));
                self.ci.emit(jmp(0));
                None
            }
            E::Continue { .. } => {
                let loop_ = self.ci.loops.last().unwrap();
                let offset = self.ci.code.len();
                self.ci.emit(jmp(loop_.offset as i32 - offset as i32));
                None
            }
            E::BinOp { left, op: op @ (T::And | T::Or), right } => {
                let lhs = self.expr_ctx(left, Ctx::default().with_ty(ty::BOOL))?;
                let lhs = self.loc_to_reg(lhs.loc, 1);
                let jump_offset = self.ci.code.len() + 3;
                let op = if op == T::And { jeq } else { jne };
                self.ci.emit(op(lhs.get(), 0, 0));

                if let Some(rhs) = self.expr_ctx(right, Ctx::default().with_ty(ty::BOOL)) {
                    let rhs = self.loc_to_reg(rhs.loc, 1);
                    self.ci.emit(cp(lhs.get(), rhs.get()));
                }

                let jump = self.ci.code.len() as i64 - jump_offset as i64;
                write_reloc(&mut self.ci.code, jump_offset, jump, 2);

                Some(Value { ty: ty::BOOL.into(), loc: Loc::reg(lhs) })
            }
            E::BinOp { left, op, right } if op != T::Decl => 'ops: {
                let left = self.expr_ctx(left, Ctx {
                    ty: ctx.ty.filter(|_| op.is_homogenous()),
                    ..Default::default()
                })?;

                if op == T::Assign {
                    let value = self.expr_ctx(right, Ctx::from(left)).unwrap();
                    self.ci.free_loc(value.loc);
                    return Some(Value::void());
                }

                if let ty::Kind::Struct(_) = left.ty.expand() {
                    let right = self.expr_ctx(right, Ctx::default().with_ty(left.ty))?;
                    _ = self.assert_ty(expr.pos(), right.ty, left.ty, "right struct operand");
                    return self.struct_op(op, left.ty, ctx, left.loc, right.loc);
                }

                let lsize = self.tys.size_of(left.ty);

                let (mut lhs, dst, drop_loc) = if let Some(dst) = &ctx.loc
                    && dst.is_reg()
                    && let Some(dst) = ctx.loc.take()
                {
                    (
                        self.loc_to_reg(&left.loc, lsize),
                        if dst.is_ref() {
                            self.loc_to_reg(&dst, lsize)
                        } else {
                            self.loc_to_reg(dst, lsize)
                        },
                        left.loc,
                    )
                } else {
                    let lhs = self.loc_to_reg(left.loc, lsize);
                    (lhs.as_ref(), lhs, Loc::default())
                };
                let right = self.expr_ctx(right, Ctx::default().with_ty(left.ty))?;
                let rsize = self.tys.size_of(right.ty);

                let ty = self.assert_ty(expr.pos(), right.ty, left.ty, "right sclalar operand");
                let size = self.tys.size_of(ty);
                let signed = ty.is_signed();

                if let Loc::Ct { value: CtValue(mut imm), derefed } = right.loc
                    && let Some(oper) = op.imm_binop(signed, size)
                {
                    if derefed {
                        let mut dst = [0u8; 8];
                        dst[..size as usize].copy_from_slice(unsafe {
                            std::slice::from_raw_parts(imm as _, rsize as usize)
                        });
                        imm = u64::from_ne_bytes(dst);
                    }
                    if matches!(op, T::Add | T::Sub)
                        && let ty::Kind::Ptr(ty) = ty::Kind::from_ty(ty)
                    {
                        let size = self.tys.size_of(self.tys.ptrs[ty as usize].base);
                        imm *= size as u64;
                    }

                    self.ci.emit(oper(dst.get(), lhs.get(), imm));
                    self.ci.regs.free(lhs);
                    self.ci.free_loc(drop_loc);
                    break 'ops Some(Value::new(ty, dst));
                }

                let mut rhs = self.loc_to_reg(&right.loc, rsize);
                if matches!(op, T::Add | T::Sub) {
                    let min_size = lsize.min(rsize);
                    if ty.is_signed() && min_size < size {
                        let operand = if lsize < rsize {
                            lhs = self.cow_reg(lhs);
                            lhs.get()
                        } else {
                            rhs = self.cow_reg(rhs);
                            rhs.get()
                        };
                        let op = [sxt8, sxt16, sxt32][min_size.ilog2() as usize];
                        self.ci.emit(op(operand, operand));
                    }

                    if left.ty.is_pointer() ^ right.ty.is_pointer() {
                        let (offset, ty) = if left.ty.is_pointer() {
                            rhs = self.cow_reg(rhs);
                            (rhs.get(), left.ty)
                        } else {
                            lhs = self.cow_reg(lhs);
                            (lhs.get(), right.ty)
                        };

                        let ty::Kind::Ptr(ty) = ty.expand() else { unreachable!() };

                        let size = self.tys.size_of(self.tys.ptrs[ty as usize].base);
                        self.ci.emit(muli64(offset, offset, size as _));
                    }
                }

                if let Some(op) = op.binop(signed, size) {
                    self.ci.emit(op(dst.get(), lhs.get(), rhs.get()));
                    self.ci.regs.free(lhs);
                    self.ci.regs.free(rhs);
                    self.ci.free_loc(right.loc);
                    self.ci.free_loc(drop_loc);
                    break 'ops Some(Value::new(ty, dst));
                }

                'cmp: {
                    let against = match op {
                        T::Le | T::Gt => 1,
                        T::Ne | T::Eq => 0,
                        T::Ge | T::Lt => (-1i64) as _,
                        _ => break 'cmp,
                    };

                    let op_fn = if signed { cmps } else { cmpu };
                    self.ci.emit(op_fn(dst.get(), lhs.get(), rhs.get()));
                    self.ci.emit(cmpui(dst.get(), dst.get(), against));
                    if matches!(op, T::Eq | T::Lt | T::Gt) {
                        self.ci.emit(not(dst.get(), dst.get()));
                    }

                    self.ci.regs.free(lhs);
                    self.ci.regs.free(rhs);
                    self.ci.free_loc(right.loc);
                    self.ci.free_loc(drop_loc);
                    break 'ops Some(Value::new(ty::BOOL, dst));
                }

                unimplemented!("{:#?}", op)
            }
            E::Comment { .. } => Some(Value::void()),
            ref ast => self.report_unhandled_ast(ast, "expression"),
        }?;

        if let Some(ty) = ctx.ty {
            _ = self.assert_ty(expr.pos(), value.ty, ty, format_args!("'{expr}'"));
        }

        Some(match ctx.loc {
            Some(dest) => {
                self.store_typed(value.loc, dest, value.ty);
                Value { ty: value.ty, loc: Loc::ct(0) }
            }
            None => value,
        })
    }

    fn compute_signature(&mut self, func: &mut ty::Func, pos: Pos, args: &[Expr]) -> Option<Sig> {
        let fuc = &self.tys.funcs[*func as usize];
        let fast = self.files[fuc.file as usize].clone();
        let Expr::BinOp { right: &Expr::Closure { args: cargs, ret, .. }, .. } =
            fuc.expr.get(&fast).unwrap()
        else {
            unreachable!();
        };

        Some(if let Some(sig) = fuc.sig {
            sig
        } else {
            let arg_base = self.tys.args.len();

            for (arg, carg) in args.iter().zip(cargs) {
                let ty = self.ty(&carg.ty);
                self.tys.args.push(ty);
                let sym = parser::find_symbol(&fast.symbols, carg.id);
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
                    id: carg.id,
                    value: Value { ty, loc },
                    uses_left: idfl::count(sym.flags) as u32,
                });
            }

            let args = self.pack_args(pos, arg_base);
            let ret = self.ty(ret);

            let sym = SymKey { file: !args.repr(), ident: ty::Kind::Func(*func).compress().repr() };
            let ct = || {
                let func_id = self.tys.funcs.len();
                let fuc = &self.tys.funcs[*func as usize];
                self.tys.funcs.push(Func {
                    file: fuc.file,
                    sig: Some(Sig { args, ret }),
                    expr: fuc.expr,
                    ..Default::default()
                });

                ty::Kind::Func(func_id as _).compress()
            };
            *func = self.tys.syms.entry(sym).or_insert_with(ct).expand().inner();

            Sig { args, ret }
        })
    }

    fn has_ct(&self, expr: &Expr) -> bool {
        expr.has_ct(&self.cfile().symbols)
    }

    fn infer_type(&mut self, expr: &Expr) -> ty::Id {
        // FIXME: very inneficient
        let mut ci = ItemCtx {
            file: self.ci.file,
            id: self.ci.id,
            ret: self.ci.ret,
            task_base: self.ci.task_base,
            loops: self.ci.loops.clone(),
            vars: self
                .ci
                .vars
                .iter()
                .map(|v| Variable {
                    id: v.id,
                    value: Value { ty: v.value.ty, loc: v.value.loc.as_ref() },
                    uses_left: v.uses_left,
                })
                .collect(),
            stack_relocs: self.ci.stack_relocs.clone(),
            ret_relocs: self.ci.ret_relocs.clone(),
            loop_relocs: self.ci.loop_relocs.clone(),
            ..Default::default()
        };
        ci.regs.init();
        std::mem::swap(&mut self.ci, &mut ci);
        let value = self.expr(expr).unwrap();
        self.ci.free_loc(value.loc);
        std::mem::swap(&mut self.ci, &mut ci);
        value.ty
    }

    fn eval_const(&mut self, expr: &Expr, ty: impl Into<ty::Id>) -> u64 {
        self.eval_const_low(expr, Some(ty.into())).0
    }

    fn eval_const_low(&mut self, expr: &Expr, mut ty: Option<ty::Id>) -> (u64, ty::Id) {
        let mut ci = ItemCtx {
            file: self.ci.file,
            id: ty::Kind::Builtin(u32::MAX),
            ret: ty,
            ..self.pool.cis.pop().unwrap_or_default()
        };
        ci.vars.append(&mut self.ci.vars);

        let loc = self.ct_eval(ci, |s, prev| {
            s.ci.emit_prelude();

            if s.ci.ret.map_or(true, |r| s.tys.size_of(r) > 16) {
                let reg = s.ci.regs.allocate();
                s.ci.emit(instrs::cp(reg.get(), 1));
                s.ci.ret_reg = reg;
            };

            let ctx = Ctx { loc: None, ty: s.ci.ret };
            if s.expr_ctx(&Expr::Return { pos: 0, val: Some(expr) }, ctx).is_some() {
                s.report(expr.pos(), "we fucked up");
            };

            ty = s.ci.ret;

            s.complete_call_graph();

            prev.vars.append(&mut s.ci.vars);
            s.ci.finalize();
            s.ci.emit(tx());

            Ok(1)
        });

        match loc {
            Ok(i) | Err(i) => {
                (self.ct.vm.read_reg(i).cast::<u64>(), ty.expect("you have died (in brahmaputra)"))
            }
        }
    }

    fn assign_ct_pattern(&mut self, pat: &Expr, ty: ty::Id, offset: *mut u8) -> bool {
        let size = self.tys.size_of(ty);
        match *pat {
            Expr::Ident { id, .. }
                if find_symbol(&self.cfile().symbols, id).flags & idfl::REFERENCED == 0
                    && size <= 8 =>
            {
                let loc = Loc::ct(load_value(offset, size));
                self.ci.vars.push(Variable { id, value: Value { ty, loc }, uses_left: u32::MAX });
                true
            }
            Expr::Ident { id, .. } => {
                let var = Variable {
                    id,
                    value: Value { ty, loc: Loc::ct_ptr(offset as _) },
                    uses_left: u32::MAX,
                };
                self.ci.vars.push(var);
                false
            }
            ref pat => self.report_unhandled_ast(pat, "comptime pattern"),
        }
    }

    fn assign_pattern(&mut self, pat: &Expr, right: Value) -> Option<Value> {
        match *pat {
            Expr::Ident { id, .. } => {
                let mut loc = self.make_loc_owned(right.loc, right.ty);
                let sym = parser::find_symbol(&self.cfile().symbols, id).flags;
                if sym & idfl::REFERENCED != 0 {
                    loc = self.spill(loc, self.tys.size_of(right.ty));
                }
                self.ci.vars.push(Variable {
                    id,
                    value: Value { ty: right.ty, loc },
                    uses_left: idfl::count(sym) as u32,
                });
            }
            Expr::Ctor { pos, fields, .. } => {
                let ty::Kind::Struct(idx) = right.ty.expand() else {
                    self.report(pos, "can't use struct destruct on non struct value (TODO: shold work with modules)");
                };

                for &CtorField { pos, name, ref value } in fields {
                    let Some((offset, ty)) = self.tys.offset_of(idx, name) else {
                        self.report(pos, format_args!("field not found: {name:?}"));
                    };
                    let loc = self.ci.dup_loc(&right.loc).offset(offset);
                    self.assign_pattern(value, Value::new(ty, loc));
                }

                self.ci.free_loc(right.loc);
            }
            ref pat => self.report_unhandled_ast(pat, "pattern"),
        };

        Some(Value::void())
    }

    fn prepare_struct_ctor(
        &mut self,
        pos: Pos,
        ctx: Ctx,
        ty: Option<&Expr>,
        field_len: usize,
    ) -> (ty::Id, Loc) {
        let Some(mut ty) = ty.map(|ty| self.ty(ty)).or(ctx.ty) else {
            self.report(pos, "expected type, (it cannot be inferred)");
        };

        match ty.expand() {
            ty::Kind::Struct(stru) => {
                let field_count = self.tys.structs[stru as usize].fields.len();
                if field_count != field_len {
                    self.report(
                        pos,
                        format_args!("expected {field_count} fields, got {field_len}"),
                    );
                }
            }
            ty::Kind::Slice(arr) => {
                let arr = &self.tys.arrays[arr as usize];
                if arr.len == ArrayLen::MAX {
                    ty = self.tys.make_array(arr.ty, field_len as _);
                } else if arr.len != field_len as u32 {
                    self.report(
                        pos,
                        format_args!(
                            "literal has {} elements, but explicit array type has {} elements",
                            arr.len, field_len
                        ),
                    );
                }
            }
            _ => self.report(pos, "expected expression to evaluate to struct (or array maybe)"),
        }

        let size = self.tys.size_of(ty);
        let loc = ctx.loc.unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(size)));
        (ty, loc)
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
                let ctx = Ctx::from(Value { ty, loc: loc.as_ref().offset(offset) });
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

        if let Loc::Ct { value, derefed: false } = right
            && let Some(op) = op.imm_binop(signed, size)
        {
            self.ci.emit(op(lhs.get(), lhs.get(), value.0));
            return Some(if let Some(value) = ctx.into_value() {
                self.store_typed(Loc::reg(lhs.as_ref()), value.loc, value.ty);
                Value::void()
            } else {
                Value { ty, loc: Loc::reg(lhs) }
            });
        }

        let rhs = self.loc_to_reg(right, size);

        if let Some(op) = op.binop(signed, size) {
            self.ci.emit(op(lhs.get(), lhs.get(), rhs.get()));
            self.ci.regs.free(rhs);
            return if let Some(value) = ctx.into_value() {
                self.store_typed(Loc::reg(lhs), value.loc, value.ty);
                Some(Value::void())
            } else {
                Some(Value { ty, loc: Loc::reg(lhs) })
            };
        }

        unimplemented!("{:#?}", op)
    }

    fn handle_global(&mut self, id: ty::Global) -> Option<Value> {
        let ptr = self.ci.regs.allocate();

        let reloc = Reloc::new(self.ci.code.len(), 3, 4);
        let global = &mut self.tys.globals[id as usize];
        self.ci.relocs.push(TypedReloc { target: ty::Kind::Global(id).compress(), reloc });
        self.ci.emit(instrs::lra(ptr.get(), 0, 0));

        Some(Value { ty: global.ty, loc: Loc::reg(ptr).into_derefed() })
    }

    fn spill(&mut self, loc: Loc, size: Size) -> Loc {
        if loc.is_ref() || !loc.is_stack() {
            let stack = Loc::stack(self.ci.stack.allocate(size));
            self.store_sized(loc, &stack, size);
            stack
        } else {
            loc
        }
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

    fn complete_call_graph(&mut self) {
        while self.ci.task_base < self.tasks.len()
            && let Some(task_slot) = self.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.handle_task(task);
        }
    }

    fn handle_task(&mut self, FTask { file, id }: FTask) {
        let func = &self.tys.funcs[id as usize];
        debug_assert!(func.file == file);
        let sig = func.sig.unwrap();
        let ast = self.files[file as usize].clone();
        let expr = func.expr.get(&ast).unwrap();
        let ct_stack_base = self.ct.vm.read_reg(reg::STACK_PTR).0;

        let repl = ItemCtx {
            file,
            id: ty::Kind::Func(id),
            ret: Some(sig.ret),
            ..self.pool.cis.pop().unwrap_or_default()
        };
        let prev_ci = std::mem::replace(&mut self.ci, repl);
        self.ci.regs.init();

        let Expr::BinOp {
            left: Expr::Ident { .. },
            op: TokenKind::Decl,
            right: &Expr::Closure { body, args, .. },
        } = expr
        else {
            unreachable!("{expr}")
        };

        self.ci.emit_prelude();

        let mut parama = self.tys.parama(sig.ret);
        let mut sig_args = sig.args.range();
        for arg in args.iter() {
            let ty = self.tys.args[sig_args.next().unwrap()];
            let sym = parser::find_symbol(&ast.symbols, arg.id).flags;
            let loc = match sym & idfl::COMPTIME != 0 {
                true => Loc::ty(self.tys.args[sig_args.next().unwrap()]),
                false => self.load_arg(sym, ty, &mut parama),
            };
            self.ci.vars.push(Variable {
                id: arg.id,
                value: Value { ty, loc },
                uses_left: idfl::count(sym) as u32,
            });
        }

        if self.tys.size_of(sig.ret) > 16 {
            let reg = self.ci.regs.allocate();
            self.ci.emit(instrs::cp(reg.get(), 1));
            self.ci.ret_reg = reg;
        } else {
            self.ci.ret_reg = reg::Id::RET;
        }

        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        for vars in self.ci.vars.drain(..).collect::<Vec<_>>() {
            self.ci.free_loc(vars.value.loc);
        }

        self.ci.finalize();
        self.ci.emit(jala(ZERO, RET_ADDR, 0));
        self.ci.regs.free(std::mem::take(&mut self.ci.ret_reg));
        self.tys.funcs[id as usize].code.append(&mut self.ci.code);
        self.tys.funcs[id as usize].relocs.append(&mut self.ci.relocs);
        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
        self.ct.vm.write_reg(reg::STACK_PTR, ct_stack_base);
    }

    fn load_arg(&mut self, flags: parser::IdentFlags, ty: ty::Id, parama: &mut ParamAlloc) -> Loc {
        let size = self.tys.size_of(ty) as Size;
        if size == 0 {
            return Loc::default();
        }
        let (src, dst) = match size {
            0 => (Loc::default(), Loc::default()),
            ..=8 if flags & idfl::REFERENCED == 0 => {
                (Loc::reg(parama.next()), Loc::reg(self.ci.regs.allocate()))
            }
            1..=8 => (Loc::reg(parama.next()), Loc::stack(self.ci.stack.allocate(size))),
            9..=16 => (Loc::reg(parama.next_wide()), Loc::stack(self.ci.stack.allocate(size))),
            _ if flags & (idfl::MUTABLE | idfl::REFERENCED) == 0 => {
                let ptr = parama.next();
                let reg = self.ci.regs.allocate();
                self.ci.emit(instrs::cp(reg.get(), ptr));
                return Loc::reg(reg).into_derefed();
            }
            _ => (Loc::reg(parama.next()).into_derefed(), Loc::stack(self.ci.stack.allocate(size))),
        };

        self.store_sized(src, &dst, size);
        dst
    }

    fn eca(&mut self, trap: trap::Trap, ret: impl Into<ty::Id>) -> Value {
        self.ci.write_trap(trap);
        Value { ty: ret.into(), loc: Loc::reg(1) }
    }

    fn alloc_ret(&mut self, ret: ty::Id, ctx: Ctx, custom_ret_reg: bool) -> Loc {
        let size = self.tys.size_of(ret);
        if size == 0 {
            debug_assert!(ctx.loc.is_none(), "{}", self.ty_display(ret));
            return Loc::default();
        }

        if ctx.loc.is_some() && size < 16 {
            return ctx.loc.unwrap();
        }

        match size {
            0 => Loc::default(),
            1..=8 if custom_ret_reg => Loc::reg(self.ci.regs.allocate()),
            1..=8 => Loc::reg(1),
            9..=16 => Loc::stack(self.ci.stack.allocate(size)),
            17.. => {
                let loc = ctx.loc.unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(size)));
                let Loc::Rt { reg, stack, offset, .. } = &loc else {
                    todo!("old man with the beard looks at the sky scared");
                };
                self.stack_offset(1, reg.get(), stack.as_ref(), *offset);
                loc
            }
        }
    }

    fn loc_to_reg(&mut self, loc: impl Into<LocCow>, size: Size) -> reg::Id {
        match loc.into() {
            LocCow::Owned(Loc::Rt { derefed: false, mut reg, offset, stack }) => {
                debug_assert!(stack.is_none(), "TODO");
                assert_eq!(offset, 0, "TODO");
                if reg.is_ref() {
                    let new_reg = self.ci.regs.allocate();
                    debug_assert_ne!(reg.get(), 0);
                    self.ci.emit(cp(new_reg.get(), reg.get()));
                    reg = new_reg;
                }
                reg
            }
            LocCow::Ref(&Loc::Rt { derefed: false, ref reg, offset, ref stack }) => {
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
            let Loc::Rt { reg, stack, offset, .. } = loc else { unreachable!() };
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

        if size == 0 {
            return;
        }

        src.as_ref().assert_valid();
        dst.as_ref().assert_valid();

        match (src.as_ref(), dst.as_ref()) {
            (&Loc::Ct { value, derefed }, lpat!(true, reg, off, ref sta)) => {
                let ct = self.ci.regs.allocate();
                self.ci.emit(li64(ct.get(), ensure_loaded(value, derefed, size)));
                let off = self.opt_stack_reloc(sta.as_ref(), off, 3);
                self.ci.emit(st(ct.get(), reg.get(), off, size as _));
                self.ci.regs.free(ct);
            }
            (&Loc::Ct { value, derefed }, lpat!(false, reg, 0, None)) => {
                self.ci.emit(li64(reg.get(), ensure_loaded(value, derefed, size)))
            }
            (&Loc::Ct { value, derefed }, lpat!(false, reg, 8, None))
                if reg.get() == 1 && size == 8 =>
            {
                self.ci.emit(li64(reg.get() + 1, ensure_loaded(value, derefed, size)));
            }
            (&Loc::Ct { value, derefed }, lpat!(false, reg, off, None)) if reg.get() == 1 => {
                let freg = reg.get() + (off / 8) as u8;
                let mask = !(((1u64 << (8 * size)) - 1) << (8 * (off % 8)));
                self.ci.emit(andi(freg, freg, mask));
                let value = ensure_loaded(value, derefed, size) << (8 * (off % 8));
                self.ci.emit(ori(freg, freg, value));
            }
            (lpat!(true, src, soff, ref ssta), lpat!(true, dst, doff, ref dsta)) => {
                // TODO: some oportuinies to ellit more optimal code
                let src_off = if src.is_ref() { self.ci.regs.allocate() } else { src.as_ref() };
                let dst_off = if dst.is_ref() { self.ci.regs.allocate() } else { dst.as_ref() };
                self.stack_offset(src_off.get(), src.get(), ssta.as_ref(), soff);
                self.stack_offset(dst_off.get(), dst.get(), dsta.as_ref(), doff);
                self.ci.emit(bmc(src_off.get(), dst_off.get(), size as _));
                self.ci.regs.free(src_off);
                self.ci.regs.free(dst_off);
            }
            (lpat!(false, src, 0, None), lpat!(false, dst, 0, None)) => {
                if src != dst {
                    debug_assert_ne!(src.get(), 0);
                    self.ci.emit(cp(dst.get(), src.get()));
                }
            }
            (lpat!(true, src, soff, ref ssta), lpat!(false, dst, 0, None)) => {
                if size < 8 {
                    self.ci.emit(cp(dst.get(), 0));
                }
                let off = self.opt_stack_reloc(ssta.as_ref(), soff, 3);
                self.ci.emit(ld(dst.get(), src.get(), off, size as _));
            }
            (lpat!(false, src, 0, None), lpat!(true, dst, doff, ref dsta)) => {
                let off = self.opt_stack_reloc(dsta.as_ref(), doff, 3);
                self.ci.emit(st(src.get(), dst.get(), off, size as _))
            }
            (a, b) => unreachable!("{a:?} {b:?}"),
        }

        self.ci.free_loc(src);
        self.ci.free_loc(dst);
    }

    fn stack_offset(&mut self, dst: u8, op: u8, stack: Option<&stack::Id>, off: Offset) {
        let Some(stack) = stack else {
            self.ci.emit_addi(dst, op, off as _);
            return;
        };

        let off = self.stack_reloc(stack, off, 3);
        self.ci.emit(addi64(dst, op, off));
    }

    fn opt_stack_reloc(&mut self, stack: Option<&stack::Id>, off: Offset, sub_offset: u8) -> u64 {
        stack.map(|s| self.stack_reloc(s, off, sub_offset)).unwrap_or(off as _)
    }

    fn stack_reloc(&mut self, stack: &stack::Id, off: Offset, sub_offset: u8) -> u64 {
        let offset = self.ci.code.len();
        self.ci.stack_relocs.push(Reloc::new(offset, sub_offset, 8));
        Reloc::pack_srel(stack, off)
    }

    // TODO: sometimes its better to do this in bulk
    fn ty(&mut self, expr: &Expr) -> ty::Id {
        ty::Id::from(self.eval_const(expr, ty::TYPE))
    }

    fn read_trap(addr: u64) -> Option<&'static trap::Trap> {
        // TODO: make this debug only
        if unsafe { *(addr as *const u8) } != 255 {
            return None;
        }
        Some(unsafe { &*((addr + 1) as *const trap::Trap) })
    }

    fn handle_ecall(&mut self) {
        let trap = Self::read_trap(self.ct.vm.pc.get()).unwrap();
        self.ct.vm.pc = self.ct.vm.pc.wrapping_add(trap.size() + 1);

        let mut extra_jump = 0;
        let mut code_index = Some(self.ct.vm.pc.get() as usize - self.ci.code.as_ptr() as usize);

        match *trap {
            trap::Trap::MakeStruct(trap::MakeStruct { file, struct_expr }) => {
                let cfile = self.files[file as usize].clone();
                let &Expr::Struct { fields, captured, .. } = struct_expr.get(&cfile).unwrap()
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
                        value: Value::new(ty, Loc::ct(u64::from_ne_bytes(imm))),
                        uses_left: u32::MAX,
                    });
                }

                let stru = ty::Kind::Struct(self.build_struct(fields)).compress();
                self.ci.vars.truncate(prev_len);
                self.ct.vm.write_reg(1, stru.repr() as u64);
            }
            trap::Trap::MomizedCall(trap::MomizedCall { func }) => {
                let sym = SymKey { file: u32::MAX, ident: ty::Kind::Func(func).compress().repr() };
                if let Some(&ty) = self.tys.syms.get(&sym) {
                    self.ct.vm.write_reg(1, ty.repr());
                    extra_jump = jal(0, 0, 0).0 + tx().0;
                } else {
                    code_index = None;
                    self.run_vm();
                    self.tys.syms.insert(sym, self.ct.vm.read_reg(1).0.into());
                }
            }
        }

        if let Some(lpc) = code_index {
            let offset = lpc + self.ci.code.as_ptr() as usize;
            self.ct.vm.pc = hbvm::mem::Address::new(offset as _);
        }
        self.ct.vm.pc += extra_jump;
    }

    fn find_or_declare(
        &mut self,
        pos: Pos,
        file: FileId,
        name: Result<Ident, &str>,
        lit_name: &str,
    ) -> ty::Kind {
        log::trc!("find_or_declare: {lit_name} {file}");
        let f = self.files[file as usize].clone();
        let Some((expr, ident)) = f.find_decl(name) else {
            match name {
                Ok(_) => self.report(pos, format_args!("undefined indentifier: {lit_name}")),
                Err("main") => self.report(pos, format_args!("missing main function: {f}")),
                Err(name) => self.report(pos, format_args!("undefined indentifier: {name}")),
            }
        };

        if let Some(existing) = self.tys.syms.get(&SymKey { file, ident }) {
            if let ty::Kind::Func(id) = existing.expand()
                && let func = &mut self.tys.funcs[id as usize]
                && let Err(idx) = task::unpack(func.offset)
                && idx < self.tasks.len()
            {
                func.offset = task::id(self.tasks.len());
                let task = self.tasks[idx].take();
                self.tasks.push(task);
            }
            return existing.expand();
        }

        let prev_file = std::mem::replace(&mut self.ci.file, file);
        let sym = match expr {
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: &Expr::Closure { pos, args, ret, .. },
            } => {
                let func = Func {
                    file,
                    sig: 'b: {
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

                        let args = self.pack_args(pos, arg_base);
                        let ret = self.ty(ret);

                        Some(Sig { args, ret })
                    },
                    expr: {
                        let refr = ExprRef::new(expr);
                        debug_assert!(refr.get(&f).is_some());
                        refr
                    },
                    ..Default::default()
                };

                let id = self.tys.funcs.len() as _;
                self.tys.funcs.push(func);

                ty::Kind::Func(id)
            }
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: Expr::Struct { fields, .. },
            } => ty::Kind::Struct(self.build_struct(fields)),
            Expr::BinOp { left, op: TokenKind::Decl, right } => {
                let gid = self.tys.globals.len() as ty::Global;
                self.tys.globals.push(Global { file, name: ident, ..Default::default() });

                let ci = ItemCtx {
                    file,
                    id: ty::Kind::Builtin(u32::MAX),
                    ..self.pool.cis.pop().unwrap_or_default()
                };

                _ = left.find_pattern_path(ident, right, |expr| {
                    self.tys.globals[gid as usize] = self
                        .ct_eval(ci, |s, _| Ok::<_, !>(s.generate_global(expr, file, ident)))
                        .into_ok();
                });

                ty::Kind::Global(gid)
            }
            e => unimplemented!("{e:#?}"),
        };
        self.ci.file = prev_file;
        self.tys.syms.insert(SymKey { ident, file }, sym.compress());
        sym
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let fuc = &mut self.tys.funcs[func as usize];
        if fuc.offset == u32::MAX {
            fuc.offset = task::id(self.tasks.len() as _);
            self.tasks.push(Some(FTask { file: fuc.file, id: func }));
        }
    }

    fn generate_global(&mut self, expr: &Expr, file: FileId, name: Ident) -> Global {
        self.ci.emit_prelude();

        let ret = self.ci.regs.allocate();
        self.ci.emit(instrs::cp(ret.get(), 1));
        self.ci.task_base = self.tasks.len();

        let ctx = Ctx::default().with_loc(Loc::reg(ret).into_derefed());
        let Some(ret) = self.expr_ctx(expr, ctx) else {
            self.report(expr.pos(), "expression is not reachable");
        };

        self.complete_call_graph();

        let mut data = vec![0; self.tys.size_of(ret.ty) as usize];

        self.ci.finalize();
        self.ci.emit(tx());

        self.ct.vm.write_reg(1, data.as_mut_ptr() as u64);

        self.ci.free_loc(ret.loc);

        Global { ty: ret.ty, file, name, data, ..Default::default() }
    }

    fn ct_eval<T, E>(
        &mut self,
        ci: ItemCtx,
        compile: impl FnOnce(&mut Self, &mut ItemCtx) -> Result<T, E>,
    ) -> Result<T, E> {
        log::trc!("eval");

        let mut prev_ci = std::mem::replace(&mut self.ci, ci);
        self.ci.task_base = self.tasks.len();
        self.ci.regs.init();

        let ret = compile(self, &mut prev_ci);
        let mut rr = std::mem::take(&mut self.ci.ret_reg);
        let is_on_stack = !rr.is_ref();
        if !rr.is_ref() {
            self.ci.emit(instrs::cp(1, rr.get()));
            let rref = rr.as_ref();
            self.ci.regs.free(std::mem::replace(&mut rr, rref));
        }

        if ret.is_ok() {
            let last_fn = self.tys.funcs.len();
            self.tys.funcs.push(Default::default());

            self.tys.funcs[last_fn].code.append(&mut self.ci.code);
            self.tys.funcs[last_fn].relocs.append(&mut self.ci.relocs);

            if is_on_stack {
                let size =
                    self.tys.size_of(self.ci.ret.expect("you have died (colaterall fuck up)"));
                let slot = self.ct.vm.read_reg(reg::STACK_PTR).0;
                self.ct.vm.write_reg(reg::STACK_PTR, slot.wrapping_add(size as _));
                self.ct.vm.write_reg(1, slot);
            }

            self.tys.dump_reachable(last_fn as _, &mut self.ct.code);
            let entry = &mut self.ct.code[self.tys.funcs[last_fn].offset as usize] as *mut _ as _;
            let prev_pc = std::mem::replace(&mut self.ct.vm.pc, hbvm::mem::Address::new(entry));

            #[cfg(debug_assertions)]
            {
                let mut vc = Vec::<u8>::new();
                if let Err(e) = self.tys.disasm(&self.ct.code, &self.files, &mut vc, |bts| {
                    if let Some(trap) = Self::read_trap(bts.as_ptr() as _) {
                        bts.take(..trap.size() + 1).unwrap();
                    }
                }) {
                    panic!("{e} {}", String::from_utf8(vc).unwrap());
                } else {
                    //log::inf!("{}", String::from_utf8(vc).unwrap());
                }
            }

            self.run_vm();
            self.ct.vm.pc = prev_pc;

            self.tys.funcs.pop().unwrap();
        }

        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));

        log::trc!("eval-end");

        ret
    }

    pub fn disasm(&mut self, output: &mut impl std::io::Write) -> std::io::Result<()> {
        let mut bin = Vec::new();
        self.assemble(&mut bin);
        self.tys.disasm(&bin, &self.files, output, |_| {})
    }

    fn run_vm(&mut self) {
        loop {
            match self.ct.vm.run().unwrap() {
                hbvm::VmRunOk::End => break,
                hbvm::VmRunOk::Timer => unreachable!(),
                hbvm::VmRunOk::Ecall => self.handle_ecall(),
                hbvm::VmRunOk::Breakpoint => unreachable!(),
            }
        }
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, &self.files, ty)
    }

    #[must_use]
    #[track_caller]
    fn assert_ty(&self, pos: Pos, ty: ty::Id, expected: ty::Id, hint: impl Display) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected) {
            res
        } else {
            let ty = self.ty_display(ty);
            let expected = self.ty_display(expected);
            self.report(pos, format_args!("expected {hint} of type {expected}, got {ty}"));
        }
    }

    fn assert_arg_count(&self, pos: Pos, got: usize, expected: usize, hint: impl Display) {
        if got != expected {
            let s = if expected != 1 { "s" } else { "" };
            self.report(pos, format_args!("{hint} expected {expected} argument{s}, got {got}"))
        }
    }

    fn report_log(&self, pos: Pos, msg: impl std::fmt::Display) {
        let mut out = String::new();
        self.cfile().report_to(pos, msg, &mut out);
        eprintln!("{out}");
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

    fn cow_reg(&mut self, rhs: reg::Id) -> reg::Id {
        if rhs.is_ref() {
            let reg = self.ci.regs.allocate();
            self.ci.emit(cp(reg.get(), rhs.get()));
            reg
        } else {
            rhs
        }
    }

    pub fn assemble(&mut self, buf: &mut Vec<u8>) {
        self.tys.funcs.iter_mut().for_each(|f| f.offset = u32::MAX);
        self.tys.globals.iter_mut().for_each(|g| g.offset = u32::MAX);
        self.tys.assemble(buf)
    }
}

#[cfg(test)]
mod tests {
    const README: &str = include_str!("../README.md");

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        let mut codegen =
            super::Codegen { files: crate::test_parse_files(ident, input), ..Default::default() };

        codegen.generate();
        let mut out = Vec::new();
        codegen.assemble(&mut out);

        let mut buf = Vec::<u8>::new();
        let err = codegen.tys.disasm(&out, &codegen.files, &mut buf, |_| {});
        output.push_str(String::from_utf8(buf).unwrap().as_str());
        if err.is_err() {
            return;
        }

        crate::test_run_vm(&out, output);
    }

    crate::run_tests! { generate:
        arithmetic => README;
        variables => README;
        functions => README;
        comments => README;
        if_statements => README;
        loops => README;
        //fb_driver => README;
        pointers => README;
        structs => README;
        different_types => README;
        struct_operators => README;
        directives => README;
        global_variables => README;
        generic_types => README;
        generic_functions => README;
        c_strings => README;
        idk => README;
        struct_patterns => README;
        arrays => README;
        struct_return_from_module_function => README;
        //comptime_pointers => README;
        sort_something_viredly => README;
        hex_octal_binary_literals => README;
        //comptime_min_reg_leak => README;
        // structs_in_registers => README;
        comptime_function_from_another_file => README;
        inline => README;
        inline_test => README;
        some_generic_code => README;
        integer_inference_issues => README;
        writing_into_string => README;
        request_page => README;
        tests_ptr_to_ptr_copy => README;
    }
}
