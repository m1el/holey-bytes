use {
    crate::{
        ident::{self, Ident},
        instrs::{self, *},
        lexer::TokenKind,
        parser::{
            self, find_symbol, idfl, CommentOr, CtorField, Expr, ExprRef, FileId, Pos, StructField,
        },
        reg,
        ty::{self, TyCheck},
        Comptime, Field, Func, Global, OffsetIter, PLoc, ParamAlloc, Reloc, Sig, Struct, SymKey,
        TypeParser, TypedReloc, Types,
    },
    alloc::{string::String, vec::Vec},
    core::{assert_matches::debug_assert_matches, fmt::Display},
};

type Offset = u32;
type Size = u32;
type ArrayLen = u32;

fn load_value(ptr: *const u8, size: u32) -> u64 {
    let mut dst = [0u8; 8];
    dst[..size as usize]
        .copy_from_slice(unsafe { core::slice::from_raw_parts(ptr, size as usize) });
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
        crate::debug,
        alloc::vec::Vec,
        core::num::NonZeroU32,
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
            if !debug::panicking() && !self.is_ref() {
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
            core::mem::forget(id);
            //if id.is_ref() {}
            //let meta = &mut self.meta[id.index()];
            //meta.rc -= 1;
            //if meta.rc != 0 {
            //    return;
            //}
            //meta.offset = self.height;
            //self.height -= meta.size;
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

mod rall {
    use {
        crate::{debug, reg::*},
        alloc::vec::Vec,
    };

    type Reg = u8;

    #[cfg(all(debug_assertions, feature = "std"))]
    type Bt = std::backtrace::Backtrace;
    #[cfg(not(all(debug_assertions, feature = "std")))]
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

    #[cfg(all(debug_assertions, feature = "std"))]
    impl Drop for Id {
        fn drop(&mut self) {
            if !debug::panicking()
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
                #[cfg(all(debug_assertions, feature = "std"))]
                Some(std::backtrace::Backtrace::capture()),
                #[cfg(not(all(debug_assertions, feature = "std")))]
                Some(()),
            )
        }

        pub fn free(&mut self, mut reg: Id) {
            if reg.1.take().is_some() {
                self.free.push(reg.0);
                core::mem::forget(reg);
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
        Self { ty: ty::Id::VOID, loc: Loc::ct(0) }
    }

    fn imm(value: u64) -> Self {
        Self { ty: ty::Id::UINT, loc: Loc::ct(value) }
    }

    fn ty(ty: ty::Id) -> Self {
        Self { ty: ty::Id::TYPE, loc: Loc::ct(ty.repr() as u64) }
    }
}

enum LocCow<'a> {
    Ref(&'a Loc),
    Owned(Loc),
}

impl LocCow<'_> {
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

impl From<Loc> for LocCow<'_> {
    fn from(value: Loc) -> Self {
        Self::Owned(value)
    }
}

#[repr(packed)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct CtValue(u64);

#[derive(Debug, PartialEq, Eq)]
enum Loc {
    Rt { derefed: bool, reg: rall::Id, stack: Option<stack::Id>, offset: Offset },
    Ct { derefed: bool, value: CtValue },
}

impl Loc {
    fn stack(stack: stack::Id) -> Self {
        Self::Rt { stack: Some(stack), reg: reg::STACK_PTR.into(), derefed: true, offset: 0 }
    }

    fn reg(reg: impl Into<rall::Id>) -> Self {
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

        Some(core::mem::replace(self, self.as_ref()))
    }

    fn is_ref(&self) -> bool {
        matches!(self, Self::Rt { reg, stack, .. } if reg.is_ref() && stack.as_ref().map_or(true, stack::Id::is_ref))
    }

    fn to_ty(&self) -> Option<ty::Id> {
        match *self {
            Self::Ct { derefed: false, value } => Some(ty::Id::from(value.0)),
            Self::Ct { derefed: true, value } => {
                Some(unsafe { core::ptr::read(value.0 as *const u8 as _) })
            }
            Self::Rt { .. } => None,
        }
    }

    fn is_stack(&self) -> bool {
        matches!(self, Self::Rt { derefed: true, reg, stack: Some(_), offset: 0 } if reg.get() == reg::STACK_PTR)
    }

    fn is_reg(&self) -> bool {
        matches!(self, Self::Rt { derefed: false, reg: _, stack, offset } if ({ debug_assert_eq!(*offset,  0); debug_assert_matches!(stack, None); true }))
    }
}

impl From<rall::Id> for Loc {
    fn from(reg: rall::Id) -> Self {
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
    ret: Option<ty::Id>,
    ret_reg: rall::Id,
    inline_ret_loc: Loc,

    task_base: usize,

    stack: stack::Alloc,
    regs: rall::Alloc,

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
        self.code.extend_from_slice(&instr[..len]);
    }

    fn emit_prelude(&mut self) {
        self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
        self.emit(instrs::st(reg::RET_ADDR, reg::STACK_PTR, 0, 0));
    }

    fn emit_entry_prelude(&mut self) {
        self.emit(jal(reg::RET_ADDR, reg::ZERO, 0));
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
    check: TyCheck,
}

impl Ctx {
    pub fn with_loc(self, loc: Loc) -> Self {
        Self { loc: Some(loc), ..self }
    }

    pub fn with_ty(self, ty: impl Into<ty::Id>) -> Self {
        Self { ty: Some(ty.into()), ..self }
    }

    pub fn with_check(self, check: TyCheck) -> Self {
        Self { check, ..self }
    }

    fn into_value(self) -> Option<Value> {
        Some(Value { ty: self.ty.unwrap(), loc: self.loc? })
    }
}

impl From<Value> for Ctx {
    fn from(value: Value) -> Self {
        Self { loc: Some(value.loc), ty: Some(value.ty), ..Default::default() }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
    arg_locs: Vec<Loc>,
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
                        $(Self::$variant(_) => core::mem::size_of::<$variant>(),)*
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
            unsafe { core::slice::from_raw_parts(self as *const _ as _, self.size()) }
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

impl TypeParser for Codegen {
    fn tys(&mut self) -> &mut Types {
        &mut self.tys
    }

    fn infer_type(&mut self, expr: &Expr) -> ty::Id {
        let mut ci = ItemCtx {
            file: self.ci.file,
            ret: self.ci.ret,
            task_base: self.ci.task_base,
            ..self.pool.cis.pop().unwrap_or_default()
        };
        ci.loops.extend(self.ci.loops.iter());
        ci.vars.extend(self.ci.vars.iter().map(|v| Variable {
            id: v.id,
            value: Value { ty: v.value.ty, loc: v.value.loc.as_ref() },
        }));
        ci.stack_relocs.extend(self.ci.stack_relocs.iter());
        ci.ret_relocs.extend(self.ci.ret_relocs.iter());
        ci.loop_relocs.extend(self.ci.loop_relocs.iter());
        ci.regs.init();

        core::mem::swap(&mut self.ci, &mut ci);
        let value = self.expr(expr).unwrap();
        self.ci.free_loc(value.loc);
        core::mem::swap(&mut self.ci, &mut ci);

        ci.loops.clear();
        ci.vars.clear();
        ci.stack_relocs.clear();
        ci.ret_relocs.clear();
        ci.loop_relocs.clear();
        ci.code.clear();
        ci.relocs.clear();
        self.pool.cis.push(ci);

        value.ty
    }

    fn eval_const(&mut self, file: FileId, expr: &Expr, ty: ty::Id) -> u64 {
        self.eval_const_low(file, expr, Some(ty)).0
    }

    fn on_reuse(&mut self, existing: ty::Id) {
        if let ty::Kind::Func(id) = existing.expand()
            && let func = &mut self.tys.ins.funcs[id as usize]
            && let Err(idx) = task::unpack(func.offset)
            && idx < self.tasks.len()
        {
            func.offset = task::id(self.tasks.len());
            let task = self.tasks[idx].take();
            self.tasks.push(task);
        }
    }

    fn eval_global(&mut self, file: FileId, name: Ident, expr: &Expr) -> ty::Id {
        let gid = self.tys.ins.globals.len() as ty::Global;
        self.tys.ins.globals.push(Global { file, name, ..Default::default() });

        let ci = ItemCtx { file, ..self.pool.cis.pop().unwrap_or_default() };

        self.tys.ins.globals[gid as usize] =
            self.ct_eval(ci, |s, _| Ok::<_, !>(s.generate_global(expr, file, name))).into_ok();

        ty::Kind::Global(gid).compress()
    }

    fn report(&self, pos: Pos, msg: impl Display) -> ty::Id {
        self.report(pos, msg)
    }

    fn find_local_ty(&mut self, name: Ident) -> Option<ty::Id> {
        self.ci.vars.iter().rfind(|v| v.id == name && v.value.ty == ty::Id::TYPE).map(|v| {
            match v.value.loc {
                Loc::Rt { .. } => unreachable!(),
                Loc::Ct { derefed, value } => ty::Id::from(ensure_loaded(value, derefed, 4)),
            }
        })
    }
}

impl Codegen {
    pub fn push_embeds(&mut self, embeds: Vec<Vec<u8>>) {
        self.tys.ins.globals = embeds
            .into_iter()
            .map(|data| Global {
                ty: self.tys.make_array(ty::Id::U8, data.len() as _),
                data,
                ..Default::default()
            })
            .collect();
    }

    pub fn generate(&mut self, root: FileId) {
        self.ci.emit_entry_prelude();
        self.ci.file = root;
        self.find_type(0, root, Err("main"), &self.files.clone());
        self.make_func_reachable(0);
        self.complete_call_graph();
    }

    fn expr(&mut self, expr: &Expr) -> Option<Value> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn build_struct(
        &mut self,
        file: FileId,
        pos: Option<Pos>,
        explicit_alignment: Option<u8>,
        fields: &[CommentOr<StructField>],
    ) -> ty::Struct {
        let sym = pos.map(|pos| SymKey::Struct(file, pos, Default::default()));
        if let Some(sym) = sym
            && let Some(&ty) = self.tys.syms.get(sym, &self.tys.ins)
        {
            return ty.expand().inner();
        }

        let prev_tmp = self.tys.tmp.fields.len();
        for sf in fields.iter().filter_map(CommentOr::or) {
            let f = Field { name: self.tys.names.intern(sf.name), ty: self.ty(&sf.ty) };
            self.tys.tmp.fields.push(f);
        }
        self.tys.ins.structs.push(Struct {
            field_start: self.tys.ins.fields.len() as _,
            pos: pos.unwrap_or(Pos::MAX),
            explicit_alignment,
            file,
            ..Default::default()
        });
        self.tys.ins.fields.extend(self.tys.tmp.fields.drain(prev_tmp..));

        if let Some(sym) = sym {
            self.tys.syms.insert(
                sym,
                ty::Kind::Struct(self.tys.ins.structs.len() as u32 - 1).compress(),
                &self.tys.ins,
            );
        }

        self.tys.ins.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Value> {
        use {Expr as E, TokenKind as T};
        let value = match *expr {
            E::Mod { id, .. } => Some(Value::ty(ty::Kind::Module(id).compress())),
            E::Embed { id, .. } => self.handle_global(id),
            E::Struct { captured, packed, fields, pos, .. } => {
                if captured.is_empty() {
                    Some(Value::ty(
                        ty::Kind::Struct(self.build_struct(
                            self.ci.file,
                            Some(pos),
                            packed.then_some(1),
                            fields,
                        ))
                        .compress(),
                    ))
                } else {
                    let values = captured
                        .iter()
                        .map(|&id| E::Ident { pos: 0, is_ct: false, id, is_first: false })
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

                    self.stack_offset(2, reg::STACK_PTR, Some(&stack), 0);
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
                let len = size.map_or(ArrayLen::MAX, |expr| {
                    self.eval_const(self.ci.file, expr, ty::Id::U32) as _
                });
                Some(Value::ty(self.tys.make_array(ty, len)))
            }
            E::Index { base, index } => {
                // TODO: we need to check if index is in bounds on debug builds

                let mut base_val = self.expr(base)?;
                if base_val.ty.is_pointer() {
                    base_val.loc = self.make_loc_owned(base_val.loc, base_val.ty);
                }
                let index_val = self.expr(index)?;
                _ = self.assert_ty(
                    index.pos(),
                    index_val.ty,
                    ty::Id::INT,
                    TyCheck::BinOp,
                    "subsctipt",
                );

                if let Some(ty) = self.tys.base_of(base_val.ty) {
                    base_val.ty = ty;
                    base_val.loc = base_val.loc.into_derefed();
                }

                match base_val.ty.expand() {
                    ty::Kind::Slice(arr) => {
                        let ty = self.tys.ins.slices[arr as usize].elem;
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
                let ty = self.ty(func_ast);
                let ty::Kind::Func(mut func) = ty.expand() else {
                    self.report(
                        func_ast.pos(),
                        format_args!(
                            "first argument of inline needs to be a function, but its '{}'",
                            self.ty_display(ty)
                        ),
                    );
                };

                let fuc = &self.tys.ins.funcs[func as usize];
                let ast = self.files[fuc.file as usize].clone();
                let &E::Closure { args: cargs, body, .. } = fuc.expr.get(&ast) else {
                    unreachable!();
                };

                let scope = self.ci.vars.len();
                let sig = self.compute_signature(&mut func, func_ast.pos(), args)?;
                self.ci.vars.truncate(scope);

                self.assert_arg_count(expr.pos(), args.len(), cargs.len(), "inline function call");

                let mut sig_args = sig.args.range();
                for (arg, carg) in args.iter().zip(cargs) {
                    let ty = self.tys.ins.args[sig_args.next().unwrap()];
                    let sym = parser::find_symbol(&ast.symbols, carg.id);
                    let loc = match sym.flags & idfl::COMPTIME != 0 {
                        true => Loc::ty(self.tys.ins.args[sig_args.next().unwrap()]),
                        false => self.expr_ctx(arg, Ctx::default().with_ty(ty))?.loc,
                    };
                    self.ci.vars.push(Variable { id: carg.id, value: Value { ty, loc } });
                }

                let ret_reloc_base = self.ci.ret_relocs.len();

                let loc = self.alloc_ret(sig.ret, ctx, true);
                let prev_ret_reg = core::mem::replace(&mut self.ci.inline_ret_loc, loc);
                let fuc = &self.tys.ins.funcs[func as usize];
                let prev_file = core::mem::replace(&mut self.ci.file, fuc.file);
                let prev_ret = core::mem::replace(&mut self.ci.ret, Some(sig.ret));
                self.expr(body);
                let loc = core::mem::replace(&mut self.ci.inline_ret_loc, prev_ret_reg);
                self.ci.file = prev_file;
                self.ci.ret = prev_ret;

                let mut vars = core::mem::take(&mut self.ci.vars);
                for var in vars.drain(scope..) {
                    self.ci.free_loc(var.value.loc);
                }
                self.ci.vars = vars;

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
            E::Directive { name: "eca", args, pos } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "type to return form eca is unknown, use `@as(<type>, @eca(...<expr>))`",
                    );
                };

                let (_, mut parama) = self.tys.parama(ty);
                let base = self.pool.arg_locs.len();
                for arg in args {
                    let arg = self.expr(arg)?;
                    if arg.ty == ty::Id::from(ty::TYPE) {
                        self.report(pos, "na na na nana, no passing types to ecas");
                    }
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
                Some(Value::imm(self.tys.size_of(ty) as _))
            }
            E::Directive { name: "alignof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(Value::imm(self.tys.align_of(ty) as _))
            }
            E::Directive { name: "intcast", args: [val], .. } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, @intcast(<expr>))`",
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
            E::Bool { value, .. } => Some(Value { ty: ty::Id::BOOL, loc: Loc::ct(value as u64) }),
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
            E::String { pos, literal } => {
                let literal = &literal[1..literal.len() - 1];

                let report = |bytes: &core::str::Bytes, message: &str| {
                    self.report(pos + (literal.len() - bytes.len()) as u32 - 1, message)
                };

                let mut str = Vec::<u8>::with_capacity(literal.len());
                crate::endoce_string(literal, &mut str, report);

                let reloc = Reloc::new(self.ci.code.len() as _, 3, 4);
                let glob = self.tys.ins.globals.len() as ty::Global;
                self.tys.ins.globals.push(Global { data: str, ..Default::default() });
                self.ci
                    .relocs
                    .push(TypedReloc { target: ty::Kind::Global(glob).compress(), reloc });
                let reg = self.ci.regs.allocate();
                self.ci.emit(instrs::lra(reg.get(), 0, 0));
                Some(Value::new(self.tys.make_ptr(ty::U8.into()), reg))
            }
            E::Ctor { pos, ty, fields, .. } => {
                let (ty, loc) = self.prepare_struct_ctor(pos, &mut ctx, ty, fields.len());

                let ty::Kind::Struct(stru) = ty.expand() else {
                    self.report(
                        pos,
                        "our current technology does not (yet) allow\
                        us to construct '{}' with struct constructor",
                    );
                };

                for &CtorField { pos, name, ref value, .. } in fields {
                    let Some((offset, ty)) = OffsetIter::offset_of(&self.tys, stru, name) else {
                        self.report(pos, format_args!("field not found: {name:?}"));
                    };
                    let loc = loc.as_ref().offset(offset);
                    let value = self.expr_ctx(value, Ctx::default().with_loc(loc).with_ty(ty))?;
                    self.ci.free_loc(value.loc);
                }

                if let Some(dst_loc) = ctx.loc {
                    self.store_typed(loc, &dst_loc, ty);
                    return Some(Value { ty, loc: dst_loc });
                } else {
                    return Some(Value { ty, loc });
                }
            }
            E::Tupl { pos, ty, fields, .. } => {
                let (ty, loc) = self.prepare_struct_ctor(pos, &mut ctx, ty, fields.len());

                match ty.expand() {
                    ty::Kind::Struct(stru) => {
                        let mut oiter = OffsetIter::new(stru, &self.tys);
                        for field in fields {
                            let (ty, offset) = oiter.next_ty(&self.tys).unwrap();
                            let loc = loc.as_ref().offset(offset);
                            let ctx = Ctx::default().with_loc(loc).with_ty(ty);
                            let value = self.expr_ctx(field, ctx)?;
                            self.ci.free_loc(value.loc);
                        }
                    }
                    ty::Kind::Slice(arr) => {
                        let arr = self.tys.ins.slices[arr as usize];
                        let item_size = self.tys.size_of(arr.elem);
                        for (i, value) in fields.iter().enumerate() {
                            let loc = loc.as_ref().offset(i as u32 * item_size);
                            let value = self
                                .expr_ctx(value, Ctx::default().with_loc(loc).with_ty(arr.elem))?;
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

                if let Some(dst_loc) = ctx.loc {
                    self.store_typed(loc, &dst_loc, ty);
                    return Some(Value { ty, loc: dst_loc });
                } else {
                    return Some(Value { ty, loc });
                }
            }
            E::Field { target, name: field, pos } => {
                let checkpoint = self.ci.snap();
                let mut tal = self.expr(target)?;

                if let Some(ty) = self.tys.base_of(tal.ty) {
                    tal.ty = ty;
                    tal.loc = tal.loc.into_derefed();
                }

                match tal.ty.expand() {
                    ty::Kind::Struct(idx) => {
                        let Some((offset, ty)) = OffsetIter::offset_of(&self.tys, idx, field)
                        else {
                            self.report(pos, format_args!("field not found: {field:?}"));
                        };
                        Some(Value { ty, loc: tal.loc.offset(offset) })
                    }
                    ty::Kind::Builtin(ty::TYPE) => {
                        self.ci.free_loc(tal.loc);
                        self.ci.revert(checkpoint);
                        match self.ty(target).expand() {
                            ty::Kind::Module(idx) => {
                                match self
                                    .find_type(pos, idx, Err(field), &self.files.clone())
                                    .expand()
                                {
                                    ty::Kind::Global(idx) => self.handle_global(idx),
                                    e => Some(Value::ty(e.compress())),
                                }
                            }
                            ty::Kind::Global(idx) => self.handle_global(idx),
                            e => unimplemented!("{e:?}"),
                        }
                    }
                    _ => self.report(
                        target.pos(),
                        format_args!(
                            "the field operation is not supported: {}",
                            self.ty_display(tal.ty)
                        ),
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
                self.ci.regs.free(oper);

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
                let offset = core::mem::take(offset) as _;
                if reg.is_ref() {
                    let new_reg = self.ci.regs.allocate();
                    self.stack_offset(new_reg.get(), reg.get(), stack.as_ref(), offset);
                    *reg = new_reg;
                } else {
                    self.stack_offset(reg.get(), reg.get(), stack.as_ref(), offset);
                }

                // FIXME: we might be able to track this but it will be pain
                core::mem::forget(stack.take());

                Some(Value { ty: self.tys.make_ptr(val.ty), loc: val.loc })
            }
            E::UnOp { op: T::Mul, val, pos } => {
                let val = self.expr(val)?;
                match self.tys.base_of(val.ty) {
                    Some(ty) => Some(Value {
                        ty,
                        loc: Loc::reg(self.loc_to_reg(val.loc, self.tys.size_of(val.ty)))
                            .into_derefed(),
                    }),
                    _ => self.report(
                        pos,
                        format_args!("expected pointer, got {}", self.ty_display(val.ty)),
                    ),
                }
            }
            E::BinOp { left, op: T::Decl, right, .. } if self.has_ct(left) => {
                let slot_base = self.ct.vm.read_reg(reg::STACK_PTR).0;
                let (cnt, ty) = self.eval_const_low(self.ci.file, right, None);
                if self.assign_ct_pattern(left, ty, cnt as _) {
                    self.ct.vm.write_reg(reg::STACK_PTR, slot_base);
                }
                Some(Value::void())
            }
            E::BinOp { left, op: T::Decl, right, .. } => {
                let value = self.expr(right)?;
                self.assign_pattern(left, value)
            }
            E::Call { func: fast, args, .. } => {
                log::trace!("call {}", self.ast_display(fast));
                let func_ty = self.ty(fast);
                let ty::Kind::Func(mut func) = func_ty.expand() else {
                    self.report(
                        fast.pos(),
                        format_args!(
                            "can't '{}' this, maybe in the future",
                            self.ty_display(func_ty)
                        ),
                    );
                };

                // TODO: this will be usefull but not now
                let scope = self.ci.vars.len();
                let sig = self.compute_signature(&mut func, expr.pos(), args)?;
                self.ci.vars.truncate(scope);

                let fuc = &self.tys.ins.funcs[func as usize];
                let ast = self.files[fuc.file as usize].clone();
                let &E::Closure { args: cargs, .. } = fuc.expr.get(&ast) else {
                    unreachable!();
                };

                let (_, mut parama) = self.tys.parama(sig.ret);
                let base = self.pool.arg_locs.len();
                let mut sig_args = sig.args.range();
                let mut should_momize = !args.is_empty() && sig.ret == ty::Id::from(ty::TYPE);

                self.assert_arg_count(expr.pos(), args.len(), cargs.len(), "function call");

                for (i, (arg, carg)) in args.iter().zip(cargs).enumerate() {
                    let ty = self.tys.ins.args[sig_args.next().unwrap()];
                    let sym = parser::find_symbol(&ast.symbols, carg.id);
                    if sym.flags & idfl::COMPTIME != 0 {
                        sig_args.next().unwrap();
                        continue;
                    }

                    // TODO: pass the arg as dest
                    let varg = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    _ = self.assert_ty(
                        arg.pos(),
                        varg.ty,
                        ty,
                        TyCheck::Assign,
                        format_args!("argument({i})"),
                    );
                    self.pass_arg(&varg, &mut parama);
                    self.pool.arg_locs.push(varg.loc);
                    should_momize = false;
                }

                for value in self.pool.arg_locs.drain(base..) {
                    self.ci.free_loc(value);
                }

                let loc = self.alloc_ret(sig.ret, ctx, true);

                if should_momize {
                    self.ci.write_trap(trap::Trap::MomizedCall(trap::MomizedCall { func }));
                }

                let reloc = Reloc::new(self.ci.code.len(), 3, 4);
                self.ci.relocs.push(TypedReloc { target: ty::Kind::Func(func).compress(), reloc });
                self.ci.emit(jal(reg::RET_ADDR, reg::ZERO, 0));
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
                    self.ci.vars.iter_mut().enumerate().rfind(|(_, v)| v.id == id) =>
            {
                let loc = var.value.loc.as_ref();
                Some(Value { ty: self.ci.vars[var_index].value.ty, loc })
            }
            E::Ident { id, pos, .. } => {
                match self.find_type(pos, self.ci.file, Ok(id), &self.files.clone()).expand() {
                    ty::Kind::Global(id) => self.handle_global(id),
                    tk => Some(Value::ty(tk.compress())),
                }
            }
            E::Return { pos, val, .. } => {
                let size = self.ci.ret.map_or(17, |ty| self.tys.size_of(ty));
                let loc = match size {
                    _ if self.ci.inline_ret_loc != Loc::default() => {
                        Some(self.ci.inline_ret_loc.as_ref())
                    }
                    0 => None,
                    1..=16 => Some(Loc::reg(1)),
                    _ => Some(Loc::reg(self.ci.ret_reg.as_ref()).into_derefed()),
                };
                let value = if let Some(val) = val {
                    self.expr_ctx(val, Ctx { ty: self.ci.ret, loc, ..Default::default() })?
                } else {
                    Value::void()
                };

                match self.ci.ret {
                    None => self.ci.ret = Some(value.ty),
                    Some(ret) => {
                        _ = self.assert_ty(pos, value.ty, ret, TyCheck::Assign, "return type")
                    }
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
                    let ty = ctx.ty.map(ty::Id::strip_pointer).unwrap_or(ty::Id::INT);
                    if !ty.is_integer() && !ty.is_pointer() {
                        self.report(
                            pos,
                            format_args!(
                                "this integer was inferred to be '{}'",
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
                if let &E::BinOp { left, op, right, .. } = cond
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
                        core::mem::swap(&mut then, &mut else_);
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

                let mut vars = core::mem::take(&mut self.ci.vars);
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
            E::BinOp { left, op: op @ (T::And | T::Or), right, .. } => {
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

                Some(Value { ty: ty::Id::BOOL, loc: Loc::reg(lhs) })
            }
            E::BinOp { left, op, right, .. } if op != T::Decl => 'ops: {
                let left = self.expr_ctx(left, Ctx {
                    ty: ctx.ty.filter(|_| op.is_homogenous()),
                    check: ctx.check,
                    ..Default::default()
                })?;

                if op == T::Assign {
                    let value = self.expr_ctx(right, Ctx::from(left)).unwrap();
                    self.ci.free_loc(value.loc);
                    return Some(Value::void());
                }

                if let ty::Kind::Struct(_) = left.ty.expand() {
                    let right = self.expr_ctx(right, Ctx::default().with_ty(left.ty))?;
                    _ = self.assert_ty(
                        expr.pos(),
                        right.ty,
                        left.ty,
                        TyCheck::Assign,
                        "right struct operand",
                    );
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
                let right = self
                    .expr_ctx(right, Ctx::default().with_ty(left.ty).with_check(TyCheck::BinOp))?;
                let rsize = self.tys.size_of(right.ty);

                let ty = self.assert_ty(
                    expr.pos(),
                    right.ty,
                    left.ty,
                    TyCheck::BinOp,
                    "right sclalar operand",
                );
                let size = self.tys.size_of(ty);
                let signed = ty.is_signed();

                if let Loc::Ct { value: CtValue(mut imm), derefed } = right.loc
                    && let Some(oper) = op.imm_binop(signed, size)
                {
                    if derefed {
                        let mut dst = [0u8; 8];
                        dst[..size as usize].copy_from_slice(unsafe {
                            core::slice::from_raw_parts(imm as _, rsize as usize)
                        });
                        imm = u64::from_ne_bytes(dst);
                    }
                    if matches!(op, T::Add | T::Sub)
                        && let Some(ty) = self.tys.base_of(ty)
                    {
                        imm *= self.tys.size_of(ty) as u64;
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

                        let ty = self.tys.base_of(ty).unwrap();
                        let size = self.tys.size_of(ty);
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
            ref ast => self.report_unhandled_ast(ast, "something"),
        }?;

        if let Some(ty) = ctx.ty {
            _ = self.assert_ty(expr.pos(), value.ty, ty, ctx.check, "something");
        }

        Some(match ctx.loc {
            Some(dest) => {
                self.store_sized(
                    value.loc,
                    dest,
                    self.tys.size_of(ctx.ty.unwrap_or(value.ty)).min(self.tys.size_of(value.ty)),
                );
                Value { ty: value.ty, loc: Loc::ct(0) }
            }
            None => value,
        })
    }

    fn compute_signature(&mut self, func: &mut ty::Func, pos: Pos, args: &[Expr]) -> Option<Sig> {
        let fuc = &self.tys.ins.funcs[*func as usize];
        let fast = self.files[fuc.file as usize].clone();
        let &Expr::Closure { args: cargs, ret, .. } = fuc.expr.get(&fast) else {
            unreachable!();
        };

        Some(if let Some(sig) = fuc.sig {
            sig
        } else {
            let arg_base = self.tys.tmp.args.len();

            for (arg, carg) in args.iter().zip(cargs) {
                let ty = self.ty(&carg.ty);
                self.tys.tmp.args.push(ty);
                let sym = parser::find_symbol(&fast.symbols, carg.id);
                let loc = if sym.flags & idfl::COMPTIME == 0 {
                    // FIXME: could fuck us
                    Loc::default()
                } else {
                    debug_assert_eq!(
                        ty,
                        ty::Id::TYPE,
                        "TODO: we dont support anything except type generics"
                    );
                    let arg = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    self.tys.tmp.args.push(arg.loc.to_ty().unwrap());
                    arg.loc
                };

                self.ci.vars.push(Variable { id: carg.id, value: Value { ty, loc } });
            }

            let args = self
                .tys
                .pack_args(arg_base)
                .unwrap_or_else(|| self.report(pos, "function instance has too many arguments"));
            let ret = self.ty(ret);

            let sym = SymKey::FuncInst(*func, args);
            let ct = |ins: &mut crate::TypeIns| {
                let func_id = ins.funcs.len();
                let fuc = &ins.funcs[*func as usize];
                ins.funcs.push(Func {
                    file: fuc.file,
                    name: fuc.name,
                    base: Some(*func),
                    sig: Some(Sig { args, ret }),
                    expr: fuc.expr,
                    ..Default::default()
                });

                ty::Kind::Func(func_id as _).compress()
            };
            *func = self.tys.syms.get_or_insert(sym, &mut self.tys.ins, ct).expand().inner();

            Sig { args, ret }
        })
    }

    fn has_ct(&self, expr: &Expr) -> bool {
        expr.has_ct(&self.cfile().symbols)
    }

    fn eval_const_low(
        &mut self,
        file: FileId,
        expr: &Expr,
        mut ty: Option<ty::Id>,
    ) -> (u64, ty::Id) {
        let mut ci = ItemCtx { file, ret: ty, ..self.pool.cis.pop().unwrap_or_default() };
        ci.vars.append(&mut self.ci.vars);

        let loc = self.ct_eval(ci, |s, prev| {
            s.ci.emit_prelude();

            if s.ci.ret.map_or(true, |r| s.tys.size_of(r) > 16) {
                let reg = s.ci.regs.allocate();
                s.ci.emit(instrs::cp(reg.get(), 1));
                s.ci.ret_reg = reg;
            };

            let ctx = Ctx { ty: s.ci.ret, ..Default::default() };
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
                self.ci.vars.push(Variable { id, value: Value { ty, loc } });
                true
            }
            Expr::Ident { id, .. } => {
                let var = Variable { id, value: Value { ty, loc: Loc::ct_ptr(offset as _) } };
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
                self.ci.vars.push(Variable { id, value: Value { ty: right.ty, loc } });
            }
            Expr::Ctor { pos, fields, .. } => {
                let ty::Kind::Struct(idx) = right.ty.expand() else {
                    self.report(pos, "can't use struct destruct on non struct value (TODO: shold work with modules)");
                };

                for &CtorField { pos, name, ref value } in fields {
                    let Some((offset, ty)) = OffsetIter::offset_of(&self.tys, idx, name) else {
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
        ctx: &mut Ctx,
        ty: Option<&Expr>,
        field_len: usize,
    ) -> (ty::Id, Loc) {
        let Some(mut ty) = ty.map(|ty| self.ty(ty)).or(ctx.ty) else {
            self.report(pos, "expected type, (it cannot be inferred)");
        };

        if let Some(expected) = ctx.ty {
            _ = self.assert_ty(pos, ty, expected, TyCheck::Assign, "struct");
        }

        match ty.expand() {
            ty::Kind::Struct(stru) => {
                let field_count = self.tys.struct_field_range(stru).len();
                if field_count != field_len {
                    self.report(
                        pos,
                        format_args!("expected {field_count} fields, got {field_len}"),
                    );
                }
            }
            ty::Kind::Slice(arr) => {
                let arr = &self.tys.ins.slices[arr as usize];
                if arr.len == ArrayLen::MAX {
                    ty = self.tys.make_array(arr.elem, field_len as _);
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
            _ => self.report(
                pos,
                format_args!(
                    "expected expression to evaluate to struct (or array maybe) but it evaluated to {}",
                    self.ty_display(ty)
                ),
            ),
        }

        let size = self.tys.size_of(ty);
        if ctx.loc.as_ref().map_or(true, |l| l.is_reg()) {
            (ty, Loc::stack(self.ci.stack.allocate(size)))
        } else {
            (ty, ctx.loc.take().unwrap_or_else(|| Loc::stack(self.ci.stack.allocate(size))))
        }
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

            let mut oiter = OffsetIter::new(stuct, &self.tys);
            while let Some((ty, offset)) = oiter.next_ty(&self.tys) {
                let ctx = Ctx::from(Value { ty, loc: loc.as_ref().offset(offset) });
                let left = left.as_ref().offset(offset);
                let right = right.as_ref().offset(offset);
                let value = self.struct_op(op, ty, ctx, left, right)?;
                self.ci.free_loc(value.loc);
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
        let global = &mut self.tys.ins.globals[id as usize];
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
            1..=8 if !loc.is_stack() => Loc::reg(self.loc_to_reg(loc, size)),
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
        let func = &self.tys.ins.funcs[id as usize];
        debug_assert!(func.file == file);
        let sig = func.sig.unwrap();
        let ast = self.files[file as usize].clone();
        let expr = func.expr.get(&ast);
        let ct_stack_base = self.ct.vm.read_reg(reg::STACK_PTR).0;

        let repl = ItemCtx { file, ret: Some(sig.ret), ..self.pool.cis.pop().unwrap_or_default() };
        let prev_ci = core::mem::replace(&mut self.ci, repl);
        self.ci.regs.init();

        let Expr::Closure { body, args, .. } = expr else {
            unreachable!("{}", self.ast_display(expr))
        };

        self.ci.emit_prelude();

        let (ret, mut parama) = self.tys.parama(sig.ret);
        let mut sig_args = sig.args.range();
        for arg in args.iter() {
            let ty = self.tys.ins.args[sig_args.next().unwrap()];
            let sym = parser::find_symbol(&ast.symbols, arg.id).flags;
            let loc = match sym & idfl::COMPTIME != 0 {
                true => Loc::ty(self.tys.ins.args[sig_args.next().unwrap()]),
                false => self.load_arg(sym, ty, &mut parama),
            };
            self.ci.vars.push(Variable { id: arg.id, value: Value { ty, loc } });
        }

        if let Some(PLoc::Ref(..)) = ret {
            let reg = self.ci.regs.allocate();
            self.ci.emit(instrs::cp(reg.get(), 1));
            self.ci.ret_reg = reg;
        } else {
            self.ci.ret_reg = rall::Id::RET;
        }

        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        let mut vars = core::mem::take(&mut self.ci.vars);
        for var in vars.drain(..) {
            self.ci.free_loc(var.value.loc);
        }
        self.ci.vars = vars;

        self.ci.finalize();
        self.ci.emit(jala(reg::ZERO, reg::RET_ADDR, 0));
        self.ci.regs.free(core::mem::take(&mut self.ci.ret_reg));
        self.tys.ins.funcs[id as usize].code.append(&mut self.ci.code);
        self.tys.ins.funcs[id as usize].relocs = self.ci.relocs.drain(..).collect();
        self.pool.cis.push(core::mem::replace(&mut self.ci, prev_ci));
        self.ct.vm.write_reg(reg::STACK_PTR, ct_stack_base);
    }

    fn load_arg(&mut self, flags: parser::IdentFlags, ty: ty::Id, parama: &mut ParamAlloc) -> Loc {
        let size = self.tys.size_of(ty) as Size;
        if size == 0 {
            return Loc::default();
        }
        let (src, dst) = match parama.next(ty, &self.tys) {
            None => (Loc::default(), Loc::default()),
            Some(PLoc::Reg(r, _)) if flags & idfl::REFERENCED == 0 => {
                (Loc::reg(r), Loc::reg(self.ci.regs.allocate()))
            }
            Some(PLoc::Reg(r, _)) => (Loc::reg(r), Loc::stack(self.ci.stack.allocate(size))),
            Some(PLoc::WideReg(r, _)) => (Loc::reg(r), Loc::stack(self.ci.stack.allocate(size))),
            Some(PLoc::Ref(ptr, _)) if flags & (idfl::MUTABLE | idfl::REFERENCED) == 0 => {
                let reg = self.ci.regs.allocate();
                self.ci.emit(instrs::cp(reg.get(), ptr));
                return Loc::reg(reg).into_derefed();
            }
            Some(PLoc::Ref(ptr, _)) => {
                (Loc::reg(ptr).into_derefed(), Loc::stack(self.ci.stack.allocate(size)))
            }
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

    fn loc_to_reg(&mut self, loc: impl Into<LocCow>, size: Size) -> rall::Id {
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
        match parama.next(value.ty, &self.tys) {
            None => {}
            Some(PLoc::Reg(r, _) | PLoc::WideReg(r, _)) => {
                self.store_typed(&value.loc, Loc::reg(r), value.ty)
            }
            Some(PLoc::Ref(ptr, _)) => {
                let Loc::Rt { reg, stack, offset, .. } = &value.loc else { unreachable!() };
                self.stack_offset(ptr, reg.get(), stack.as_ref(), *offset as _);
            }
        }
    }

    fn store_typed(&mut self, src: impl Into<LocCow>, dst: impl Into<LocCow>, ty: ty::Id) {
        self.store_sized(src, dst, self.tys.size_of(ty) as _)
    }

    fn store_sized(&mut self, src: impl Into<LocCow>, dst: impl Into<LocCow>, size: Size) {
        self.store_sized_low(src.into(), dst.into(), size);
    }

    fn store_sized_low(&mut self, src: LocCow, dst: LocCow, mut size: Size) {
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
            (lpat!(true, src, soff, ref ssta), lpat!(true, dst, doff, ref dsta)) => 'a: {
                if size <= 8 {
                    let tmp = self.ci.regs.allocate();
                    let off = self.opt_stack_reloc(ssta.as_ref(), soff, 3);
                    self.ci.emit(ld(tmp.get(), src.get(), off, size as _));
                    let off = self.opt_stack_reloc(dsta.as_ref(), doff, 3);
                    self.ci.emit(st(tmp.get(), dst.get(), off, size as _));
                    self.ci.regs.free(tmp);
                    break 'a;
                }

                // TODO: some oportuinies to ellit more optimal code
                let src_off = if src.is_ref() { self.ci.regs.allocate() } else { src.as_ref() };
                let dst_off = if dst.is_ref() { self.ci.regs.allocate() } else { dst.as_ref() };
                self.stack_offset(src_off.get(), src.get(), ssta.as_ref(), soff);
                self.stack_offset(dst_off.get(), dst.get(), dsta.as_ref(), doff);
                loop {
                    match u16::try_from(size) {
                        Ok(o) => {
                            self.ci.emit(bmc(src_off.get(), dst_off.get(), o));
                            break;
                        }
                        Err(_) => {
                            self.ci.emit(bmc(src_off.get(), dst_off.get(), u16::MAX));
                            self.ci.emit(addi64(src_off.get(), src_off.get(), u16::MAX as _));
                            self.ci.emit(addi64(dst_off.get(), dst_off.get(), u16::MAX as _));
                            size -= u16::MAX as u32;
                        }
                    }
                }
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

    fn ty(&mut self, expr: &Expr) -> ty::Id {
        self.parse_ty(self.ci.file, expr, None, &self.files.clone())
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

        let mut code_index = self.ct.vm.pc.get() as usize - self.ct.code.as_ptr() as usize;
        debug_assert!(code_index < self.ct.code.len());

        match *trap {
            trap::Trap::MakeStruct(trap::MakeStruct { file, struct_expr }) => {
                let cfile = self.files[file as usize].clone();
                let &Expr::Struct { fields, captured, packed, .. } = struct_expr.get(&cfile) else {
                    unreachable!()
                };

                let prev_len = self.ci.vars.len();

                let mut values = self.ct.vm.read_reg(2).0 as *const u8;
                for &id in captured {
                    let ty: ty::Id = unsafe { core::ptr::read_unaligned(values.cast()) };
                    unsafe { values = values.add(4) };
                    let size = self.tys.size_of(ty) as usize;
                    let mut imm = [0u8; 8];
                    assert!(size <= imm.len(), "TODO");
                    unsafe { core::ptr::copy_nonoverlapping(values, imm.as_mut_ptr(), size) };
                    self.ci.vars.push(Variable {
                        id,
                        value: Value::new(ty, Loc::ct(u64::from_ne_bytes(imm))),
                    });
                }

                let stru = ty::Kind::Struct(self.build_struct(
                    self.ci.file,
                    packed.then_some(1),
                    None,
                    fields,
                ))
                .compress();
                self.ci.vars.truncate(prev_len);
                self.ct.vm.write_reg(1, stru.repr() as u64);
            }
            trap::Trap::MomizedCall(trap::MomizedCall { func }) => {
                if let Some(ty) = self.tys.ins.funcs[func as usize].computed {
                    self.ct.vm.write_reg(1, ty.repr());
                } else {
                    self.run_vm();
                    self.tys.ins.funcs[func as usize].computed =
                        Some(self.ct.vm.read_reg(1).0.into());
                }
                code_index += jal(0, 0, 0).0 + tx().0;
            }
        }

        let offset = code_index + self.ct.code.as_ptr() as usize;
        self.ct.vm.pc = hbvm::mem::Address::new(offset as _);
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let fuc = &mut self.tys.ins.funcs[func as usize];
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

        Global { ty: ret.ty, file, data, name, ..Default::default() }
    }

    fn ct_eval<T, E>(
        &mut self,
        ci: ItemCtx,
        compile: impl FnOnce(&mut Self, &mut ItemCtx) -> Result<T, E>,
    ) -> Result<T, E> {
        log::trace!("eval");

        let mut prev_ci = core::mem::replace(&mut self.ci, ci);
        self.ci.task_base = self.tasks.len();
        self.ci.regs.init();

        let ret = compile(self, &mut prev_ci);
        let mut rr = core::mem::take(&mut self.ci.ret_reg);
        let is_on_stack = !rr.is_ref();
        if !rr.is_ref() {
            self.ci.emit(instrs::cp(1, rr.get()));
            let rref = rr.as_ref();
            self.ci.regs.free(core::mem::replace(&mut rr, rref));
        }

        if ret.is_ok() {
            let last_fn = self.tys.ins.funcs.len();
            self.tys.ins.funcs.push(Default::default());

            self.tys.ins.funcs[last_fn].code = core::mem::take(&mut self.ci.code);
            self.tys.ins.funcs[last_fn].relocs = core::mem::take(&mut self.ci.relocs);

            if is_on_stack {
                let size =
                    self.tys.size_of(self.ci.ret.expect("you have died (colaterall fuck up)"));
                let slot = self.ct.vm.read_reg(reg::STACK_PTR).0;
                self.ct.vm.write_reg(reg::STACK_PTR, slot.wrapping_add(size as _));
                self.ct.vm.write_reg(1, slot);
            }

            self.tys.dump_reachable(last_fn as _, &mut self.ct.code);
            let prev_pc = self.ct.push_pc(self.tys.ins.funcs[last_fn].offset);

            #[cfg(debug_assertions)]
            {
                let mut vc = String::new();
                if let Err(e) = self.tys.disasm(&self.ct.code, &self.files, &mut vc, |bts| {
                    if let Some(trap) = Self::read_trap(bts.as_ptr() as _) {
                        bts.take(..trap.size() + 1).unwrap();
                    }
                }) {
                    panic!("{e} {}", vc);
                } else {
                    log::trace!("{}", vc);
                }
            }

            self.run_vm();
            self.ct.pop_pc(prev_pc);

            let func = self.tys.ins.funcs.pop().unwrap();
            self.ci.code = func.code;
            self.ci.code.clear();
            self.ci.relocs = func.relocs;
            self.ci.relocs.clear();
        }

        self.pool.cis.push(core::mem::replace(&mut self.ci, prev_ci));

        log::trace!("eval-end");

        ret
    }

    pub fn disasm(&mut self, output: &mut String) -> Result<(), DisasmError> {
        let mut bin = Vec::new();
        self.assemble(&mut bin);
        self.tys.disasm(&bin, &self.files, output, |_| {})
    }

    fn run_vm(&mut self) {
        loop {
            match self.ct.vm.run().unwrap_or_else(|e| panic!("{e:?}")) {
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

    fn ast_display<'a>(&'a self, ast: &'a Expr<'a>) -> parser::Display<'a> {
        parser::Display::new(&self.cfile().file, ast)
    }

    #[must_use]
    #[track_caller]
    fn assert_ty(
        &self,
        pos: Pos,
        ty: ty::Id,
        expected: ty::Id,
        kind: TyCheck,
        hint: impl Display,
    ) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected, kind) {
            res
        } else {
            let dty = self.ty_display(ty);
            let dexpected = self.ty_display(expected);
            log::info!("mode: {:?}", kind);
            self.report(pos, format_args!("expected {hint} of type {dexpected}, got {dty}",));
        }
    }

    fn assert_arg_count(&self, pos: Pos, got: usize, expected: usize, hint: impl Display) {
        if got != expected {
            let s = if expected != 1 { "s" } else { "" };
            self.report(pos, format_args!("{hint} expected {expected} argument{s}, got {got}"))
        }
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl core::fmt::Display) -> ! {
        log::error!("{}", self.cfile().report(pos, msg));
        unreachable!();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: &str) -> ! {
        log::debug!("{ast:#?}");
        self.report(ast.pos(), format_args!("compiler does not (yet) know how to handle ({hint})",))
    }

    fn cfile(&self) -> &parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn cow_reg(&mut self, rhs: rall::Id) -> rall::Id {
        if rhs.is_ref() {
            let reg = self.ci.regs.allocate();
            self.ci.emit(cp(reg.get(), rhs.get()));
            reg
        } else {
            rhs
        }
    }

    pub fn assemble(&mut self, buf: &mut Vec<u8>) {
        self.tys.reassemble(buf);
    }

    pub fn assemble_comptime(mut self) -> Comptime {
        self.ct.code.clear();
        self.tys.reassemble(&mut self.ct.code);
        self.ct.reset();
        self.ct
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::String, vec::Vec};

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        _ = log::set_logger(&crate::fs::Logger);
        log::set_max_level(log::LevelFilter::Debug);

        let (files, embeds) = crate::test_parse_files(ident, input);
        let mut codegen = super::Codegen { files, ..Default::default() };
        codegen.push_embeds(embeds);

        codegen.generate(0);
        let mut out = Vec::new();
        codegen.assemble(&mut out);

        let err = codegen.tys.disasm(&out, &codegen.files, output, |_| {});
        if err.is_err() {
            return;
        }

        crate::test_run_vm(&out, output);
    }

    crate::run_tests! { generate:
        arithmetic;
        variables;
        functions;
        comments;
        if_statements;
        loops;
        //fb_driver;
        pointers;
        structs;
        different_types;
        struct_operators;
        directives;
        global_variables;
        generic_types;
        generic_functions;
        inlined_generic_functions;
        c_strings;
        idk;
        struct_patterns;
        arrays;
        struct_return_from_module_function;
        //comptime_pointers;
        sort_something_viredly;
        hex_octal_binary_literals;
        //comptime_min_reg_leak;
        // structs_in_registers;
        comptime_function_from_another_file;
        inline;
        inline_test;
        some_generic_code;
        integer_inference_issues;
        writing_into_string;
        request_page;
        tests_ptr_to_ptr_copy;
        wide_ret;
    }
}
