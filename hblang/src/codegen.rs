use std::ops::Range;

use crate::ident::Ident;

use {
    crate::{
        instrs, lexer, log,
        parser::{self},
    },
    std::rc::Rc,
};

use {lexer::TokenKind as T, parser::Expr as E};

type LabelId = u32;
type Reg = u8;
type MaskElem = u64;
type Type = u32;

#[derive(Debug)]
struct LinReg(Reg);

#[cfg(debug_assertions)]
impl Drop for LinReg {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("reg leaked");
        }
    }
}

enum CowReg {
    Lin(LinReg),
    Reg(Reg),
}

impl CowReg {
    fn get(&self) -> Reg {
        match *self {
            Self::Lin(LinReg(reg)) => reg,
            Self::Reg(reg) => reg,
        }
    }
}

#[derive(Default)]
enum Ctx {
    #[default]
    None,
    Inferred(Type),
    Dest(Value),
}
impl Ctx {
    fn ty(&self) -> Option<Type> {
        Some(match self {
            Self::Inferred(ty) => *ty,
            Self::Dest(Value { ty, .. }) => *ty,
            _ => return None,
        })
    }
}

pub mod bt {
    use super::*;

    const fn builtin_type(id: u32) -> Type {
        Type::MAX - id
    }

    macro_rules! builtin_type {
        ($($name:ident;)*) => {$(
            pub const $name: Type = ${index(0)} << 2;
        )*};
    }

    builtin_type! {
        VOID;
        NEVER;
        INT;
        I64;
        I32;
        I16;
        I8;
        UINT;
        U64;
        U32;
        U16;
        U8;
        BOOL;
    }
}

#[derive(Debug)]
enum TypeKind {
    Builtin(Type),
    Struct(Type),
    Pointer(Type),
}

impl TypeKind {
    const fn from_ty(ty: Type) -> Self {
        let (flag, index) = (ty & 0b11, ty >> 2);
        match flag {
            0 => Self::Builtin(ty),
            1 => Self::Pointer(index),
            2 => Self::Struct(index),
            _ => unreachable!(),
        }
    }

    const fn encode(self) -> Type {
        let (index, flag) = match self {
            Self::Builtin(index) => return index,
            Self::Pointer(index) => (index, 1),
            Self::Struct(index) => (index, 2),
        };
        index << 2 | flag
    }
}

const STACK_PTR: Reg = 254;
const ZERO: Reg = 0;
const RET_ADDR: Reg = 31;
const ELEM_WIDTH: usize = std::mem::size_of::<MaskElem>() * 8;

struct Frame {
    label:       LabelId,
    prev_relocs: usize,
    offset:      u32,
}

struct Reloc {
    id: LabelId,
    offset: u32,
    instr_offset: u16,
    size: u16,
}

struct StackReloc {
    offset: u32,
    size:   u16,
}

#[derive(Default)]
pub struct Func {
    code:   Vec<u8>,
    relocs: Vec<Reloc>,
}

impl Func {
    pub fn extend(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub fn offset(&mut self, id: LabelId, instr_offset: u16, size: u16) {
        self.relocs.push(Reloc {
            id,
            offset: self.code.len() as u32,
            instr_offset,
            size,
        });
    }

    fn encode(&mut self, (len, instr): (usize, [u8; instrs::MAX_SIZE])) {
        let name = instrs::NAMES[instr[0] as usize];
        log::dbg!(
            "{:08x}: {}: {}",
            self.code.len(),
            name,
            instr
                .iter()
                .take(len)
                .skip(1)
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        );
        self.code.extend_from_slice(&instr[..len]);
    }

    fn push(&mut self, value: Reg, size: usize) {
        self.subi64(STACK_PTR, STACK_PTR, size as _);
        self.encode(instrs::st(value, STACK_PTR, 0, size as _));
    }

    fn pop(&mut self, value: Reg, size: usize) {
        self.encode(instrs::ld(value, STACK_PTR, 0, size as _));
        self.encode(instrs::addi64(STACK_PTR, STACK_PTR, size as _));
    }

    fn subi64(&mut self, dest: Reg, src: Reg, imm: u64) {
        self.encode(instrs::addi64(dest, src, imm.wrapping_neg()));
    }

    fn call(&mut self, func: LabelId) {
        self.offset(func, 3, 4);
        self.encode(instrs::jal(RET_ADDR, ZERO, 0));
    }

    fn ret(&mut self) {
        self.encode(instrs::jala(ZERO, RET_ADDR, 0));
    }

    fn prelude(&mut self, entry: LabelId) {
        self.call(entry);
        self.encode(instrs::tx());
    }

    fn relocate(&mut self, labels: &[FnLabel], shift: i64) {
        for reloc in self.relocs.drain(..) {
            let label = &labels[reloc.id as usize];
            let offset = if reloc.size == 8 {
                reloc.offset as i64
            } else {
                label.offset as i64 - reloc.offset as i64
            } + shift;

            log::dbg!(
                label.name,
                offset,
                reloc.size,
                reloc.instr_offset,
                reloc.offset,
                shift,
                label.offset
            );

            let dest = &mut self.code[reloc.offset as usize + reloc.instr_offset as usize..]
                [..reloc.size as usize];
            match reloc.size {
                2 => dest.copy_from_slice(&(offset as i16).to_le_bytes()),
                4 => dest.copy_from_slice(&(offset as i32).to_le_bytes()),
                8 => dest.copy_from_slice(&(offset as i64).to_le_bytes()),
                _ => unreachable!(),
            };
        }
    }
}

#[derive(Default)]
pub struct RegAlloc {
    free:     Vec<Reg>,
    max_used: Reg,
}

impl RegAlloc {
    fn init_callee(&mut self) {
        self.free.clear();
        self.free.extend((32..=253).rev());
        self.max_used = RET_ADDR;
    }

    fn allocate(&mut self) -> LinReg {
        let reg = self.free.pop().expect("TODO: we need to spill");
        self.max_used = self.max_used.max(reg);
        LinReg(reg)
    }

    fn free(&mut self, reg: LinReg) {
        self.free.push(reg.0);
        std::mem::forget(reg);
    }

    fn free_cow(&mut self, reg: CowReg) {
        match reg {
            CowReg::Lin(reg) => self.free(reg),
            CowReg::Reg(_) => {}
        }
    }

    fn pushed_size(&self) -> usize {
        (self.max_used as usize - RET_ADDR as usize + 1) * 8
    }
}

#[derive(Clone)]
struct FnLabel {
    offset: u32,
    name:   Ident,
    args:   Rc<[Type]>,
    ret:    Type,
}

struct Variable {
    id:    Ident,
    value: Value,
}

struct RetReloc {
    offset:       u32,
    instr_offset: u16,
    size:         u16,
}

struct Loop {
    offset: u32,
    relocs: Vec<RetReloc>,
}

struct Struct {
    name:   Rc<str>,
    id:     Ident,
    fields: Rc<[(Rc<str>, Type)]>,
}

struct TypeDisplay<'a> {
    codegen: &'a Codegen<'a>,
    ty:      Type,
}

impl<'a> TypeDisplay<'a> {
    fn new(codegen: &'a Codegen<'a>, ty: Type) -> Self {
        Self { codegen, ty }
    }

    fn rety(&self, ty: Type) -> Self {
        Self::new(self.codegen, ty)
    }
}

impl<'a> std::fmt::Display for TypeDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TypeKind as TK;
        let str = match TK::from_ty(self.ty) {
            TK::Builtin(bt::VOID) => "void",
            TK::Builtin(bt::NEVER) => "never",
            TK::Builtin(bt::INT) => "int",
            TK::Builtin(bt::I64) => "i64",
            TK::Builtin(bt::I32) => "i32",
            TK::Builtin(bt::I16) => "i16",
            TK::Builtin(bt::I8) => "i8",
            TK::Builtin(bt::UINT) => "uint",
            TK::Builtin(bt::U64) => "u64",
            TK::Builtin(bt::U32) => "u32",
            TK::Builtin(bt::U16) => "u16",
            TK::Builtin(bt::U8) => "u8",
            TK::Builtin(bt::BOOL) => "bool",
            TK::Builtin(_) => unreachable!(),
            TK::Pointer(ty) => {
                return write!(f, "*{}", self.rety(self.codegen.pointers[ty as usize]))
            }
            TK::Struct(idx) => return write!(f, "{}", self.codegen.records[idx as usize].name),
        };

        f.write_str(str)
    }
}

#[derive(Default)]
pub struct Codegen<'a> {
    path:       &'a str,
    input:      &'a [u8],
    ret:        Type,
    gpa:        RegAlloc,
    code:       Func,
    temp:       Func,
    labels:     Vec<FnLabel>,
    stack_size: u64,
    vars:       Vec<Variable>,
    ret_relocs: Vec<RetReloc>,
    loops:      Vec<Loop>,
    records:    Vec<Struct>,
    pointers:   Vec<Type>,
    main:       Option<LabelId>,
}

impl<'a> Codegen<'a> {
    pub fn file(&mut self, path: &'a str, input: &'a [u8], exprs: &'a [parser::Expr<'a>]) {
        self.path = path;
        self.input = input;

        for expr in exprs {
            match expr {
                E::BinOp {
                    left: E::Ident { id, .. },
                    op: T::Decl,
                    right: E::Closure { args, ret, .. },
                } => {
                    let args = args.iter().map(|arg| self.ty(&arg.ty)).collect::<Rc<[_]>>();
                    let ret = self.ty(ret);
                    self.declare_fn_label(*id, args, ret);
                }
                E::BinOp {
                    left: E::Ident { id, name, .. },
                    op: T::Decl,
                    right: E::Struct { .. },
                } => {
                    self.records.push(Struct {
                        id:     *id,
                        name:   (*name).into(),
                        fields: Rc::from([]),
                    });
                }
                _ => self.report(expr.pos(), "expected declaration"),
            }
        }

        for expr in exprs {
            let E::BinOp {
                left: E::Ident { id, name, .. },
                op: T::Decl,
                right,
            } = expr
            else {
                self.report(expr.pos(), format_args!("expected declaration"));
            };

            match right {
                E::Struct { fields, .. } => {
                    let fields = fields
                        .iter()
                        .map(|&(name, ty)| (name.into(), self.ty(&ty)))
                        .collect();
                    self.records
                        .iter_mut()
                        .find(|r| r.id == *id)
                        .unwrap()
                        .fields = fields;
                }
                E::Closure { body, args, .. } => {
                    log::dbg!("fn: {}", name);
                    let frame = self.define_fn_label(*id);
                    if *name == "main" {
                        self.main = Some(frame.label);
                    }

                    let fn_label = self.labels[frame.label as usize].clone();

                    log::dbg!("fn-args");
                    let mut parama = 3..12;
                    for (arg, &ty) in args.iter().zip(fn_label.args.iter()) {
                        let loc = self.load_arg(ty, &mut parama);
                        self.vars.push(Variable {
                            id:    arg.id,
                            value: Value { ty, loc },
                        });
                    }

                    self.gpa.init_callee();
                    self.ret = fn_label.ret;

                    log::dbg!("fn-body");
                    if self.expr(body).is_some() {
                        self.report(body.pos(), "expected all paths in the fucntion to return");
                    }
                    self.vars.clear();

                    log::dbg!("fn-prelude, stack: {:x}", self.stack_size);

                    log::dbg!("fn-relocs");
                    self.write_fn_prelude(frame);

                    log::dbg!("fn-ret");
                    self.reloc_rets();
                    self.ret();
                    self.stack_size = 0;
                }
                _ => unreachable!(),
            }
        }
    }

    fn align_of(&self, ty: Type) -> u64 {
        use TypeKind as TK;
        match TypeKind::from_ty(ty) {
            TK::Struct(t) => self.records[t as usize]
                .fields
                .iter()
                .map(|&(_, ty)| self.align_of(ty))
                .max()
                .unwrap(),
            _ => self.size_of(ty).max(1),
        }
    }

    fn size_of(&self, ty: Type) -> u64 {
        use TypeKind as TK;
        match TK::from_ty(ty) {
            TK::Pointer(_) => 8,
            TK::Builtin(bt::VOID) => 0,
            TK::Builtin(bt::NEVER) => unreachable!(),
            TK::Builtin(bt::INT | bt::I64 | bt::UINT | bt::U64) => 8,
            TK::Builtin(bt::I32 | bt::U32) => 4,
            TK::Builtin(bt::I16 | bt::U16) => 2,
            TK::Builtin(bt::I8 | bt::U8 | bt::BOOL) => 1,
            TK::Builtin(e) => unreachable!("{:?}", e),
            TK::Struct(ty) => {
                log::dbg!("size_of: {:?}", ty);
                let mut offset = 0;
                let record = &self.records[ty as usize];
                for &(_, ty) in record.fields.iter() {
                    let align = self.align_of(ty);
                    offset = (offset + align - 1) & !(align - 1);
                    offset += self.size_of(ty);
                }
                offset
            }
        }
    }

    fn display_ty(&self, ty: Type) -> TypeDisplay {
        TypeDisplay::new(self, ty)
    }

    fn offset_of(&self, pos: parser::Pos, ty: Type, field: &str) -> (u64, Type) {
        let TypeKind::Struct(idx) = TypeKind::from_ty(ty) else {
            self.report(
                pos,
                format_args!("expected struct, got {}", self.display_ty(ty)),
            );
        };
        let record = &self.records[idx as usize];
        let mut offset = 0;
        for (name, ty) in record.fields.iter() {
            if name.as_ref() == field {
                return (offset, *ty);
            }
            let align = self.align_of(*ty);
            offset = (offset + align - 1) & !(align - 1);
            offset += self.size_of(*ty);
        }
        self.report(pos, format_args!("field not found: {}", field));
    }

    fn loc_to_reg(&mut self, loc: Loc) -> LinReg {
        match loc {
            Loc::RegRef(rr) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::cp(reg.0, rr));
                reg
            }
            Loc::Return => todo!(),
            Loc::Reg(reg) => reg,
            Loc::Deref(dreg, offset) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::ld(reg.0, dreg.0, offset, 8));
                self.gpa.free(dreg);
                reg
            }
            Loc::DerefRef(dreg, offset) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::ld(reg.0, dreg, offset, 8));
                reg
            }
            Loc::Imm(imm) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::li64(reg.0, imm));
                reg
            }
            Loc::Stack(offset) | Loc::StackRef(offset) => {
                let reg = self.gpa.allocate();
                self.load_stack(reg.0, offset, 8);
                reg
            }
        }
    }

    fn alloc_stack(&mut self, size: u32) -> u64 {
        let offset = self.stack_size;
        log::dbg!("alloc_stack: {} {}", offset, size);
        self.stack_size += size as u64;
        offset
    }

    fn store_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.code.encode(instrs::st(reg, STACK_PTR, offset, size));
    }

    fn load_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.code.encode(instrs::ld(reg, STACK_PTR, offset, size));
    }

    fn reloc_rets(&mut self) {
        let len = self.code.code.len() as i32;
        for reloc in self.ret_relocs.drain(..) {
            let dest = &mut self.code.code[reloc.offset as usize + reloc.instr_offset as usize..]
                [..reloc.size as usize];
            debug_assert!(dest.iter().all(|&b| b == 0));
            let offset = len - reloc.offset as i32;
            dest.copy_from_slice(&offset.to_ne_bytes());
        }
    }

    fn ty(&mut self, expr: &parser::Expr<'a>) -> Type {
        match *expr {
            E::Ident { name: "int", .. } => bt::INT,
            E::Ident { name: "bool", .. } => bt::BOOL,
            E::Ident { name: "void", .. } => bt::VOID,
            E::Ident { name: "never", .. } => bt::NEVER,
            E::UnOp {
                op: T::Star, val, ..
            } => {
                let ty = self.ty(val);
                self.alloc_pointer(ty)
            }
            E::Ident { id, name, .. } => {
                let Some(index) = self.records.iter().position(|r| r.id == id) else {
                    self.report(expr.pos(), format_args!("unknown type: {}", name))
                };
                TypeKind::Struct(index as Type).encode()
            }
            expr => unimplemented!("type: {:#?}", expr),
        }
    }

    fn expr(&mut self, expr: &'a parser::Expr<'a>) -> Option<Value> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn expr_ctx(&mut self, expr: &'a parser::Expr<'a>, ctx: Ctx) -> Option<Value> {
        let value = match *expr {
            E::Bool { value, .. } => Some(Value {
                ty:  bt::BOOL,
                loc: Loc::Imm(value as u64),
            }),
            E::Ctor { ty, fields } => {
                let ty = self.ty(&ty);
                let size = self.size_of(ty);

                let loc = match ctx {
                    Ctx::Dest(dest) => dest.loc,
                    _ => Loc::Stack(self.alloc_stack(size as u32)),
                };

                for (name, field) in fields {
                    let (offset, ty) = self.offset_of(expr.pos(), ty, name);
                    let loc = loc.offset_ref(offset);
                    self.expr_ctx(field, Ctx::Dest(Value { ty, loc }))?;
                }

                return Some(Value { ty, loc });
            }
            E::Field { target, field } => {
                let mut tal = self.expr(target)?;
                if let TypeKind::Pointer(ty) = TypeKind::from_ty(tal.ty) {
                    tal.ty = self.pointers[ty as usize];
                    tal.loc = match tal.loc {
                        Loc::Reg(r) => Loc::Deref(r, 0),
                        Loc::RegRef(r) => Loc::DerefRef(r, 0),
                        Loc::StackRef(stack) | Loc::Stack(stack) => {
                            let reg = self.gpa.allocate();
                            self.load_stack(reg.0, stack, 8);
                            Loc::Deref(reg, 0)
                        }
                        l => todo!("cant get field of {:?}", l),
                    };
                }
                let (offset, ty) = self.offset_of(target.pos(), tal.ty, field);
                let loc = match tal.loc {
                    Loc::Deref(r, off) => Loc::Deref(r, off + offset),
                    Loc::DerefRef(r, off) => Loc::DerefRef(r, off + offset),
                    Loc::Stack(stack) => Loc::Stack(stack + offset),
                    Loc::StackRef(stack) => Loc::StackRef(stack + offset),
                    l => todo!("cant get field of {:?}", l),
                };
                Some(Value { ty, loc })
            }
            E::UnOp {
                op: T::Amp,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                let loc = match val.loc {
                    Loc::StackRef(off) => {
                        let reg = self.gpa.allocate();
                        self.code.encode(instrs::addi64(reg.0, STACK_PTR, off));
                        Loc::Reg(reg)
                    }
                    Loc::Deref(r, off) => {
                        self.code.encode(instrs::addi64(r.0, r.0, off));
                        Loc::Reg(r)
                    }
                    Loc::DerefRef(r, off) => {
                        let reg = self.gpa.allocate();
                        self.code.encode(instrs::addi64(reg.0, r, off));
                        Loc::Reg(reg)
                    }
                    l => self.report(
                        pos,
                        format_args!("cant take pointer of {} ({:?})", self.display_ty(val.ty), l),
                    ),
                };
                Some(Value {
                    ty: self.alloc_pointer(val.ty),
                    loc,
                })
            }
            E::UnOp {
                op: T::Star,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                let reg = self.loc_to_reg(val.loc);
                match TypeKind::from_ty(val.ty) {
                    TypeKind::Pointer(ty) => Some(Value {
                        ty:  self.pointers[ty as usize],
                        loc: Loc::Deref(reg, 0),
                    }),
                    _ => self.report(
                        pos,
                        format_args!("expected pointer, got {}", self.display_ty(val.ty)),
                    ),
                }
            }
            E::BinOp {
                left: E::Ident { id, .. },
                op: T::Decl,
                right,
            } => {
                let val = self.expr(right)?;
                let loc = self.make_loc_owned(val.loc, val.ty);
                let loc = self.ensure_spilled(loc);
                self.vars.push(Variable {
                    id:    *id,
                    value: Value { ty: val.ty, loc },
                });
                Some(Value::VOID)
            }
            E::Call {
                func: E::Ident { id, .. },
                args,
            } => {
                let func = self.get_label(*id);
                let fn_label = self.labels[func as usize].clone();
                let mut parama = 3..12;
                for (earg, &ty) in args.iter().zip(fn_label.args.iter()) {
                    let arg = self.expr_ctx(earg, Ctx::Inferred(ty))?;
                    self.assert_ty(earg.pos(), ty, arg.ty);
                    self.pass_arg(arg, &mut parama);
                }

                let size = self.size_of(fn_label.ret);
                let loc = match size {
                    0 => Loc::Imm(0),
                    8 => Loc::RegRef(1),
                    16 => match ctx {
                        Ctx::Dest(dest) => dest.loc,
                        _ => Loc::Stack(self.alloc_stack(size as u32)),
                    },
                    24..=u64::MAX => {
                        let val = match ctx {
                            Ctx::Dest(dest) => dest.loc,
                            _ => Loc::Stack(self.alloc_stack(size as u32)),
                        };
                        let (ptr, off) = val.ref_to_ptr(size);
                        self.code.encode(instrs::cp(1, ptr));
                        self.code.encode(instrs::addi64(1, ptr, off));
                        val
                    }
                    s => todo!("call return size: {}", s),
                };

                self.code.call(func);

                match size {
                    0 => {}
                    8 => {}
                    16 => {
                        if let Loc::Stack(stack) = loc {
                            self.store_stack(1, stack, 16);
                        }
                    }
                    24..=u64::MAX => {}
                    s => todo!("call return size: {}", s),
                }

                return Some(Value {
                    ty: fn_label.ret,
                    loc,
                });
            }
            E::Ident { name, id, .. } => {
                let Some(var) = self.vars.iter().find(|v| v.id == id) else {
                    self.report(expr.pos(), format_args!("unknown variable: {}", name))
                };
                Some(Value {
                    ty:  var.value.ty,
                    loc: var.value.loc.take_ref(),
                })
            }
            E::Return { val, pos } => {
                if let Some(val) = val {
                    let val = self.expr_ctx(val, Ctx::Inferred(self.ret))?;
                    self.assert_ty(pos, self.ret, val.ty);
                    self.assign(self.ret, Loc::Return, val.loc);
                }
                self.ret_relocs.push(RetReloc {
                    offset:       self.code.code.len() as u32,
                    instr_offset: 1,
                    size:         4,
                });
                self.code.encode(instrs::jmp(0));
                None
            }
            E::Block { stmts, .. } => {
                for stmt in stmts {
                    if let Loc::Reg(reg) = self.expr(stmt)?.loc {
                        self.gpa.free(reg);
                    }
                }
                Some(Value::VOID)
            }
            E::Number { value, .. } => Some(Value {
                ty:  ctx.ty().unwrap_or(bt::INT),
                loc: Loc::Imm(value),
            }),
            E::If {
                cond, then, else_, ..
            } => 'b: {
                log::dbg!("if-cond");
                let cond = self.expr_ctx(cond, Ctx::Inferred(bt::BOOL))?;
                let reg = self.loc_to_reg(cond.loc);
                let jump_offset = self.code.code.len() as u32;
                self.code.encode(instrs::jeq(reg.0, 0, 0));
                self.gpa.free(reg);

                log::dbg!("if-then");
                let then_unreachable = self.expr(then).is_none();
                let mut else_unreachable = false;

                let mut jump = self.code.code.len() as i16 - jump_offset as i16;

                if let Some(else_) = else_ {
                    log::dbg!("if-else");
                    let else_jump_offset = self.code.code.len() as u32;
                    if !then_unreachable {
                        self.code.encode(instrs::jmp(0));
                        jump = self.code.code.len() as i16 - jump_offset as i16;
                    }

                    else_unreachable = self.expr(else_).is_none();

                    if !then_unreachable {
                        let jump = self.code.code.len() as i32 - else_jump_offset as i32;
                        log::dbg!("if-else-jump: {}", jump);
                        self.code.code[else_jump_offset as usize + 1..][..4]
                            .copy_from_slice(&jump.to_ne_bytes());
                    }
                }

                log::dbg!("if-then-jump: {}", jump);
                self.code.code[jump_offset as usize + 3..][..2]
                    .copy_from_slice(&jump.to_ne_bytes());

                if then_unreachable && else_unreachable {
                    break 'b None;
                }

                Some(Value::VOID)
            }
            E::Loop { body, .. } => 'a: {
                log::dbg!("loop");
                let loop_start = self.code.code.len() as u32;
                self.loops.push(Loop {
                    offset: loop_start,
                    relocs: Default::default(),
                });
                let body_unreachable = self.expr(body).is_none();

                log::dbg!("loop-end");
                if !body_unreachable {
                    let loop_end = self.code.code.len();
                    self.code
                        .encode(instrs::jmp(loop_start as i32 - loop_end as i32));
                }

                let loop_end = self.code.code.len() as u32;

                let loop_ = self.loops.pop().unwrap();
                let is_unreachable = loop_.relocs.is_empty();
                for reloc in loop_.relocs {
                    let dest = &mut self.code.code
                        [reloc.offset as usize + reloc.instr_offset as usize..]
                        [..reloc.size as usize];
                    let offset = loop_end as i32 - reloc.offset as i32;
                    dest.copy_from_slice(&offset.to_ne_bytes());
                }

                if is_unreachable {
                    log::dbg!("infinite loop");
                    break 'a None;
                }

                Some(Value::VOID)
            }
            E::Break { .. } => {
                let loop_ = self.loops.last_mut().unwrap();
                let offset = self.code.code.len() as u32;
                self.code.encode(instrs::jmp(0));
                loop_.relocs.push(RetReloc {
                    offset,
                    instr_offset: 1,
                    size: 4,
                });
                None
            }
            E::Continue { .. } => {
                let loop_ = self.loops.last().unwrap();
                let offset = self.code.code.len() as u32;
                self.code
                    .encode(instrs::jmp(loop_.offset as i32 - offset as i32));
                None
            }
            E::BinOp { left, op, right } => {
                if op == T::Assign {
                    let left = self.expr(left)?;
                    let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                    return self.assign(left.ty, left.loc, right.loc);
                }

                let left = self.expr(left)?;
                let lhs = self.loc_to_reg(left.loc);
                let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                let rhs = self.loc_to_reg(right.loc);

                let op = match op {
                    T::Plus => instrs::add64,
                    T::Minus => instrs::sub64,
                    T::Star => instrs::mul64,
                    T::Le => {
                        self.code.encode(instrs::cmpu(lhs.0, lhs.0, rhs.0));
                        self.gpa.free(rhs);
                        self.code.encode(instrs::cmpui(lhs.0, lhs.0, 1));
                        return Some(Value {
                            ty:  bt::BOOL,
                            loc: Loc::Reg(lhs),
                        });
                    }
                    T::Eq => {
                        self.code.encode(instrs::cmpu(lhs.0, lhs.0, rhs.0));
                        self.gpa.free(rhs);
                        self.code.encode(instrs::cmpui(lhs.0, lhs.0, 0));
                        self.code.encode(instrs::not(lhs.0, lhs.0));
                        return Some(Value {
                            ty:  bt::BOOL,
                            loc: Loc::Reg(lhs),
                        });
                    }
                    T::FSlash => |reg0, reg1, reg2| instrs::diru64(reg0, ZERO, reg1, reg2),
                    _ => unimplemented!("{:#?}", op),
                };

                self.code.encode(op(lhs.0, lhs.0, rhs.0));
                self.gpa.free(rhs);

                Some(Value {
                    ty:  left.ty,
                    loc: Loc::Reg(lhs),
                })
            }
            ast => unimplemented!("{:#?}", ast),
        }?;

        if let Ctx::Dest(dest) = ctx {
            self.assign(dest.ty, dest.loc, value.loc);
            Some(Value::VOID)
        } else {
            Some(value)
        }
    }

    fn assign(&mut self, ty: Type, right: Loc, left: Loc) -> Option<Value> {
        let size = self.size_of(ty);

        match size {
            8 => {
                let lhs = self.loc_to_reg(left);
                match right {
                    Loc::Deref(reg, off) => {
                        self.code.encode(instrs::st(lhs.0, reg.0, off, 8));
                        self.gpa.free(reg);
                    }
                    Loc::RegRef(reg) => self.code.encode(instrs::cp(reg, lhs.0)),
                    Loc::StackRef(offset) | Loc::Stack(offset) => {
                        self.store_stack(lhs.0, offset, 8)
                    }
                    Loc::Return => self.code.encode(instrs::cp(1, lhs.0)),
                    l => unimplemented!("{:?}", l),
                }
                self.gpa.free(lhs);
            }
            16 if matches!(right, Loc::Return) => {
                let (lhs, loff) = left.to_ptr(size);
                self.code.encode(instrs::st(1, lhs.get(), loff, 16));
                self.gpa.free_cow(lhs);
            }
            16..=u64::MAX => {
                let (rhs, roff) = right.to_ptr(size);
                let (lhs, loff) = left.to_ptr(size);
                let (rhs, lhs) = (self.to_owned(rhs), self.to_owned(lhs));

                self.code.encode(instrs::addi64(rhs.0, rhs.0, roff));
                self.code.encode(instrs::addi64(lhs.0, lhs.0, loff));
                self.code
                    .encode(instrs::bmc(lhs.0, rhs.0, size.try_into().unwrap()));

                self.gpa.free(rhs);
                self.gpa.free(lhs);
            }
            s => unimplemented!("size: {}", s),
        }

        Some(Value::VOID)
    }

    fn to_owned(&mut self, loc: CowReg) -> LinReg {
        match loc {
            CowReg::Lin(reg) => reg,
            CowReg::Reg(reg) => {
                let new = self.gpa.allocate();
                self.code.encode(instrs::cp(new.0, reg));
                new
            }
        }
    }

    fn declare_fn_label(&mut self, name: Ident, args: Rc<[Type]>, ret: Type) -> LabelId {
        self.labels.push(FnLabel {
            offset: 0,
            name,
            args,
            ret,
        });
        self.labels.len() as u32 - 1
    }

    fn define_fn_label(&mut self, name: Ident) -> Frame {
        let offset = self.code.code.len() as u32;
        let label = self.get_label(name);
        self.labels[label as usize].offset = offset;
        Frame {
            label,
            prev_relocs: self.code.relocs.len(),
            offset,
        }
    }

    fn get_label(&self, name: Ident) -> LabelId {
        self.labels.iter().position(|l| l.name == name).unwrap() as _
    }

    fn write_fn_prelude(&mut self, frame: Frame) {
        self.temp.push(RET_ADDR, self.gpa.pushed_size());
        self.temp.subi64(STACK_PTR, STACK_PTR, self.stack_size);

        for reloc in &mut self.code.relocs[frame.prev_relocs..] {
            reloc.offset += self.temp.code.len() as u32;
        }

        for reloc in &mut self.ret_relocs {
            reloc.offset += self.temp.code.len() as u32;
        }

        self.code.code.splice(
            frame.offset as usize..frame.offset as usize,
            self.temp.code.drain(..),
        );
    }

    fn ret(&mut self) {
        self.code
            .encode(instrs::addi64(STACK_PTR, STACK_PTR, self.stack_size));
        self.code.pop(RET_ADDR, self.gpa.pushed_size());
        self.code.ret();
    }

    pub fn dump(mut self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        self.temp.prelude(self.main.unwrap());
        self.temp
            .relocate(&self.labels, self.temp.code.len() as i64);

        self.code.relocate(&self.labels, 0);
        out.write_all(&self.temp.code)?;
        out.write_all(&self.code.code)
    }

    fn alloc_pointer(&mut self, ty: Type) -> Type {
        let ty = self
            .pointers
            .iter()
            .position(|&p| p == ty)
            .unwrap_or_else(|| {
                self.pointers.push(ty);
                self.pointers.len() - 1
            });

        TypeKind::Pointer(ty as Type).encode()
    }

    fn make_loc_owned(&mut self, loc: Loc, ty: Type) -> Loc {
        match loc {
            Loc::RegRef(rreg) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::cp(reg.0, rreg));
                Loc::Reg(reg)
            }
            Loc::Imm(imm) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::li64(reg.0, imm));
                Loc::Reg(reg)
            }
            Loc::StackRef(off) => {
                let size = self.size_of(ty);
                let stack = self.alloc_stack(size as u32);
                self.assign(ty, Loc::Stack(stack), Loc::StackRef(off));
                Loc::Stack(stack)
            }
            l => l,
        }
    }

    fn pass_arg(&mut self, value: Value, parama: &mut Range<u8>) {
        let size = self.size_of(value.ty);
        let p = parama.next().unwrap() as Reg;

        if size > 16 {
            let (Loc::Stack(stack) | Loc::StackRef(stack)) = value.loc else {
                todo!("expected stack location, got {:?}", value.loc);
            };
            self.code.encode(instrs::addi64(p, STACK_PTR, stack));
            return;
        }

        match value.loc {
            Loc::Reg(reg) => {
                self.code.encode(instrs::cp(p, reg.0));
                self.gpa.free(reg);
            }
            Loc::RegRef(reg) => {
                self.code.encode(instrs::cp(p, reg));
            }
            Loc::Return => todo!(),
            Loc::Deref(reg, off) => {
                self.code.encode(instrs::ld(p, reg.0, off, size as _));
                self.gpa.free(reg);
            }
            Loc::DerefRef(reg, off) => {
                self.code.encode(instrs::ld(p, reg, off, size as _));
            }
            Loc::Imm(imm) => {
                self.code.encode(instrs::li64(p, imm));
            }
            Loc::Stack(stack) | Loc::StackRef(stack) => {
                self.load_stack(p, stack, size as _);
                self.load_stack(parama.next().unwrap(), stack + 8, size as _);
            }
        }
    }

    fn load_arg(&mut self, ty: Type, parama: &mut Range<u8>) -> Loc {
        let size = self.size_of(ty);
        match size {
            8 => {
                let stack = self.alloc_stack(8);
                self.store_stack(parama.next().unwrap(), stack, 8);
                Loc::Stack(stack)
            }
            16 => {
                let stack = self.alloc_stack(16);
                self.store_stack(parama.next().unwrap(), stack, 8);
                self.store_stack(parama.next().unwrap(), stack + 8, 8);
                Loc::Stack(stack)
            }
            24..=u64::MAX => {
                let ptr = parama.next().unwrap();
                let stack = self.alloc_stack(size as u32);
                self.assign(ty, Loc::StackRef(stack), Loc::DerefRef(ptr, 0));
                Loc::Stack(stack)
            }
            blah => todo!("{blah:?}"),
        }
    }

    fn ensure_spilled(&mut self, loc: Loc) -> Loc {
        match loc {
            Loc::Reg(reg) => {
                let stack = self.alloc_stack(8);
                self.store_stack(reg.0, stack, 8);
                self.gpa.free(reg);
                Loc::Stack(stack)
            }
            l => l,
        }
    }

    fn assert_ty(&self, pos: parser::Pos, ty: Type, expected: Type) {
        if ty != expected {
            let ty = self.display_ty(ty);
            let expected = self.display_ty(expected);
            self.report(pos, format_args!("expected {ty}, got {expected}"));
        }
    }

    fn report(&self, pos: parser::Pos, msg: impl std::fmt::Display) -> ! {
        let (line, col) = lexer::line_col(self.input, pos);
        println!("{}:{}:{}: {}", self.path, line, col, msg);
        unreachable!();
    }
}

pub struct Value {
    ty:  Type,
    loc: Loc,
}

impl Value {
    const VOID: Self = Self {
        ty:  bt::VOID,
        loc: Loc::Imm(0),
    };
}

#[derive(Debug)]
enum Loc {
    Reg(LinReg),
    RegRef(Reg),
    Return,
    Deref(LinReg, u64),
    DerefRef(Reg, u64),
    Imm(u64),
    Stack(u64),
    StackRef(u64),
}
impl Loc {
    fn take_ref(&self) -> Loc {
        match self {
            Self::Reg(reg) => Self::RegRef(reg.0),
            Self::Stack(off) => Self::StackRef(*off),
            un => unreachable!("{:?}", un),
        }
    }

    fn to_ptr(self, size: u64) -> (CowReg, u64) {
        match self {
            Loc::Return if size > 16 => (CowReg::Reg(1), 0),
            Loc::Deref(reg, off) => (CowReg::Lin(reg), off),
            Loc::DerefRef(reg, off) => (CowReg::Reg(reg), off),
            Loc::Stack(offset) | Loc::StackRef(offset) => (CowReg::Reg(STACK_PTR), offset),
            l => panic!("expected stack location, got {:?}", l),
        }
    }

    fn ref_to_ptr(&self, size: u64) -> (Reg, u64) {
        match *self {
            Loc::Return if size > 16 => (1, 0),
            Loc::Deref(LinReg(reg), off) => (reg, off),
            Loc::DerefRef(reg, off) => (reg, off),
            Loc::Stack(offset) | Loc::StackRef(offset) => (STACK_PTR, offset),
            ref l => panic!("expected stack location, got {:?}", l),
        }
    }

    fn offset_ref(&self, offset: u64) -> Loc {
        match *self {
            Self::Deref(LinReg(r), off) => Self::DerefRef(r, off + offset),
            Self::DerefRef(r, off) => Self::DerefRef(r, off + offset),
            Self::Stack(off) => Self::Stack(off + offset),
            Self::StackRef(off) => Self::StackRef(off + offset),
            ref un => unreachable!("{:?}", un),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{instrs, log};

    struct TestMem;

    impl hbvm::mem::Memory for TestMem {
        #[inline]
        unsafe fn load(
            &mut self,
            addr: hbvm::mem::Address,
            target: *mut u8,
            count: usize,
        ) -> Result<(), hbvm::mem::LoadError> {
            log::dbg!(
                "read: {:x} {} {:?}",
                addr.get(),
                count,
                core::slice::from_raw_parts(addr.get() as *const u8, count)
                    .iter()
                    .rev()
                    .skip_while(|&&b| b == 0)
                    .map(|&b| format!("{:02x}", b))
                    .collect::<String>()
            );
            unsafe { core::ptr::copy(addr.get() as *const u8, target, count) }
            Ok(())
        }

        #[inline]
        unsafe fn store(
            &mut self,
            addr: hbvm::mem::Address,
            source: *const u8,
            count: usize,
        ) -> Result<(), hbvm::mem::StoreError> {
            log::dbg!(
                "write: {:x} {} {:?}",
                addr.get(),
                count,
                core::slice::from_raw_parts(source, count)
                    .iter()
                    .rev()
                    .skip_while(|&&b| b == 0)
                    .map(|&b| format!("{:02x}", b))
                    .collect::<String>()
            );
            unsafe { core::ptr::copy(source, addr.get() as *mut u8, count) }
            Ok(())
        }

        #[inline]
        unsafe fn prog_read<T: Copy>(&mut self, addr: hbvm::mem::Address) -> T {
            log::dbg!(
                "read-typed: {:x} {} {:?}",
                addr.get(),
                std::any::type_name::<T>(),
                if core::mem::size_of::<T>() == 1 {
                    instrs::NAMES[std::ptr::read(addr.get() as *const u8) as usize].to_string()
                } else {
                    core::slice::from_raw_parts(addr.get() as *const u8, core::mem::size_of::<T>())
                        .iter()
                        .map(|&b| format!("{:02x}", b))
                        .collect::<String>()
                }
            );
            unsafe { core::ptr::read(addr.get() as *const T) }
        }
    }

    fn generate(input: &'static str, output: &mut String) {
        let path = "test";
        let arena = crate::parser::Arena::default();
        let mut parser = super::parser::Parser::new(&arena);
        let exprs = parser.file(input, path);
        let mut codegen = super::Codegen::default();
        codegen.file(path, input.as_bytes(), &exprs);
        let mut out = Vec::new();
        codegen.dump(&mut out).unwrap();

        use std::fmt::Write;

        let mut stack = [0_u64; 128];

        let mut vm = unsafe {
            hbvm::Vm::<TestMem, 0>::new(TestMem, hbvm::mem::Address::new(out.as_ptr() as u64))
        };

        vm.write_reg(
            super::STACK_PTR,
            unsafe { stack.as_mut_ptr().add(stack.len()) } as u64,
        );

        let stat = loop {
            match vm.run() {
                Ok(hbvm::VmRunOk::End) => break Ok(()),
                Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
                Err(e) => break Err(e),
            }
        };

        writeln!(output, "code size: {}", out.len()).unwrap();
        writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
        writeln!(output, "status: {:?}", stat).unwrap();
    }

    crate::run_tests! { generate:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
        variables => include_str!("../examples/variables.hb");
        functions => include_str!("../examples/functions.hb");
        if_statements => include_str!("../examples/if_statement.hb");
        loops => include_str!("../examples/loops.hb");
        fb_driver => include_str!("../examples/fb_driver.hb");
        pointers => include_str!("../examples/pointers.hb");
        structs => include_str!("../examples/structs.hb");
    }
}
