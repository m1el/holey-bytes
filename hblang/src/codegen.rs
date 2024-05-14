use std::{cell::Cell, ops::Range};

use crate::ident::{self, Ident};

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

fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

#[derive(Debug, PartialEq, Eq)]
struct LinReg(Reg);

#[cfg(debug_assertions)]
impl Drop for LinReg {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("reg leaked: {:x}", self.0);
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct LinStack {
    offset: u64,
    size:   u64,
}

impl Drop for LinStack {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            panic!("stack leaked: {:x} {:x}", self.offset, self.size);
        }
    }
}

#[derive(Default)]
struct StackAlloc {
    ranges: Vec<Range<u64>>,
    height: u64,
}

impl StackAlloc {
    fn alloc(&mut self, size: u64) -> LinStack {
        if let Some((index, range)) = self
            .ranges
            .iter_mut()
            .enumerate()
            .filter(|(_, range)| range.end - range.start >= size)
            .min_by_key(|(_, range)| range.end - range.start)
        {
            let offset = range.start;
            range.start += size;
            if range.start == range.end {
                self.ranges.swap_remove(index);
            }
            return LinStack { offset, size };
        }

        let offset = self.height;
        self.height += size;
        LinStack { offset, size }
    }

    fn free(&mut self, l @ LinStack { offset, size }: LinStack) {
        std::mem::forget(l);

        let range = offset..offset + size;
        // FIXME: we do more wor then we need to, rather we keep the sequence sorted and only scan
        // element before and after the modified range
        self.ranges.push(range);
        self.ranges.sort_by_key(|range| range.start);
        self.ranges.dedup_by(|b, a| {
            if a.end == b.start {
                a.end = b.end;
                true
            } else {
                false
            }
        });
    }

    fn clear(&mut self) {
        assert!(self.ranges.len() <= 1, "{:?}", self.ranges);
        self.ranges.clear();
        self.height = 0;
    }
}

#[derive(Default, Debug)]
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

    macro_rules! builtin_type {
        ($($name:ident;)*) => {$(
            pub const $name: Type = ${index(0)};
        )*};
    }

    builtin_type! {
        NEVER;
        VOID;
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

    pub fn is_signed(ty: Type) -> bool {
        ty >= I8 && ty <= INT
    }

    pub fn is_unsigned(ty: Type) -> bool {
        ty >= U8 && ty <= UINT
    }

    pub fn strip_pointer(ty: Type) -> Type {
        match TypeKind::from_ty(ty) {
            TypeKind::Pointer(_) => INT,
            _ => ty,
        }
    }

    pub fn is_pointer(ty: Type) -> bool {
        matches!(TypeKind::from_ty(ty), TypeKind::Pointer(_))
    }

    pub fn try_upcast(oa: Type, ob: Type) -> Option<Type> {
        let (oa, ob) = (oa.min(ob), oa.max(ob));
        let (a, b) = (strip_pointer(oa), strip_pointer(ob));
        Some(match () {
            _ if oa == ob => oa,
            _ if is_signed(a) && is_signed(b) || is_unsigned(a) && is_unsigned(b) => ob,
            _ if is_unsigned(a) && is_signed(b) && a - U8 < b - I8 => ob,
            _ => return None,
        })
    }
}

#[derive(Debug)]
enum TypeKind {
    Builtin(Type),
    Struct(Type),
    Pointer(Type),
}

impl TypeKind {
    const FLAG_BITS: u32 = 2;
    const FLAG_OFFSET: u32 = std::mem::size_of::<Type>() as u32 * 8 - Self::FLAG_BITS;
    const INDEX_MASK: u32 = (1 << (32 - Self::FLAG_BITS)) - 1;

    fn from_ty(ty: Type) -> Self {
        let (flag, index) = (ty >> Self::FLAG_OFFSET, ty & Self::INDEX_MASK);
        match flag {
            0 => Self::Builtin(index),
            1 => Self::Pointer(index),
            2 => Self::Struct(index),
            _ => unreachable!(),
        }
    }

    const fn encode(self) -> Type {
        let (index, flag) = match self {
            Self::Builtin(index) => (index, 0),
            Self::Pointer(index) => (index, 1),
            Self::Struct(index) => (index, 2),
        };
        (flag << Self::FLAG_OFFSET) | index
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

    fn short_cut_bin_op(&mut self, dest: Reg, src: Reg, imm: u64) -> bool {
        if imm == 0 {
            if dest != src {
                self.encode(instrs::cp(dest, src));
            }
        }
        imm != 0
    }

    fn subi64(&mut self, dest: Reg, src: Reg, imm: u64) {
        if self.short_cut_bin_op(dest, src, imm) {
            self.encode(instrs::addi64(dest, src, imm.wrapping_neg()));
        }
    }

    fn addi64(&mut self, dest: Reg, src: Reg, imm: u64) {
        if self.short_cut_bin_op(dest, src, imm) {
            self.encode(instrs::addi64(dest, src, imm));
        }
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
    var_count: usize,
    offset:    u32,
    relocs:    Vec<RetReloc>,
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
            TK::Builtin(bt::I32) => "i32",
            TK::Builtin(bt::I16) => "i16",
            TK::Builtin(bt::I8) => "i8",
            TK::Builtin(bt::UINT) => "uint",
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
    sa:         StackAlloc,
    code:       Func,
    temp:       Func,
    labels:     Vec<FnLabel>,
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
                    for var in std::mem::take(&mut self.vars) {
                        self.free_loc(var.value.loc);
                    }

                    log::dbg!("fn-prelude, stack: {:x}", self.sa.height);

                    log::dbg!("fn-relocs");
                    self.write_fn_prelude(frame);

                    log::dbg!("fn-ret");
                    self.reloc_rets();
                    self.ret();
                    self.sa.clear();
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
            TK::Builtin(bt::INT | bt::UINT) => 8,
            TK::Builtin(bt::I32 | bt::U32) => 4,
            TK::Builtin(bt::I16 | bt::U16) => 2,
            TK::Builtin(bt::I8 | bt::U8 | bt::BOOL) => 1,
            TK::Builtin(e) => unreachable!("{:?}", e),
            TK::Struct(ty) => {
                let mut offset = 0;
                let record = &self.records[ty as usize];
                for &(_, ty) in record.fields.iter() {
                    let align = self.align_of(ty);
                    offset = align_up(offset, align);
                    offset += self.size_of(ty);
                }
                offset
            }
        }
    }

    fn display_ty(&self, ty: Type) -> TypeDisplay {
        TypeDisplay::new(self, ty)
    }

    fn unwrap_struct(&self, ty: Type, pos: parser::Pos, context: impl std::fmt::Display) -> Type {
        match TypeKind::from_ty(ty) {
            TypeKind::Struct(idx) => idx,
            _ => self.report(
                pos,
                format_args!("expected struct, got {} ({context})", self.display_ty(ty)),
            ),
        }
    }

    fn offset_of(&self, pos: parser::Pos, ty: Type, field: Result<&str, usize>) -> (u64, Type) {
        let idx = self.unwrap_struct(ty, pos, "field access");
        let record = &self.records[idx as usize];
        let mut offset = 0;
        for (i, (name, ty)) in record.fields.iter().enumerate() {
            if Ok(name.as_ref()) == field || Err(i) == field {
                return (offset, *ty);
            }
            let align = self.align_of(*ty);
            offset = align_up(offset, align);
            offset += self.size_of(*ty);
        }

        match field {
            Ok(i) => self.report(pos, format_args!("field not found: {i}")),
            Err(field) => self.report(pos, format_args!("field not found: {field}")),
        }
    }

    fn loc_to_reg(&mut self, loc: Loc, size: u64) -> LinReg {
        match loc {
            Loc::RegRef(rr) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::cp(reg.0, rr));
                reg
            }
            Loc::Reg(reg) => reg,
            Loc::Deref(dreg, offset) => {
                let reg = self.gpa.allocate();
                self.code
                    .encode(instrs::ld(reg.0, dreg.0, offset, size as _));
                self.free_reg(dreg);
                reg
            }
            Loc::DerefRef(dreg, offset) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::ld(reg.0, dreg, offset, size as _));
                reg
            }
            Loc::Imm(imm) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::li64(reg.0, imm));
                reg
            }
            Loc::Stack(stack, off) => {
                let reg = self.gpa.allocate();
                self.load_stack(reg.0, stack.offset + off, size as _);
                self.sa.free(stack);
                reg
            }
        }
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
            E::Ident { id, .. } if ident::is_null(id) => id,
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
            E::Ctor {
                pos, ty, fields, ..
            } => {
                let Some(ty) = ty.map(|ty| self.ty(&ty)).or(ctx.ty()) else {
                    self.report(pos, "expected type, (it cannot be inferred)");
                };
                let size = self.size_of(ty);

                let loc = match ctx {
                    Ctx::Dest(dest) => dest.loc,
                    _ => Loc::Stack(self.sa.alloc(size), 0),
                };

                let stuct = self.unwrap_struct(ty, pos, "struct literal");
                let field_count = self.records[stuct as usize].fields.len();
                if field_count != fields.len() {
                    self.report(
                        pos,
                        format_args!("expected {} fields, got {}", field_count, fields.len()),
                    );
                }

                for (i, (name, field)) in fields.iter().enumerate() {
                    let (offset, ty) = self.offset_of(field.pos(), ty, name.ok_or(i));
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
                        l => Loc::Deref(self.loc_to_reg(l, 8), 0),
                    };
                }
                let (offset, ty) = self.offset_of(target.pos(), tal.ty, Ok(field));
                let loc = tal.loc.offset(offset);
                Some(Value { ty, loc })
            }
            E::UnOp {
                op: T::Amp,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                let loc = match val.loc {
                    Loc::Deref(r, off) => {
                        self.code.addi64(r.0, r.0, off);
                        Loc::Reg(r)
                    }
                    Loc::DerefRef(r, off) => {
                        let reg = self.gpa.allocate();
                        self.code.addi64(reg.0, r, off);
                        Loc::Reg(reg)
                    }
                    Loc::Stack(stack, off) => {
                        let reg = self.gpa.allocate();
                        self.code
                            .encode(instrs::addi64(reg.0, STACK_PTR, stack.offset + off));
                        std::mem::forget(stack); // TODO: we might be able to do better
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
                let reg = self.loc_to_reg(val.loc, self.size_of(val.ty));
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
                let loc = match loc {
                    Loc::Reg(r) => {
                        let size = self.size_of(val.ty);
                        let stack = self.sa.alloc(size);
                        self.store_stack(r.0, stack.offset, size as _);
                        self.free_reg(r);
                        Loc::Stack(stack, 0)
                    }
                    l => l,
                };
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
                let mut values = Vec::with_capacity(args.len());
                for (earg, &ty) in args.iter().zip(fn_label.args.iter()) {
                    let arg = self.expr_ctx(earg, Ctx::Inferred(ty))?;
                    _ = self.assert_ty(earg.pos(), ty, arg.ty);
                    self.pass_arg(&arg, &mut parama);
                    values.push(arg.loc);
                }

                for value in values {
                    self.free_loc(value);
                }

                let size = self.size_of(fn_label.ret);
                let loc = match size {
                    0 => Loc::Imm(0),
                    ..=8 => Loc::RegRef(1),
                    ..=16 => match ctx {
                        Ctx::Dest(dest) => dest.loc,
                        _ => Loc::Stack(self.sa.alloc(size), 0),
                    },
                    ..=u64::MAX => {
                        let val = match ctx {
                            Ctx::Dest(dest) => dest.loc,
                            _ => Loc::Stack(self.sa.alloc(size), 0),
                        };
                        let (ptr, off) = val.ref_to_ptr();
                        self.code.encode(instrs::cp(1, ptr));
                        self.code.addi64(1, ptr, off);
                        val
                    }
                };

                self.code.call(func);

                match size {
                    0 => {}
                    ..=8 => {}
                    ..=16 => {
                        if let Loc::Stack(ref stack, off) = loc {
                            self.store_stack(1, stack.offset + off, 16);
                        } else {
                            unreachable!()
                        }
                    }
                    ..=u64::MAX => {}
                }

                return Some(Value {
                    ty: fn_label.ret,
                    loc,
                });
            }
            E::Ident { name, id, last } => {
                let Some((index, var)) = self.vars.iter_mut().enumerate().find(|(_, v)| v.id == id)
                else {
                    self.report(expr.pos(), format_args!("unknown variable: {}", name))
                };

                let loc = match last.is_some_and(Cell::get)
                    && !self.loops.last().is_some_and(|l| l.var_count > index)
                {
                    true => std::mem::replace(&mut var.value.loc, Loc::Imm(0)),
                    false => var.value.loc.take_ref(),
                };

                Some(Value {
                    ty: var.value.ty,
                    loc,
                })
            }
            E::Return { val, .. } => {
                if let Some(val) = val {
                    let size = self.size_of(self.ret);
                    let loc = match size {
                        0 => Loc::Imm(0),
                        ..=16 => Loc::RegRef(1),
                        _ => Loc::DerefRef(1, 0),
                    };
                    self.expr_ctx(val, Ctx::Dest(Value { loc, ty: self.ret }))?;
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
                        self.free_reg(reg);
                    }
                }
                Some(Value::VOID)
            }
            E::Number { value, .. } => Some(Value {
                ty:  ctx.ty().map(bt::strip_pointer).unwrap_or(bt::INT),
                loc: Loc::Imm(value),
            }),
            E::If {
                cond, then, else_, ..
            } => 'b: {
                log::dbg!("if-cond");
                let cond = self.expr_ctx(cond, Ctx::Inferred(bt::BOOL))?;
                let reg = self.loc_to_reg(cond.loc, 1);
                let jump_offset = self.code.code.len() as u32;
                self.code.encode(instrs::jeq(reg.0, 0, 0));
                self.free_reg(reg);

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
                    var_count: self.vars.len() as _,
                    offset:    loop_start,
                    relocs:    Default::default(),
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

                for var in self
                    .vars
                    .drain(loop_.var_count as usize..)
                    .collect::<Vec<_>>()
                {
                    self.free_loc(var.value.loc);
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
            E::BinOp { left, op, right } => 'ops: {
                use instrs as i;

                log::dbg!("binop: {}", expr);
                let left = self.expr(left)?;

                if op == T::Assign {
                    self.expr_ctx(right, Ctx::Dest(left)).unwrap();
                    return Some(Value::VOID);
                }

                if let TypeKind::Struct(_) = TypeKind::from_ty(left.ty) {
                    let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                    _ = self.assert_ty(expr.pos(), left.ty, right.ty);
                    return self.struct_op(op, left.ty, ctx, left.loc, right.loc);
                }

                let lsize = self.size_of(left.ty);
                let lhs = self.loc_to_reg(left.loc, lsize);
                let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                let rsize = self.size_of(right.ty);
                let rhs = self.loc_to_reg(right.loc, rsize);

                let ty = self.assert_ty(expr.pos(), left.ty, right.ty);
                let size = self.size_of(ty);
                let signed = bt::is_signed(ty);

                log::dbg!(
                    "binop: {} {} {}",
                    self.display_ty(ty),
                    self.display_ty(left.ty),
                    self.display_ty(right.ty)
                );

                if matches!(op, T::Plus | T::Minus) {
                    let min_size = lsize.min(rsize);
                    if bt::is_signed(ty) && min_size < size {
                        let operand = if lsize < rsize { lhs.0 } else { rhs.0 };
                        let op = [i::sxt8, i::sxt16, i::sxt32][min_size.ilog2() as usize];
                        self.code.encode(op(operand, operand));
                    }

                    if bt::is_pointer(left.ty) ^ bt::is_pointer(right.ty) {
                        let (offset, ty) = if bt::is_pointer(left.ty) {
                            (rhs.0, left.ty)
                        } else {
                            (lhs.0, right.ty)
                        };

                        let TypeKind::Pointer(ty) = TypeKind::from_ty(ty) else {
                            unreachable!()
                        };

                        let size = self.size_of(self.pointers[ty as usize]);
                        self.code.encode(i::muli64(offset, offset, size as _));
                    }
                }

                if let Some(op) = Self::math_op(op, signed, size) {
                    self.code.encode(op(lhs.0, lhs.0, rhs.0));
                    self.free_reg(rhs);

                    break 'ops Some(Value {
                        ty,
                        loc: Loc::Reg(lhs),
                    });
                }

                'cmp: {
                    let against = match op {
                        T::Le | T::Lt => 1,
                        T::Ne | T::Eq => 0,
                        T::Ge | T::Gt => (-1i64) as _,
                        _ => break 'cmp,
                    };

                    let op_fn = if signed { i::cmps } else { i::cmpu };
                    self.code.encode(op_fn(lhs.0, lhs.0, rhs.0));
                    self.free_reg(rhs);
                    self.code.encode(instrs::cmpui(lhs.0, lhs.0, against));
                    if matches!(op, T::Eq | T::Lt | T::Gt) {
                        self.code.encode(instrs::not(lhs.0, lhs.0));
                    }

                    break 'ops Some(Value {
                        ty:  bt::BOOL,
                        loc: Loc::Reg(lhs),
                    });
                }

                unimplemented!("{:#?}", op)
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

    fn free_reg(&mut self, reg: LinReg) {
        self.gpa.free(reg);
    }

    fn math_op(
        op: T,
        signed: bool,
        size: u64,
    ) -> Option<fn(u8, u8, u8) -> (usize, [u8; instrs::MAX_SIZE])> {
        use instrs as i;
        Some(
            match op {
                T::Plus => [i::add8, i::add16, i::add32, i::add64],
                T::Minus => [i::sub8, i::sub16, i::sub32, i::sub64],
                T::Star => [i::mul8, i::mul16, i::mul32, i::mul64],
                T::FSlash if signed => [
                    |a, b, c| i::dirs8(a, ZERO, b, c),
                    |a, b, c| i::dirs16(a, ZERO, b, c),
                    |a, b, c| i::dirs32(a, ZERO, b, c),
                    |a, b, c| i::dirs64(a, ZERO, b, c),
                ],
                T::FSlash => [
                    |a, b, c| i::diru8(a, ZERO, b, c),
                    |a, b, c| i::diru16(a, ZERO, b, c),
                    |a, b, c| i::diru32(a, ZERO, b, c),
                    |a, b, c| i::diru64(a, ZERO, b, c),
                ],
                _ => return None,
            }[size.ilog2() as usize],
        )
    }

    fn free_loc(&mut self, loc: Loc) {
        match loc {
            Loc::Reg(reg) => self.free_reg(reg),
            Loc::Deref(reg, ..) => self.free_reg(reg),
            Loc::Stack(stack, ..) => self.sa.free(stack),
            _ => {}
        }
    }

    fn struct_op(&mut self, op: T, ty: Type, ctx: Ctx, left: Loc, right: Loc) -> Option<Value> {
        if let TypeKind::Struct(stuct) = TypeKind::from_ty(ty) {
            let dst = match ctx {
                Ctx::Dest(dest) => dest.loc,
                _ => Loc::Stack(self.sa.alloc(self.size_of(ty)), 0),
            };
            let mut offset = 0;
            for &(_, ty) in self.records[stuct as usize].fields.clone().iter() {
                let align = self.align_of(ty);
                offset = align_up(offset, align);
                let size = self.size_of(ty);
                let ctx = Ctx::Dest(Value::new(ty, dst.offset_ref(offset)));
                let left = left.offset_ref(offset);
                let right = right.offset_ref(offset);
                let value = self.struct_op(op, ty, ctx, left, right)?;
                self.free_loc(value.loc);
                offset += size;
            }

            self.free_loc(left);
            self.free_loc(right);

            return Some(Value { ty, loc: dst });
        }

        let size = self.size_of(ty);
        let (lhs, owned) = self.loc_to_reg_ref(&left, size);
        let rhs = self.loc_to_reg(right, size);
        let signed = bt::is_signed(ty);

        if let Some(op) = Self::math_op(op, signed, size) {
            self.code.encode(op(lhs, lhs, rhs.0));
            self.free_reg(rhs);
            return if let Ctx::Dest(dest) = ctx {
                self.assign(dest.ty, dest.loc, owned.map_or(Loc::RegRef(lhs), Loc::Reg));
                Some(Value::VOID)
            } else {
                Some(Value::new(ty, owned.map_or(Loc::RegRef(lhs), Loc::Reg)))
            };
        }

        unimplemented!("{:#?}", op)
    }

    fn loc_to_reg_ref(&mut self, loc: &Loc, size: u64) -> (u8, Option<LinReg>) {
        match *loc {
            Loc::RegRef(reg) => (reg, None),
            Loc::Reg(LinReg(reg)) => (reg, None),
            Loc::Deref(LinReg(reg), off) | Loc::DerefRef(reg, off) => {
                let new = self.gpa.allocate();
                self.code.encode(instrs::ld(new.0, reg, off, size as _));
                (new.0, Some(new))
            }
            Loc::Stack(ref stack, off) => {
                let new = self.gpa.allocate();
                self.load_stack(new.0, stack.offset + off, size as _);
                (new.0, Some(new))
            }
            Loc::Imm(imm) => {
                let new = self.gpa.allocate();
                self.code.encode(instrs::li64(new.0, imm));
                (new.0, Some(new))
            }
        }
    }

    fn ensure_sign_extended(&mut self, val: Value, ty: Type) -> Value {
        let size = self.size_of(ty);
        let lsize = self.size_of(val.ty);
        if lsize < size {
            let reg = self.loc_to_reg(val.loc, lsize);
            let op = [instrs::sxt8, instrs::sxt16, instrs::sxt32][lsize.ilog2() as usize];
            self.code.encode(op(reg.0, reg.0));
            Value {
                ty,
                loc: Loc::Reg(reg),
            }
        } else {
            val
        }
    }

    fn assign(&mut self, ty: Type, right: Loc, left: Loc) -> Option<Value> {
        if left == right {
            return Some(Value::VOID);
        }

        let size = self.size_of(ty);

        match size {
            0 => {}
            ..=8 => {
                let lhs = self.loc_to_reg(left, size);
                match right {
                    Loc::RegRef(reg) => self.code.encode(instrs::cp(reg, lhs.0)),
                    Loc::Deref(reg, off) => {
                        self.code.encode(instrs::st(lhs.0, reg.0, off, size as _));
                        self.free_reg(reg);
                    }
                    Loc::DerefRef(reg, off) => {
                        self.code.encode(instrs::st(lhs.0, reg, off, size as _));
                    }
                    Loc::Stack(stack, off) => {
                        self.store_stack(lhs.0, stack.offset + off, size as _);
                        self.sa.free(stack);
                    }
                    l => unimplemented!("{:?}", l),
                }
                self.free_reg(lhs);
            }
            ..=16 if matches!(right, Loc::RegRef(1)) => {
                let (lhs, loff) = left.ref_to_ptr();
                self.code.encode(instrs::st(1, lhs, loff, 16));
                self.free_loc(left);
            }
            ..=u64::MAX => {
                let rhs = self.to_ptr(right);
                let lhs = self.to_ptr(left);
                self.code
                    .encode(instrs::bmc(lhs.0, rhs.0, size.try_into().unwrap()));

                self.free_reg(lhs);
                self.free_reg(rhs);
            }
        }

        Some(Value::VOID)
    }

    fn to_ptr(&mut self, loc: Loc) -> LinReg {
        match loc {
            Loc::Deref(reg, off) => {
                self.code.addi64(reg.0, reg.0, off);
                reg
            }
            Loc::DerefRef(reg, off) => {
                let new = self.gpa.allocate();
                self.code.addi64(new.0, reg, off);
                new
            }
            Loc::Stack(stack, off) => {
                let reg = self.gpa.allocate();
                self.code.addi64(reg.0, STACK_PTR, stack.offset + off);
                self.sa.free(stack);
                reg
            }
            l => unreachable!("{:?}", l),
        }
    }

    fn ensure_owned(&mut self, loc: Loc, ty: Type) -> Loc {
        match loc {
            Loc::RegRef(reg) => {
                let new = self.gpa.allocate();
                self.code.encode(instrs::cp(new.0, reg));
                Loc::Reg(new)
            }
            l => {
                let size = self.size_of(ty);
                let stack = self.sa.alloc(size);
                self.assign(ty, Loc::DerefRef(STACK_PTR, stack.offset), l);
                Loc::Stack(stack, 0)
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
        self.temp.subi64(STACK_PTR, STACK_PTR, self.sa.height);

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
            .encode(instrs::addi64(STACK_PTR, STACK_PTR, self.sa.height));
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
            l @ (Loc::DerefRef(..) | Loc::Deref(..)) => {
                let size = self.size_of(ty);
                let stack = self.sa.alloc(size);
                self.assign(ty, Loc::DerefRef(STACK_PTR, stack.offset), l);
                Loc::Stack(stack, 0)
            }
            l => l,
        }
    }

    fn pass_arg(&mut self, value: &Value, parama: &mut Range<u8>) {
        let size = self.size_of(value.ty);
        let p = parama.next().unwrap() as Reg;

        if size > 16 {
            let (ptr, off) = value.loc.ref_to_ptr();
            self.code.addi64(p, ptr, off);
            return;
        }

        match value.loc {
            Loc::Reg(ref reg) => {
                self.code.encode(instrs::cp(p, reg.0));
            }
            Loc::RegRef(reg) => {
                self.code.encode(instrs::cp(p, reg));
            }
            Loc::Deref(ref reg, off) => {
                self.code.encode(instrs::ld(p, reg.0, off, size as _));
            }
            Loc::DerefRef(reg, off) => {
                self.code.encode(instrs::ld(p, reg, off, size as _));
            }
            Loc::Imm(imm) => {
                self.code.encode(instrs::li64(p, imm));
            }
            Loc::Stack(ref stack, off) => {
                self.load_stack(p, stack.offset + off, size as _);
                self.load_stack(parama.next().unwrap(), stack.offset + off + 8, size as _);
            }
        }
    }

    fn load_arg(&mut self, ty: Type, parama: &mut Range<u8>) -> Loc {
        let size = self.size_of(ty);
        match size {
            0 => Loc::Imm(0),
            ..=8 => {
                let stack = self.sa.alloc(size as _);
                self.store_stack(parama.next().unwrap(), stack.offset, size as _);
                Loc::Stack(stack, 0)
            }
            ..=16 => {
                let stack = self.sa.alloc(size);
                self.store_stack(parama.next().unwrap(), stack.offset, size as _);
                parama.next().unwrap();
                Loc::Stack(stack, 0)
            }
            ..=u64::MAX => {
                let ptr = parama.next().unwrap();
                let stack = self.sa.alloc(size);
                self.assign(
                    ty,
                    Loc::DerefRef(STACK_PTR, stack.offset),
                    Loc::DerefRef(ptr, 0),
                );
                Loc::Stack(stack, 0)
            }
        }
    }

    #[must_use]
    fn assert_ty(&self, pos: parser::Pos, ty: Type, expected: Type) -> Type {
        if let Some(res) = bt::try_upcast(ty, expected) {
            res
        } else {
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

#[derive(Debug)]
pub struct Value {
    ty:  Type,
    loc: Loc,
}

impl Value {
    const VOID: Self = Self {
        ty:  bt::VOID,
        loc: Loc::Imm(0),
    };

    fn new(ty: Type, loc: Loc) -> Self {
        Self { ty, loc }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Loc {
    Reg(LinReg),
    RegRef(Reg),
    Deref(LinReg, u64),
    DerefRef(Reg, u64),
    Imm(u64),
    Stack(LinStack, u64),
}
impl Loc {
    fn take_ref(&self) -> Loc {
        match *self {
            Self::Reg(LinReg(reg)) | Self::RegRef(reg) => Self::RegRef(reg),
            Self::Deref(LinReg(reg), off) | Self::DerefRef(reg, off) => Self::DerefRef(reg, off),
            Self::Stack(ref stack, off) => Self::DerefRef(STACK_PTR, stack.offset + off),
            ref un => unreachable!("{:?}", un),
        }
    }

    fn ref_to_ptr(&self) -> (Reg, u64) {
        match *self {
            Loc::Deref(LinReg(reg), off) => (reg, off),
            Loc::DerefRef(reg, off) => (reg, off),
            Loc::Stack(ref stack, off) => (STACK_PTR, stack.offset + off),
            ref l => panic!("expected stack location, got {:?}", l),
        }
    }

    fn offset_ref(&self, offset: u64) -> Loc {
        self.take_ref().offset(offset)
    }

    fn offset(self, offset: u64) -> Loc {
        match self {
            Loc::Deref(r, off) => Loc::Deref(r, off + offset),
            Loc::DerefRef(r, off) => Loc::DerefRef(r, off + offset),
            Loc::Stack(s, off) => Loc::Stack(s, off + offset),
            l => unreachable!("{:?}", l),
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
        different_types => include_str!("../examples/different_types.hb");
        struct_operators => include_str!("../examples/struct_operators.hb");
    }
}
