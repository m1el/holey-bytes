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

pub mod bt {
    use super::*;

    const fn builtin_type(id: u32) -> Type {
        Type::MAX - id
    }

    macro_rules! builtin_type {
        ($($name:ident;)*) => {$(
            pub const $name: Type = TypeKind::Builtin(${index(0)}).encode();
        )*};
    }

    builtin_type! {
        VOID;
        UNREACHABLE;
        INT;
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
        self.pop(RET_ADDR, 8);
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
    free: Vec<Reg>,
    // TODO:use 256 bit mask instead
    used: Vec<Reg>,
}

impl RegAlloc {
    fn init_callee(&mut self) {
        self.clear();
        self.free.extend(32..=253);
    }

    fn clear(&mut self) {
        self.free.clear();
        self.used.clear();
    }

    fn allocate(&mut self) -> LinReg {
        let reg = self.free.pop().expect("TODO: we need to spill");
        if self.used.binary_search_by_key(&!reg, |&r| !r).is_err() {
            self.used.push(reg);
        }
        LinReg(reg)
    }

    fn free(&mut self, reg: LinReg) {
        self.free.push(reg.0);
        std::mem::forget(reg);
    }
}

struct FnLabel {
    offset: u32,
    // TODO: use different stile of identifier that does not allocate, eg. index + length into a
    // file
    name:   Ident,
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
    id:     Ident,
    fields: Rc<[(Rc<str>, Type)]>,
}

pub struct Codegen<'a> {
    path: &'a std::path::Path,
    ret: Type,
    gpa: RegAlloc,
    code: Func,
    temp: Func,
    labels: Vec<FnLabel>,
    stack_size: u64,
    vars: Vec<Variable>,
    stack_relocs: Vec<StackReloc>,
    ret_relocs: Vec<RetReloc>,
    loops: Vec<Loop>,
    records: Vec<Struct>,
    pointers: Vec<Type>,
    main: Option<LabelId>,
}

impl<'a> Codegen<'a> {
    pub fn new() -> Self {
        Self {
            path: std::path::Path::new(""),
            ret: bt::VOID,
            gpa: Default::default(),
            code: Default::default(),
            temp: Default::default(),
            labels: Default::default(),
            stack_size: 0,
            vars: Default::default(),
            stack_relocs: Default::default(),
            ret_relocs: Default::default(),
            loops: Default::default(),
            records: Default::default(),
            pointers: Default::default(),
            main: None,
        }
    }

    pub fn file(
        &mut self,
        path: &'a std::path::Path,
        exprs: &'a [parser::Expr<'a>],
    ) -> std::fmt::Result {
        self.path = path;
        for expr in exprs {
            self.expr(expr, None);
        }
        Ok(())
    }

    fn size_of(&self, ty: Type) -> u64 {
        match ty {
            bt::INT => 8,
            bt::BOOL => 1,
            _ => match TypeKind::from_ty(ty) {
                TypeKind::Pointer(_) => 8,
                TypeKind::Builtin(e) => unreachable!("{:?}", e),
                TypeKind::Struct(ty) => self.records[ty as usize]
                    .fields
                    .iter()
                    .map(|(_, ty)| self.size_of(*ty))
                    .sum(),
            },
        }
    }

    fn loc_to_reg(&mut self, loc: Loc) -> LinReg {
        match loc {
            Loc::RegRef(rr) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::cp(reg.0, rr));
                reg
            }
            Loc::Reg(reg) => reg,
            Loc::Deref(dreg) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::ld(reg.0, dreg.0, 0, 8));
                self.gpa.free(dreg);
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
        self.stack_size += size as u64;
        offset
    }

    fn store_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.stack_relocs.push(StackReloc {
            offset: self.code.code.len() as u32 + 3,
            size,
        });
        self.code.encode(instrs::st(reg, STACK_PTR, offset, size));
    }

    fn load_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.stack_relocs.push(StackReloc {
            offset: self.code.code.len() as u32 + 3,
            size,
        });
        self.code.encode(instrs::ld(reg, STACK_PTR, offset, size));
    }

    fn reloc_stack(&mut self) {
        for reloc in self.stack_relocs.drain(..) {
            let dest = &mut self.code.code[reloc.offset as usize..][..reloc.size as usize];
            let value = u64::from_ne_bytes(dest.try_into().unwrap());
            let offset = self.stack_size - value;
            dest.copy_from_slice(&offset.to_ne_bytes());
        }
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
            E::Ident { id, .. } => {
                let index = self
                    .records
                    .iter()
                    .position(|r| r.id == id)
                    .unwrap_or_else(|| {
                        panic!("type not found: {:?}", id);
                    });
                TypeKind::Struct(index as Type).encode()
            }
            expr => unimplemented!("type: {:#?}", expr),
        }
    }

    fn expr(&mut self, expr: &'a parser::Expr<'a>, expeted: Option<Type>) -> Option<Value> {
        match *expr {
            E::Ctor { ty, fields } => {
                let ty = self.ty(&ty);
                let TypeKind::Struct(idx) = TypeKind::from_ty(ty) else {
                    panic!("expected struct, got {:?}", ty);
                };
                let size = self.size_of(ty);
                let stack = self.alloc_stack(size as u32);
                let mut field_values = fields
                    .iter()
                    .map(|(name, field)| (*name, self.expr(field, None).unwrap()))
                    .collect::<Vec<_>>();
                let decl_fields = self.records[idx as usize].fields.clone();
                let mut offset = 0;
                for (name, ty) in decl_fields.as_ref() {
                    let index = field_values
                        .iter()
                        .position(|(n, _)| *n == name.as_ref())
                        .unwrap();
                    let (_, value) = field_values.remove(index);
                    if value.ty != *ty {
                        panic!("expected {:?}, got {:?}", ty, value.ty);
                    }
                    let reg = self.loc_to_reg(value.loc);
                    self.store_stack(reg.0, stack + offset, 8);
                    self.gpa.free(reg);
                    offset += 8;
                }
                Some(Value {
                    ty,
                    loc: Loc::Stack(stack),
                })
            }
            E::Field { target, field } => {
                let target = self.expr(target, None).unwrap();
                let TypeKind::Struct(idx) = TypeKind::from_ty(target.ty) else {
                    panic!("expected struct, got {:?}", target.ty);
                };
                let decl_fields = self.records[idx as usize].fields.clone();
                let index = decl_fields
                    .iter()
                    .position(|(name, _)| name.as_ref() == field)
                    .unwrap();
                let size = self.size_of(decl_fields[index].1);
                assert_eq!(size, 8, "TODO: implement other sizes");
                let value = match target.loc {
                    Loc::Reg(_) => todo!(),
                    Loc::RegRef(_) => todo!(),
                    Loc::Deref(r) => {
                        self.code.encode(instrs::addi64(r.0, r.0, index as u64 * 8));
                        Loc::Deref(r)
                    }
                    Loc::Imm(_) => todo!(),
                    Loc::Stack(stack) => Loc::Stack(stack + index as u64 * 8),
                    Loc::StackRef(stack) => Loc::StackRef(stack + index as u64 * 8),
                };
                Some(Value {
                    ty:  decl_fields[index].1,
                    loc: value,
                })
            }
            E::BinOp {
                left: E::Ident { id, .. },
                op: T::Decl,
                right: E::Struct { fields, .. },
            } => {
                let fields = fields
                    .iter()
                    .map(|&(name, ty)| (name.into(), self.ty(&ty)))
                    .collect();
                self.records.push(Struct { id: *id, fields });
                None
            }
            E::UnOp {
                op: T::Amp, val, ..
            } => {
                let val = self.expr(val, None).unwrap();
                match val.loc {
                    Loc::StackRef(off) => {
                        let reg = self.gpa.allocate();
                        self.stack_relocs.push(StackReloc {
                            offset: self.code.code.len() as u32 + 3,
                            size:   8,
                        });
                        self.code.encode(instrs::addi64(reg.0, STACK_PTR, off));
                        Some(Value {
                            ty:  self.alloc_pointer(val.ty),
                            loc: Loc::Reg(reg),
                        })
                    }
                    l => panic!("cant take pointer of {:?}", l),
                }
            }
            E::UnOp {
                op: T::Star, val, ..
            } => {
                let val = self.expr(val, None).unwrap();
                let reg = self.loc_to_reg(val.loc);
                match TypeKind::from_ty(val.ty) {
                    TypeKind::Pointer(ty) => Some(Value {
                        ty:  self.pointers[ty as usize],
                        loc: Loc::Deref(reg),
                    }),
                    _ => panic!("cant deref {:?}", val.ty),
                }
            }
            E::BinOp {
                left: E::Ident { name, id, .. },
                op: T::Decl,
                right: E::Closure {
                    ret, body, args, ..
                },
            } => {
                log::dbg!("fn: {}", name);
                let frame = self.add_label(*id);
                if *name == "main" {
                    self.main = Some(frame.label);
                }
                log::dbg!("fn-args");
                for (i, arg) in args.iter().enumerate() {
                    let offset = self.alloc_stack(8);
                    let ty = self.ty(&arg.ty);
                    self.vars.push(Variable {
                        id:    arg.id,
                        value: Value {
                            ty,
                            loc: Loc::Stack(offset),
                        },
                    });
                    self.store_stack(i as Reg + 2, offset, 8);
                }
                self.gpa.init_callee();
                self.ret = self.ty(ret);
                log::dbg!("fn-body");
                self.expr(body, None);
                self.vars.clear();
                log::dbg!("fn-relocs");
                self.reloc_stack();
                log::dbg!("fn-prelude");
                self.write_fn_prelude(frame);
                log::dbg!("fn-ret");
                self.reloc_rets();
                self.ret();
                self.stack_size = 0;
                self.vars.clear();
                None
            }
            E::BinOp {
                left: E::Ident { id, .. },
                op: T::Decl,
                right,
            } => {
                let val = self.expr(right, None).unwrap();
                let loc = self.make_loc_owned(val.loc, val.ty);
                let loc = self.ensure_spilled(loc);
                self.vars.push(Variable {
                    id:    *id,
                    value: Value { ty: val.ty, loc },
                });
                None
            }
            E::Call {
                func: E::Ident { id, .. },
                args,
            } => {
                for (i, arg) in args.iter().enumerate() {
                    let arg = self.expr(arg, None).unwrap();
                    let reg = self.loc_to_reg(arg.loc);
                    self.code.encode(instrs::cp(i as Reg + 2, reg.0));
                    self.gpa.free(reg);
                }
                let func = self.get_or_reserve_label(*id);
                self.code.call(func);
                let reg = self.gpa.allocate();
                self.code.encode(instrs::cp(reg.0, 1));
                Some(Value {
                    ty:  self.ret,
                    loc: Loc::Reg(reg),
                })
            }
            E::Ident { name, id, .. } => {
                let var = self
                    .vars
                    .iter()
                    .find(|v| v.id == id)
                    .unwrap_or_else(|| panic!("variable not found: {:?}", name));
                Some(Value {
                    ty:  var.value.ty,
                    loc: var.value.loc.take_ref(),
                })
            }
            E::Return { val, .. } => {
                if let Some(val) = val {
                    let val = self.expr(val, Some(self.ret)).unwrap();
                    if val.ty != self.ret {
                        panic!("expected {:?}, got {:?}", self.ret, val.ty);
                    }
                    self.assign(
                        Value {
                            ty:  self.ret,
                            loc: Loc::RegRef(1),
                        },
                        val,
                    );
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
                    if let Some(Loc::Reg(reg)) = self.expr(stmt, None).map(|v| v.loc) {
                        self.gpa.free(reg);
                    }
                }
                None
            }
            E::Number { value, .. } => Some(Value {
                ty:  expeted.unwrap_or(bt::INT),
                loc: Loc::Imm(value),
            }),
            E::If {
                cond, then, else_, ..
            } => {
                log::dbg!("if-cond");
                let cond = self.expr(cond, Some(bt::BOOL)).unwrap();
                let reg = self.loc_to_reg(cond.loc);
                let jump_offset = self.code.code.len() as u32;
                self.code.encode(instrs::jeq(reg.0, 0, 0));
                self.gpa.free(reg);

                log::dbg!("if-then");
                self.expr(then, None);

                let jump;

                if let Some(else_) = else_ {
                    log::dbg!("if-else");
                    let else_jump_offset = self.code.code.len() as u32;
                    self.code.encode(instrs::jmp(0));

                    jump = self.code.code.len() as i16 - jump_offset as i16;

                    self.expr(else_, None);

                    let jump = self.code.code.len() as i32 - else_jump_offset as i32;
                    log::dbg!("if-else-jump: {}", jump);
                    self.code.code[else_jump_offset as usize + 1..][..4]
                        .copy_from_slice(&jump.to_ne_bytes());
                } else {
                    jump = self.code.code.len() as i16 - jump_offset as i16;
                }

                log::dbg!("if-then-jump: {}", jump);
                self.code.code[jump_offset as usize + 3..][..2]
                    .copy_from_slice(&jump.to_ne_bytes());

                None
            }
            E::Loop { body, .. } => {
                log::dbg!("loop");
                let loop_start = self.code.code.len() as u32;
                self.loops.push(Loop {
                    offset: loop_start,
                    relocs: Default::default(),
                });
                self.expr(body, None);

                log::dbg!("loop-end");
                let loop_end = self.code.code.len();
                self.code
                    .encode(instrs::jmp(loop_start as i32 - loop_end as i32));
                let loop_end = self.code.code.len() as u32;

                let loop_ = self.loops.pop().unwrap();
                for reloc in loop_.relocs {
                    let dest = &mut self.code.code
                        [reloc.offset as usize + reloc.instr_offset as usize..]
                        [..reloc.size as usize];
                    let offset = loop_end as i32 - reloc.offset as i32;
                    dest.copy_from_slice(&offset.to_ne_bytes());
                }

                None
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
                let left = self.expr(left, expeted).unwrap();
                let right = self.expr(right, Some(left.ty)).unwrap();
                if op == T::Assign {
                    return self.assign(left, right);
                }

                let lhs = self.loc_to_reg(left.loc);
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
        }
    }

    fn assign(&mut self, left: Value, right: Value) -> Option<Value> {
        let rhs = self.loc_to_reg(right.loc);
        match left.loc {
            Loc::Deref(reg) => {
                self.code.encode(instrs::st(rhs.0, reg.0, 0, 8));
                self.gpa.free(reg);
            }
            Loc::RegRef(reg) => self.code.encode(instrs::cp(reg, rhs.0)),
            Loc::StackRef(offset) => self.store_stack(rhs.0, offset, 8),
            _ => unimplemented!(),
        }
        self.gpa.free(rhs);
        None
    }

    fn get_or_reserve_label(&mut self, name: Ident) -> LabelId {
        if let Some(label) = self.labels.iter().position(|l| l.name == name) {
            label as u32
        } else {
            self.labels.push(FnLabel { offset: 0, name });
            self.labels.len() as u32 - 1
        }
    }

    fn add_label(&mut self, name: Ident) -> Frame {
        let offset = self.code.code.len() as u32;
        let label = if let Some(label) = self.labels.iter().position(|l| l.name == name) {
            self.labels[label].offset = offset;
            label as u32
        } else {
            self.labels.push(FnLabel {
                offset,
                name: name.into(),
            });
            self.labels.len() as u32 - 1
        };

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
        self.temp.push(RET_ADDR, 8);
        for &reg in self.gpa.used.clone().iter() {
            self.temp.push(reg, 8);
        }
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
        for reg in self.gpa.used.clone().iter().rev() {
            self.code.pop(*reg, 8);
        }
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
            Loc::StackRef(mut off) => {
                let size = self.size_of(ty);
                assert!(size % 8 == 0, "TODO: implement other sizes");
                let stack = self.alloc_stack(size as u32);
                let reg = self.gpa.allocate();
                while size > 0 {
                    self.load_stack(reg.0, off, 8);
                    self.store_stack(reg.0, stack, 8);
                    off += 8;
                }
                self.gpa.free(reg);
                Loc::Stack(stack)
            }
            l => l,
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
}

pub struct Value {
    ty:  Type,
    loc: Loc,
}

#[derive(Debug)]
enum Loc {
    Reg(LinReg),
    RegRef(Reg),
    Deref(LinReg),
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
                core::slice::from_raw_parts(target, count)
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
            log::dbg!("write: {:x} {}", addr.get(), count);
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
        let path = std::path::Path::new("test");
        let arena = crate::parser::Arena::default();
        let mut parser = super::parser::Parser::new(input, path, &arena);
        let exprs = parser.file();
        let mut codegen = super::Codegen::new();
        codegen.file(path, &exprs).unwrap();
        let mut out = Vec::new();
        codegen.dump(&mut out).unwrap();

        std::fs::write("test.bin", &out).unwrap();
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
