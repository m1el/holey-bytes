use std::{iter::Cycle, mem::offset_of, ops::Range};

use crate::{
    lexer::{self, Ty},
    parser::{Exp, Function, Item, Literal, Struct, Type},
};

//| Register   | Description         | Saver  |
//|:-----------|:--------------------|:-------|
//| r0         | Hard-wired zero     | N/A    |
//| r1 - r2    | Return values       | Caller |
//| r2 - r11   | Function parameters | Caller |
//| r12 - r30  | General purpose     | Caller |
//| r31        | Return address      | Caller |
//| r32 - r253 | General purpose     | Callee |
//| r254       | Stack pointer       | Callee |
//| r255       | Thread pointer      | N/A    |

const STACK_POINTER: Reg = 254;

struct RegAlloc {
    pub regs:        Box<[Option<usize>; 256]>,
    pub used:        Box<[bool; 256]>,
    pub spill_cycle: Cycle<Range<u8>>,
}

impl RegAlloc {
    fn alloc_return(&mut self, slot: SlotId) -> Option<Reg> {
        self.regs[1..2]
            .iter_mut()
            .position(|reg| {
                if reg.is_none() {
                    *reg = Some(slot);
                    true
                } else {
                    false
                }
            })
            .map(|reg| reg as Reg + 1)
    }

    fn alloc_general(&mut self, slot: SlotId) -> Option<Reg> {
        self.regs[32..254]
            .iter_mut()
            .zip(&mut self.used[32..254])
            .position(|(reg, used)| {
                if reg.is_none() {
                    *reg = Some(slot);
                    *used = true;
                    true
                } else {
                    false
                }
            })
            .map(|reg| reg as Reg + 32)
    }

    fn free(&mut self, reg: Reg) {
        assert!(self.regs[reg as usize].take().is_some());
    }

    fn is_used(&self, reg: Reg) -> bool {
        self.regs[reg as usize].is_some()
    }

    fn spill(&mut self, for_slot: SlotId) -> (Reg, Option<SlotId>) {
        let to_spill = self.spill_cycle.next().unwrap();
        let slot = self.spill_specific(to_spill, for_slot);
        (to_spill as Reg + 32, slot)
    }

    fn spill_specific(&mut self, reg: Reg, for_slot: SlotId) -> Option<SlotId> {
        self.regs[reg as usize].replace(for_slot)
    }

    fn restore(&mut self, reg: Reg, slot: SlotId) -> SlotId {
        self.regs[reg as usize].replace(slot).unwrap()
    }

    fn alloc_specific(&mut self, reg: u8, to: usize) {
        assert!(self.regs[reg as usize].replace(to).is_none());
    }
}

pub struct ParamAlloc {
    reg_range: Range<Reg>,
    stack:     Offset,
}

impl ParamAlloc {
    fn new() -> Self {
        Self {
            stack:     16,
            reg_range: 2..12,
        }
    }

    fn alloc(&mut self, size: usize) -> Value {
        match self.try_alloc_regs(size) {
            Some(reg) => reg,
            None => panic!("Too many arguments o7"),
        }
    }

    fn try_alloc_regs(&mut self, size: usize) -> Option<Value> {
        let mut needed = size.div_ceil(8);
        if needed > 2 {
            needed = 1; // passed by ref
        }

        if self.reg_range.len() < needed {
            return None;
        }

        match needed {
            1 => {
                let reg = self.reg_range.start;
                self.reg_range.start += 1;
                Some(Value::Reg(reg, None, 0))
            }
            2 => {
                let reg = self.reg_range.start;
                self.reg_range.start += 2;
                Some(Value::Reg(reg, Some(reg + 1), 0))
            }
            _ => unreachable!(),
        }
    }
}

impl Default for RegAlloc {
    fn default() -> Self {
        Self {
            regs:        Box::new([None; 256]),
            used:        Box::new([false; 256]),
            spill_cycle: (32..254).cycle(),
        }
    }
}

struct Variable {
    name:     String,
    location: SlotId,
}

type SlotId = usize;

struct Slot {
    ty:    Type,
    value: Value,
}

#[repr(transparent)]
struct InstBuffer {
    buffer: Vec<u8>,
}

impl InstBuffer {
    fn new(vec: &mut Vec<u8>) -> &mut Self {
        unsafe { &mut *(vec as *mut Vec<u8> as *mut Self) }
    }
}

impl hbbytecode::Buffer for InstBuffer {
    fn reserve(&mut self, bytes: usize) {
        self.buffer.reserve(bytes);
    }

    unsafe fn write(&mut self, byte: u8) {
        self.buffer.push(byte);
    }
}

type Reg = u8;
type Offset = i32;
type Pushed = bool;

#[derive(Clone, Copy)]
enum Value {
    Reg(Reg, Option<Reg>, Offset),
    Stack(Offset, Pushed),
    Imm(u64),
    Spilled(Reg, SlotId, Option<Reg>, Option<SlotId>, Offset),
    DoubleSpilled(SlotId, Offset, Option<SlotId>),
}

#[derive(Clone, Copy)]
enum NormalizedValue {
    Reg(Reg, Option<Reg>, Offset),
    Stack(Offset, Pushed),
    Imm(u64),
}

impl From<Value> for NormalizedValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Reg(reg, secondary, offset) => NormalizedValue::Reg(reg, secondary, offset),
            Value::Stack(offset, pushed) => NormalizedValue::Stack(offset, pushed),
            Value::Imm(imm) => NormalizedValue::Imm(imm),
            Value::Spilled(reg, _, secondary, _, offset) => {
                NormalizedValue::Reg(reg, secondary, offset)
            }
            Value::DoubleSpilled(_, offset, _) => NormalizedValue::Stack(offset, false),
        }
    }
}

type Label = usize;
type Data = usize;

pub struct LabelReloc {
    pub label:  Label,
    pub offset: usize,
}

pub struct DataReloc {
    pub data:   Data,
    pub offset: usize,
}

#[must_use]
pub struct Frame {
    pub slot_count: usize,
    pub var_count:  usize,
}

enum Instr {
    BinOp(lexer::Op, NormalizedValue, NormalizedValue),
    Move(usize, NormalizedValue, NormalizedValue),
    Jump(Label),
    JumpIfZero(NormalizedValue, Label),
}

#[derive(Default)]
pub struct Generator<'a> {
    ast: &'a [Item],

    func_labels: Vec<(String, Label)>,

    stack_size:  usize,
    pushed_size: usize,

    regs:      RegAlloc,
    variables: Vec<Variable>,
    slots:     Vec<Slot>,

    labels:       Vec<Option<usize>>,
    label_relocs: Vec<LabelReloc>,

    data:        Vec<Option<usize>>,
    data_relocs: Vec<DataReloc>,

    code_section: Vec<u8>,
    data_section: Vec<u8>,

    instrs: Vec<Instr>,
}

impl<'a> Generator<'a> {
    fn generate(mut self) -> Vec<u8> {
        for item in self.ast {
            let Item::Function(f) = item else { continue };
            self.generate_function(f);
        }

        self.link()
    }

    fn generate_function(&mut self, f: &Function) {
        let frame = self.push_frame();

        let mut param_alloc = ParamAlloc::new();

        for param in f.args.iter() {
            let param_size = self.size_of(&param.ty);
            let value = param_alloc.alloc(param_size);
            let slot = self.add_slot(param.ty.clone(), value);
            self.allocate_value_regs(value, slot);
            self.add_variable(param.name.clone(), slot);
        }

        for stmt in f.body.iter() {
            assert!(self
                .generate_expr(Some(Type::Builtin(Ty::Void)), stmt)
                .is_none());
        }

        self.pop_frame(frame);
    }

    fn allocate_value_regs(&mut self, value: Value, to: SlotId) {
        if let Value::Reg(primary, secondary, _) = value {
            self.regs.alloc_specific(primary, to);
            if let Some(secondary) = secondary {
                self.regs.alloc_specific(secondary, to);
            }
        }
    }

    fn generate_expr(&mut self, expected: Option<Type>, expr: &Exp) -> Option<SlotId> {
        let value = match expr {
            Exp::Literal(lit) => match lit {
                Literal::Int(i) => self.add_slot(expected.clone().unwrap(), Value::Imm(*i)),
                Literal::Bool(b) => self.add_slot(Type::Builtin(Ty::Bool), Value::Imm(*b as u64)),
            },
            Exp::Variable(ident) => self.lookup_variable(ident).unwrap().location,
            Exp::Call { name, args } => self.generate_call(expected.clone(), name, args),
            Exp::Ctor { name, fields } => todo!(),
            Exp::Index { base, index } => todo!(),
            Exp::Field { base, field } => todo!(),
            Exp::Unary { op, exp } => todo!(),
            Exp::Binary { op, left, right } => todo!(),
            Exp::If { cond, then, else_ } => todo!(),
            Exp::Let { name, ty, value } => todo!(),
            Exp::For {
                init,
                cond,
                step,
                block,
            } => todo!(),
            Exp::Block(_) => todo!(),
            Exp::Return(_) => todo!(),
            Exp::Break => todo!(),
            Exp::Continue => todo!(),
        };

        if let Some(expected) = expected {
            let actual = self.slots[value].ty.clone();
            assert_eq!(expected, actual);
        }

        Some(value)
    }

    fn generate_call(&mut self, expected: Option<Type>, name: &str, args: &[Exp]) -> SlotId {
        let frame = self.push_frame();
        let func = self.lookup_function(name);

        let mut arg_alloc = ParamAlloc::new();
        for (arg, param) in args.iter().zip(&func.args) {
            let arg_slot = self.generate_expr(Some(param.ty.clone()), arg).unwrap();
            let arg_size = self.size_of(&param.ty);
            let param_slot = arg_alloc.alloc(arg_size);
            self.set_temporarly(arg_slot, param_slot);
        }

        todo!()
    }

    fn set_temporarly(&mut self, from: SlotId, to: Value) {
        match to {
            Value::Reg(dst, secondary, offset) => {
                let other_slot = secondary.and_then(|s| self.regs.spill_specific(s, usize::MAX));
                if let Some(slot) = self.regs.spill_specific(dst, usize::MAX) {
                    self.slots[slot].value =
                        Value::Spilled(dst, slot, secondary, other_slot, offset);
                } else if let Some(slot) = other_slot {
                    self.slots[slot].value =
                        Value::Spilled(secondary.unwrap(), slot, Some(dst), None, offset);
                }
            }
            _ => unreachable!(),
        };

        let size = self.size_of(&self.slots[from].ty);
        self.emit_move(size, self.slots[from].value, to);
    }

    fn emit_move(&mut self, size: usize, from: Value, to: Value) {
        self.instrs.push(Instr::Move(size, from.into(), to.into()));
    }

    fn size_of(&self, ty: &Type) -> usize {
        match ty {
            Type::Builtin(ty) => match ty {
                Ty::U8 | Ty::I8 | Ty::Bool => 1,
                Ty::U16 | Ty::I16 => 2,
                Ty::U32 | Ty::I32 => 4,
                Ty::U64 | Ty::I64 => 8,
                Ty::Void => 0,
            },
            Type::Struct(name) => self
                .lookup_struct(name)
                .fields
                .iter()
                .map(|field| self.size_of(&field.ty))
                .sum(),
            Type::Pinter(_) => 8,
        }
    }

    fn add_variable(&mut self, name: String, location: SlotId) {
        self.variables.push(Variable { name, location });
    }

    fn add_slot(&mut self, ty: Type, value: Value) -> SlotId {
        let slot = self.slots.len();
        self.slots.push(Slot { ty, value });
        slot
    }

    fn link(mut self) -> Vec<u8> {
        for reloc in self.label_relocs {
            let label = self.labels[reloc.label].unwrap();
            let offset = reloc.offset;
            let target = label - offset;
            let target_bytes = u64::to_le_bytes(target as u64);
            self.code_section[offset..offset + 8].copy_from_slice(&target_bytes);
        }

        for reloc in self.data_relocs {
            let data = self.data[reloc.data].unwrap();
            let offset = reloc.offset;
            let target = data;
            let target_bytes = u64::to_le_bytes((target + self.code_section.len()) as u64);
            self.data_section[offset..offset + 8].copy_from_slice(&target_bytes);
        }

        self.code_section.extend_from_slice(&self.data_section);
        self.code_section
    }

    fn lookup_func_label(&mut self, name: &str) -> Label {
        if let Some(label) = self.func_labels.iter().find(|(n, _)| n == name) {
            return label.1;
        }

        panic!("Function not found: {}", name);
    }

    fn declare_label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn define_label(&mut self, label: Label) {
        self.labels[label] = Some(self.code_section.len());
    }

    fn declare_data(&mut self) -> Data {
        self.data.push(None);
        self.data.len() - 1
    }

    fn define_data(&mut self, data: Data, bytes: &[u8]) {
        self.data[data] = Some(self.data.len());
        self.data_section.extend_from_slice(bytes);
    }

    fn lookup_struct(&self, name: &str) -> &Struct {
        self.lookup_item(name)
            .map(|item| match item {
                Item::Struct(s) => s,
                _ => panic!("Not a struct: {}", name),
            })
            .expect("Struct not found")
    }

    fn lookup_function(&self, name: &str) -> &'a Function {
        self.lookup_item(name)
            .map(|item| match item {
                Item::Function(f) => f,
                _ => panic!("Not a function: {}", name),
            })
            .expect("Function not found")
    }

    fn lookup_item(&self, name: &str) -> Option<&'a Item> {
        self.ast.iter().find(|item| match item {
            Item::Import(_) => false,
            Item::Struct(s) => s.name == name,
            Item::Function(f) => f.name == name,
        })
    }

    fn lookup_variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|variable| variable.name == name)
    }

    fn push_frame(&mut self) -> Frame {
        Frame {
            slot_count: self.slots.len(),
            var_count:  self.variables.len(),
        }
    }

    fn pop_frame(&mut self, frame: Frame) {
        self.slots.truncate(frame.slot_count);
        self.variables.truncate(frame.var_count);
    }
}

pub fn generate(ast: &[Item]) -> Vec<u8> {
    Generator {
        ast,
        ..Default::default()
    }
    .generate()
}
