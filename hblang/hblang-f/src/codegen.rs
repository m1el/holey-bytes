use std::{iter::Cycle, ops::Range, usize};

use crate::{
    lexer::{self, Ty},
    parser::{Exp, Function, Item, Literal, Struct, Type},
};

type Reg = u8;
type Offset = i32;
type Pushed = bool;
type SlotIndex = usize;
type Label = usize;
type Data = usize;
type Size = usize;

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

struct RegAlloc {
    pub regs:        Box<[Option<usize>; 256]>,
    pub used:        Box<[bool; 256]>,
    pub spill_cycle: Cycle<Range<u8>>,
}

impl RegAlloc {
    const STACK_POINTER: Reg = 254;
    const ZERO: Reg = 0;
    const RETURN_ADDRESS: Reg = 31;

    fn alloc_return(&mut self, slot: usize) -> Option<Reg> {
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

    fn alloc_general(&mut self, slot: usize) -> Option<Reg> {
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

    fn spill(&mut self, for_slot: usize) -> (Reg, Option<usize>) {
        let to_spill = self.spill_cycle.next().unwrap();
        let slot = self.spill_specific(to_spill, for_slot);
        (to_spill as Reg + 32, slot)
    }

    fn spill_specific(&mut self, reg: Reg, for_slot: usize) -> Option<usize> {
        self.regs[reg as usize].replace(for_slot)
    }

    fn restore(&mut self, reg: Reg, slot: usize) -> usize {
        self.regs[reg as usize].replace(slot).unwrap()
    }

    fn alloc_specific(&mut self, reg: u8, to: usize) {
        assert!(self.regs[reg as usize].replace(to).is_none());
    }

    fn alloc_specific_in_reg(&mut self, reg: InReg, to: usize) {
        match reg {
            InReg::Single(r) => self.alloc_specific(r, to),
            InReg::Pair(r1, r2) => {
                self.alloc_specific(r1, to);
                self.alloc_specific(r2, to);
            }
        }
    }
}

pub struct ParamAlloc {
    reg_range: Range<Reg>,
    stack:     Offset,
}

impl ParamAlloc {
    fn new() -> Self {
        Self {
            stack:     8, // return adress is in callers stack frame
            reg_range: 2..12,
        }
    }

    fn alloc(&mut self, size: usize) -> SlotValue {
        match self.try_alloc_regs(size) {
            Some(reg) => reg,
            None => {
                let stack = self.stack;
                self.stack += size as Offset;
                SlotValue::Stack(stack)
            }
        }
    }

    fn try_alloc_regs(&mut self, size: usize) -> Option<SlotValue> {
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
                Some(SlotValue::Reg(InReg::Single(reg)))
            }
            2 => {
                let reg = self.reg_range.start;
                self.reg_range.start += 2;
                Some(SlotValue::Reg(InReg::Pair(reg, reg + 1)))
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
    location: usize,
}

#[derive(Clone, Copy)]
struct SlotId {
    // index into slot stack
    index:  SlotIndex,
    // temorary offset carried over when eg. accessing fields
    offset: Offset,
    // this means we can mutate the value as part of computation
    owned:  bool,
}

impl SlotId {
    fn base(location: usize) -> Self {
        Self {
            index:  location,
            offset: 0,
            owned:  true,
        }
    }

    fn borrowed(self) -> Self {
        Self {
            owned: false,
            ..self
        }
    }
}

struct Slot {
    ty:    Type,
    value: SlotValue,
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

#[derive(Clone, Copy)]
enum InReg {
    Single(Reg),
    // if one of the registes is allocated, the other is too, ALWAYS
    // with the same slot
    Pair(Reg, Reg),
}

#[derive(Clone, Copy)]
enum Spill {
    Reg(InReg),
    Stack(Offset), // relative to frame end (rsp if nothing was pushed)
}

#[derive(Clone, Copy)]
enum SlotValue {
    Reg(InReg),
    Stack(Offset), // relative to frame start (rbp)
    Imm(u64),
    Spilled(Spill, SlotIndex),
}

pub struct Value {
    store:  ValueStore,
    offset: Offset,
}

#[derive(Clone, Copy)]
enum ValueStore {
    Reg(InReg),
    Stack(Offset, Pushed),
    Imm(u64),
}

impl From<SlotValue> for ValueStore {
    fn from(value: SlotValue) -> Self {
        match value {
            SlotValue::Reg(reg) => ValueStore::Reg(reg),
            SlotValue::Stack(offset) => ValueStore::Stack(offset, false),
            SlotValue::Imm(imm) => ValueStore::Imm(imm),
            SlotValue::Spilled(spill, _) => match spill {
                Spill::Reg(reg) => ValueStore::Reg(reg),
                Spill::Stack(offset) => ValueStore::Stack(offset, true),
            },
        }
    }
}

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
    BinOp(lexer::Op, Value, Value),
    Move(Size, Value, Value),
    Push(Reg),
    Jump(Label),
    Call(String),
    JumpIfZero(Value, Label),
}

#[derive(Default)]
pub struct Generator<'a> {
    ast: &'a [Item],

    func_labels: Vec<(String, Label)>,

    stack_size:  Offset,
    pushed_size: Offset,

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
            if let SlotValue::Reg(reg) = value {
                self.regs.alloc_specific_in_reg(reg, slot);
            }
            self.add_variable(param.name.clone(), slot);
        }

        for stmt in f.body.iter() {
            assert!(self
                .generate_expr(Some(Type::Builtin(Ty::Void)), stmt)
                .is_none());
        }

        self.pop_frame(frame);
    }

    fn generate_expr(&mut self, expected: Option<Type>, expr: &Exp) -> Option<SlotId> {
        let value = match expr {
            Exp::Literal(lit) => SlotId::base(match lit {
                Literal::Int(i) => self.add_slot(expected.clone().unwrap(), SlotValue::Imm(*i)),
                Literal::Bool(b) => {
                    self.add_slot(Type::Builtin(Ty::Bool), SlotValue::Imm(*b as u64))
                }
            }),
            Exp::Variable(ident) => {
                SlotId::base(self.lookup_variable(ident).unwrap().location).borrowed()
            }
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
            let actual = self.slots[value.index].ty.clone();
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

        self.instrs.push(Instr::Call(name.to_owned()));

        todo!()
    }

    fn set_temporarly(&mut self, from: SlotId, to: SlotValue) {
        let to = self.make_mutable(to, from.index);
        let to_slot = self.add_slot(self.slots[from.index].ty.clone(), to);
        self.emit_move(from, SlotId::base(to_slot));
    }

    fn make_mutable(&mut self, target: SlotValue, by: SlotIndex) -> SlotValue {
        match target {
            SlotValue::Reg(in_reg) => {
                self.regs.alloc_specific_in_reg(in_reg, by);
                target
            }
            SlotValue::Spilled(Spill::Reg(in_reg), slot) => {
                let new_val = SlotValue::Spilled(
                    match in_reg {
                        InReg::Single(reg) => Spill::Stack(self.emmit_push(reg)),
                        InReg::Pair(r1, r2) => {
                            self.emmit_push(r2);
                            Spill::Stack(self.emmit_push(r1))
                        }
                    },
                    slot,
                );
                let new_slot = self.add_slot(self.slots[slot].ty.clone(), new_val);
                SlotValue::Spilled(Spill::Reg(in_reg), new_slot)
            }
            _ => unreachable!(),
        }
    }

    fn emmit_push(&mut self, reg: Reg) -> Offset {
        self.pushed_size += 8;
        self.instrs.push(Instr::Push(reg));
        self.pushed_size
    }

    fn emit_move(&mut self, from: SlotId, to: SlotId) {
        let size = self.size_of(&self.slots[from.index].ty);
        let other_size = self.size_of(&self.slots[to.index].ty);
        assert_eq!(size, other_size);

        self.instrs.push(Instr::Move(
            size,
            self.slot_to_value(from),
            self.slot_to_value(to),
        ));
    }

    fn slot_to_value(&self, slot: SlotId) -> Value {
        let slot_val = &self.slots[slot.index];
        Value {
            store:  slot_val.value.into(),
            offset: slot.offset,
        }
    }

    fn size_of(&self, ty: &Type) -> Size {
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
}

impl<'a> Generator<'a> {
    fn add_variable(&mut self, name: String, location: usize) {
        self.variables.push(Variable { name, location });
    }

    fn add_slot(&mut self, ty: Type, value: SlotValue) -> usize {
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
