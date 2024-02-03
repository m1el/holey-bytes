use std::{iter::Cycle, ops::Range};

use crate::{
    lexer::Ty,
    parser::{Exp, Function, Item, Literal, Struct, Type},
    typechk::Type,
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

struct RegAlloc {
    pub regs:        Box<[Option<usize>; 256]>,
    pub used:        Box<[bool; 256]>,
    pub spill_cycle: Cycle<Range<usize>>,
}

impl RegAlloc {
    fn alloc_regurn(&mut self, slot: SlotId) -> Option<Reg> {
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

    fn spill(&mut self, for_slot: SlotId) -> (Reg, SlotId) {
        let to_spill = self.spill_cycle.next().unwrap();
        let slot = self.regs[to_spill].replace(for_slot).unwrap();
        (to_spill as Reg + 32, slot)
    }

    fn restore(&mut self, reg: Reg, slot: SlotId) -> SlotId {
        self.regs[reg as usize].replace(slot).unwrap()
    }
}

pub struct ParamAlloc {
    reg_range: Range<Reg>,
    stack:     Offset,
}

impl ParamAlloc {
    fn new(reg_range: Range<Reg>) -> Self {
        Self {
            stack: 16,
            reg_range,
        }
    }

    fn alloc(&mut self, mut size: usize) -> Value {
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
                Some(Value::Reg(reg))
            }
            2 => {
                let reg = self.reg_range.start;
                self.reg_range.start += 2;
                Some(Value::Pair(reg, reg + 1))
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

enum Value {
    Pair(Reg, Reg),
    Reg(Reg),
    Stack(Offset),
    Imm(u64),
    Spilled(Reg, SlotId),
    DoubleSpilled(SlotId, Offset),
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

#[derive(Default)]
pub struct Generator<'a> {
    ast: &'a [Item],

    func_labels: Vec<(String, Label)>,

    regs:      RegAlloc,
    variables: Vec<Variable>,
    slots:     Vec<Slot>,

    labels:       Vec<Option<usize>>,
    label_relocs: Vec<LabelReloc>,

    data:        Vec<Option<usize>>,
    data_relocs: Vec<DataReloc>,

    code_section: Vec<u8>,
    data_section: Vec<u8>,
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

        let mut param_alloc = ParamAlloc::new(2..12);

        for param in f.args.iter() {
            let param_size = self.size_of(&param.ty);
            let slot = self.add_slot(param.ty.clone(), param_alloc.alloc(param_size));
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
            Exp::Literal(lit) => match lit {
                Literal::Int(i) => self.add_slot(expected.unwrap(), Value::Imm(*i)),
                Literal::Bool(b) => self.add_slot(Type::Builtin(Ty::Bool), Value::Imm(*b as u64)),
            },
            Exp::Variable(ident) => self.lookup_variable(ident).unwrap().location,
            Exp::Call { name, args } => todo!(),
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
            .and_then(|item| match item {
                Item::Struct(s) => Some(s),
                _ => panic!("Not a struct: {}", name),
            })
            .expect("Struct not found")
    }

    fn lookup_function(&self, name: &str) -> &Function {
        self.lookup_item(name)
            .and_then(|item| match item {
                Item::Function(f) => Some(f),
                _ => panic!("Not a function: {}", name),
            })
            .expect("Function not found")
    }

    fn lookup_item(&self, name: &str) -> Option<&Item> {
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
