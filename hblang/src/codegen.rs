use crate::parser::Type;

struct RegAlloc {
    pub regs: Box<[Option<usize>; 256]>,
}

struct Variable {
    name:     String,
    location: usize,
}

enum Symbol {
    Type(String, Type),
    Func(String, Vec<Type>, Type),
}

struct Slot {
    ty:    Type,
    value: Value,
}

enum Value {
    Reg(u8),
    Stack(i32),
    Imm(u64),
}

type Label = usize;

pub struct Generator {
    regs:        RegAlloc,
    symbols:     Vec<Symbol>,
    variables:   Vec<Variable>,
    slots:       Vec<Slot>,
    relocations: Vec<(Label, usize)>,
}

impl Generator {
    pub fn gen();
}
