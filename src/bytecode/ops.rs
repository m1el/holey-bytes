#[repr(u8)]
pub enum Operations {
    NOP = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,

    // READs a register/memory address/constant into a register
    READ = 5,
    // WRITEs a register/memory address/constant into a memory address
    WRITE = 6,

    JUMP = 100,
    JumpEq = 101,
    JumpNeq = 102,
    RET = 103,
}

pub enum MathTypes {
    EightBit = 1,
    SixtyFourBit = 2,
}

pub enum RWTypes {
    RegisterToAddress = 0,
    RegisterToRegister = 1,
    ConstantToAddress = 2,
    ConstantToRegister = 3,
    AddressToRegister = 4,
    AddressToAddress = 5,
}
