#[repr(u8)]
pub enum Operations {
    NOP = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,

    // LOADs a memory address/constant into a register
    LOAD = 5,
    // STOREs a register/constant into a memory address
    STORE = 6,

    EnviromentCall = 10,

    JUMP = 100,
    JumpCond = 101,
    RET = 103,
}

pub enum SubTypes {
    EightBit = 1,
    SixtyFourBit = 2,
    Register8 = 3,
    Register64 = 4,
}
pub enum MathOpSubTypes {
    Unsigned = 0,
    Signed = 1,
    FloatingPoint = 2,
}

pub enum RWSubTypes {
    AddrToReg = 0,
    RegToAddr,
    ConstToReg,
    ConstToAddr,
}

pub enum JumpConditionals {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    GreaterThan = 3,
}
