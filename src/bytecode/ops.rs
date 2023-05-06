#[repr(u8)]
pub enum Operations {
    NOP = 0,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    MOD = 5,

    AND = 6,
    OR = 7,
    XOR = 8,
    NOT = 9,

    // LOADs a memory address/constant into a register
    LOAD = 15,
    // STOREs a register/constant into a memory address
    STORE = 16,

    MapPage = 17,
    UnmapPage = 18,

    // SHIFT LEFT 16 A0
    Shift = 20,

    JUMP = 100,
    JumpCond = 101,
    RET = 103,

    EnviromentCall = 255,
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
    LessThanOrEqualTo = 3,
    GreaterThan = 4,
    GreaterThanOrEqualTo = 5,
}
