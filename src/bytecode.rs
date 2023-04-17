pub const NOP: u8 = 0;
pub const ADD: u8 = 1;
pub const SUB: u8 = 2;
pub const MML: u8 = 3;
pub const DIV: u8 = 4;

pub const CONST_U8: u8 = 0x00;
pub const CONST_I8: i8 = 0x01;

pub const CONST_U16: u8 = 0x02;
pub const CONST_I16: u8 = 0x03;

pub const CONST_U32: u8 = 0x04;
pub const CONST_I32: u8 = 0x05;
pub const CONST_F32: u8 = 0x06;

pub const CONST_U64: u8 = 0x07;
pub const CONST_I64: u8 = 0x08;
pub const CONST_F64: u8 = 0x09;

pub const ADDRESS: u8 = 0x10;

pub const RegisterA8: u8 = 0xA0;
pub const RegisterB8: u8 = 0xB0;
pub const RegisterC8: u8 = 0xC0;
pub const RegisterD8: u8 = 0xD0;
pub const RegisterE8: u8 = 0xE0;
pub const RegisterF8: u8 = 0xF0;

pub const RegisterA16: u8 = 0xA1;
pub const RegisterB16: u8 = 0xB1;
pub const RegisterC16: u8 = 0xC1;
pub const RegisterD16: u8 = 0xD1;
pub const RegisterE16: u8 = 0xE1;
pub const RegisterF16: u8 = 0xF1;

pub const RegisterA32: u8 = 0xA2;
pub const RegisterB32: u8 = 0xB2;
pub const RegisterC32: u8 = 0xC2;
pub const RegisterD32: u8 = 0xD2;
pub const RegisterE32: u8 = 0xE2;
pub const RegisterF32: u8 = 0xF2;

pub const RegisterA64: u8 = 0xA3;
pub const RegisterB64: u8 = 0xB3;
pub const RegisterC64: u8 = 0xC3;
pub const RegisterD64: u8 = 0xD3;
pub const RegisterE64: u8 = 0xE3;
pub const RegisterF64: u8 = 0xF3;
