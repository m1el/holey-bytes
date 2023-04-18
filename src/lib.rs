pub mod bytecode;
pub mod engine;
use bytecode::ops::*;
use bytecode::types::{CONST_F64, CONST_U8};
use engine::Engine;

pub struct Page {
    pub data: [u8; 4096 * 2],
}

pub fn time() -> u32 {
    9
}
#[derive(Debug)]
pub enum RuntimeErrors {
    InvalidOpcode(u8),
    RegisterTooSmall,
}

// If you solve the halting problem feel free to remove this
pub enum HaltStatus {
    Halted,
    Running,
}

pub struct HandSide {
    signed: bool,
    float: bool,
    num8: Option<u8>,
    num64: Option<u64>,
}
