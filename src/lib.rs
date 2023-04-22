// #![no_std]
extern crate alloc;

pub mod bytecode;
pub mod engine;
pub mod memory;

use bytecode::ops::*;
use bytecode::types::{CONST_F64, CONST_U8};
use engine::Engine;

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
