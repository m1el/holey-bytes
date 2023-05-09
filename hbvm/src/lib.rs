#![no_std]
extern crate alloc;

pub mod bytecode;
pub mod engine;
pub mod memory;

#[derive(Debug)]
pub enum RuntimeErrors {
    InvalidOpcode(u8),
    RegisterTooSmall,
    HostError(u64),
    PageNotMapped(u64),
}

// If you solve the halting problem feel free to remove this
pub enum HaltStatus {
    Halted,
    Running,
}
