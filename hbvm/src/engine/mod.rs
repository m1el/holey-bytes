pub mod call_stack;
pub mod config;
pub mod enviroment_calls;
pub mod regs;
#[cfg(test)]
pub mod tests;

use {
    self::call_stack::CallStack,
    crate::{memory, HaltStatus, RuntimeErrors},
    alloc::vec::Vec,
    config::EngineConfig,
    log::trace,
    regs::Registers,
};
// pub const PAGE_SIZE: usize = 8192;

pub struct RealPage {
    pub ptr: *mut u8,
}

#[derive(Debug, Clone, Copy)]
pub struct VMPage {
    pub data: [u8; 8192],
}
impl VMPage {
    pub fn new() -> Self {
        Self {
            data: [0; 4096 * 2],
        }
    }
}

pub enum Page {
    VMPage(VMPage),
    RealPage(RealPage),
}
impl Page {
    pub fn data(&self) -> [u8; 4096 * 2] {
        match self {
            Page::VMPage(vmpage) => vmpage.data,
            Page::RealPage(_) => {
                unimplemented!("Memmapped hw page not yet supported")
            }
        }
    }
}

pub fn empty_enviroment_call(engine: &mut Engine) -> Result<&mut Engine, u64> {
    trace!("Registers {:?}", engine.registers);
    Err(0)
}

pub struct Engine {
    pub index:     usize,
    pub program:   Vec<u8>,
    pub registers: Registers,
    pub config:    EngineConfig,

    /// BUG: This DOES NOT account for overflowing
    pub last_timer_count: u32,
    pub timer_callback: Option<fn() -> u32>,
    pub memory: memory::Memory,
    pub enviroment_call_table: [Option<EnviromentCall>; 256],
    pub call_stack: CallStack,
}
use crate::engine::enviroment_calls::EnviromentCall;
impl Engine {
    pub fn set_timer_callback(&mut self, func: fn() -> u32) {
        self.timer_callback = Some(func);
    }
    pub fn set_register(&mut self, register: u8, value: u64) {}
}

impl Engine {
    pub fn new(program: Vec<u8>) -> Self {
        let mut mem = memory::Memory::new();
        for (addr, byte) in program.clone().into_iter().enumerate() {
            let _ = mem.set_addr8(addr as u64, byte);
        }
        trace!("{:?}", mem.read_addr8(0));
        let ecall_table: [Option<EnviromentCall>; 256] = [None; 256];
        Self {
            index: 0,
            program,
            registers: Registers::new(),
            config: EngineConfig::default(),
            last_timer_count: 0,
            timer_callback: None,
            enviroment_call_table: ecall_table,
            memory: mem,
            call_stack: Vec::new(),
        }
    }

    pub fn dump(&self) {}
    pub fn run(&mut self) -> Result<HaltStatus, RuntimeErrors> {
        Ok(HaltStatus::Halted)
    }
}
