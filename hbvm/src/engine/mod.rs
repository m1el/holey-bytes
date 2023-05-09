pub mod call_stack;
pub mod config;
pub mod enviroment_calls;
pub mod regs;

use {
    self::call_stack::CallStack,
    crate::{
        bytecode::{
            ops::{Operations::*, *},
            types::*,
        },
        engine::call_stack::FnCall,
        memory, HaltStatus, RuntimeErrors,
    },
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
    pub enviroment_call_table: [EnviromentCall; 256],
    pub call_stack: CallStack,
}
use crate::engine::enviroment_calls::EnviromentCall;
impl Engine {
    pub fn read_mem_addr_8(&mut self, address: u64) -> Result<u8, RuntimeErrors> {
        // println!("{}", address);
        self.memory.read_addr8(address)
    }
    pub fn set_timer_callback(&mut self, func: fn() -> u32) {
        self.timer_callback = Some(func);
    }
}

impl Engine {
    pub fn new(program: Vec<u8>) -> Self {
        let mut mem = memory::Memory::new();
        for (addr, byte) in program.clone().into_iter().enumerate() {
            let _ = mem.set_addr8(addr as u64, byte);
        }
        trace!("{:?}", mem.read_addr8(0));

        Self {
            index: 0,
            program,
            registers: Registers::new(),
            config: EngineConfig::default(),
            last_timer_count: 0,
            timer_callback: None,
            enviroment_call_table: [empty_enviroment_call; 256],
            memory: mem,
            call_stack: Vec::new(),
        }
    }

    pub fn dump(&self) {
        trace!("Registers");
        trace!(
            "A {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
            self.registers.a0,
            self.registers.a1,
            self.registers.a2,
            self.registers.a3,
            self.registers.a4,
            self.registers.a5,
            self.registers.a6,
            self.registers.a7,
            self.registers.a8,
            self.registers.a9,
        );
        trace!(
            "B {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
            self.registers.b0,
            self.registers.b1,
            self.registers.b2,
            self.registers.b3,
            self.registers.b4,
            self.registers.b5,
            self.registers.b6,
            self.registers.b7,
            self.registers.b8,
            self.registers.b9,
        );
        trace!(
            "C {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X}",
            self.registers.c0,
            self.registers.c1,
            self.registers.c2,
            self.registers.c3,
            self.registers.c4,
            self.registers.c5,
            self.registers.c6,
            self.registers.c7,
            self.registers.c8,
            self.registers.c9,
        );
        trace!(
            "D0-D4 {:016X} {:016X} {:016X} {:016X} {:016X}
D5-D9 {:016X} {:016X} {:016X} {:016X} {:016X}",
            self.registers.d0,
            self.registers.d1,
            self.registers.d2,
            self.registers.d3,
            self.registers.d4,
            self.registers.d5,
            self.registers.d6,
            self.registers.d7,
            self.registers.d8,
            self.registers.d9,
        );
        trace!(
            "E0-E4 {:016X} {:016X} {:016X} {:016X} {:016X}
E5-E9 {:016X} {:016X} {:016X} {:016X} {:016X}",
            self.registers.e0,
            self.registers.e1,
            self.registers.e2,
            self.registers.e3,
            self.registers.e4,
            self.registers.e5,
            self.registers.e6,
            self.registers.e7,
            self.registers.e8,
            self.registers.e9,
        );
        trace!(
            "F0-F4 {:016X} {:016X} {:016X} {:016X} {:016X}
F5-F9 {:016X} {:016X} {:016X} {:016X} {:016X}",
            self.registers.f0,
            self.registers.f1,
            self.registers.f2,
            self.registers.f3,
            self.registers.f4,
            self.registers.f5,
            self.registers.f6,
            self.registers.f7,
            self.registers.f8,
            self.registers.f9,
        );
    }
    pub fn run(&mut self) -> Result<HaltStatus, RuntimeErrors> {
        use {HaltStatus::*, RuntimeErrors::*};
        loop {
            // Break out of the loop
            if self.index + 1 == self.program.len() {
                break;
            }
            let op = (self.program[self.index], self.program[self.index + 1]);
            // println!("OP {} INDEX {}", op.0, self.index);
            match op {
                (0, _) => {
                    trace!("NO OP");
                    self.index += 1;
                }
                // Add a 8 bit num
                (1, 1) => {
                    let lhs = self.program[self.index + 2];
                    // println!("LHS 8BIT {}", lhs);
                    let rhs = self.program[self.index + 3];
                    // println!("RHS 8BIT {}", rhs);

                    let ret = lhs + rhs;
                    let reg = self.program[self.index + 4];

                    match reg {
                        0xA0..=0xC9 => {
                            self.set_register_8(reg, ret);
                        }
                        0xD0..=0xF9 => {
                            panic!("Register oversized")
                        }
                        _ => {
                            panic!("Not a register.")
                        }
                    }

                    self.index += 4;
                }
                // Add a 64 bit num
                (1, 2) => {
                    let mut lhs_array = [0; 8];
                    let mut rhs_array = [0; 8];

                    for (index, byte) in self.program[self.index + 2..self.index + 10]
                        .into_iter()
                        .enumerate()
                    {
                        lhs_array[index] = *byte;
                    }
                    let lhs = u64::from_be_bytes(lhs_array);
                    // println!("LHS 64BIT {}", lhs);

                    for (index, byte) in self.program[self.index + 10..self.index + 18]
                        .into_iter()
                        .enumerate()
                    {
                        rhs_array[index] = *byte;
                    }

                    let rhs = u64::from_be_bytes(rhs_array);

                    // println!("RHS 64BIT {}", rhs);

                    let ret = lhs + rhs;

                    let reg = self.program[self.index + 18];
                    // println!("Store {} in {:02X}", ret, reg);

                    match reg {
                        0xA0..=0xC9 => {
                            panic!("Register undersized")
                        }
                        0xD0..=0xF9 => {
                            self.set_register_64(reg, ret);
                        }
                        _ => {
                            panic!("Not a register.")
                        }
                    }

                    self.index += 18;
                }
                (2, 1) => {
                    let lhs = self.program[self.index + 2];
                    // println!("LHS 8BIT {}", lhs);
                    let rhs = self.program[self.index + 3];
                    // println!("RHS 8BIT {}", rhs);
                    let ret = lhs - rhs;
                    let reg = self.program[self.index + 4];

                    match reg {
                        0xA0..=0xC9 => {
                            self.set_register_8(reg, ret);
                        }
                        0xD0..=0xF9 => {
                            panic!("Register oversized")
                        }
                        _ => {
                            panic!("Not a register.")
                        }
                    }

                    self.index += 4;
                }
                (2, 2) => {
                    let mut lhs_array = [0; 8];
                    let mut rhs_array = [0; 8];

                    for (index, byte) in self.program[self.index + 2..self.index + 10]
                        .into_iter()
                        .enumerate()
                    {
                        lhs_array[index] = *byte;
                    }
                    let lhs = u64::from_be_bytes(lhs_array);
                    // println!("LHS 64BIT {}", lhs);

                    for (index, byte) in self.program[self.index + 10..self.index + 18]
                        .into_iter()
                        .enumerate()
                    {
                        rhs_array[index] = *byte;
                    }

                    let rhs = u64::from_be_bytes(rhs_array);

                    // println!("RHS 64BIT {}", rhs);

                    let ret = lhs - rhs;

                    let reg = self.program[self.index + 18];
                    // println!("Store {} in {:02X}", ret, reg);

                    match reg {
                        0xA0..=0xC9 => {
                            panic!("Register undersized")
                        }
                        0xD0..=0xF9 => {
                            self.set_register_64(reg, ret);
                        }
                        _ => {
                            panic!("Not a register.")
                        }
                    }

                    self.index += 19;
                }
                (2, 3) => {
                    // 8 bit
                    self.index += 4;
                }
                // TODO: Implement 64 bit register to register subtraction
                (2, 4) => {
                    // 64 bit

                    let mut lhs_array = [0; 8];
                    let mut rhs_array = [0; 8];

                    for (index, byte) in self.program[self.index + 2..self.index + 10]
                        .into_iter()
                        .enumerate()
                    {
                        lhs_array[index] = *byte;
                    }
                    let lhs = u64::from_be_bytes(lhs_array);

                    for (index, byte) in self.program[self.index + 10..self.index + 18]
                        .into_iter()
                        .enumerate()
                    {
                        rhs_array[index] = *byte;
                    }

                    let rhs = u64::from_be_bytes(rhs_array);

                    // println!("RHS 64BIT {}", rhs);

                    let ret = lhs - rhs;

                    let reg = self.program[self.index + 18];
                    // println!("Store {} in {:02X}", ret, reg);

                    match reg {
                        0xA0..=0xC9 => {
                            panic!("Register undersized")
                        }
                        0xD0..=0xF9 => {
                            self.set_register_64(reg, ret);
                        }
                        _ => {
                            panic!("Not a register.")
                        }
                    }

                    self.index += 18;
                }

                // Read from address to register
                (5, 0) => {
                    let mut addr_array = [0; 8];

                    for (index, byte) in self.program[self.index + 2..self.index + 10]
                        .into_iter()
                        .enumerate()
                    {
                        addr_array[index] = *byte;
                    }
                    let addr = u64::from_be_bytes(addr_array);

                    trace!("addr {}", addr);

                    let ret = self.read_mem_addr_8(addr);
                    match ret {
                        Ok(ret) => {
                            let reg = self.program[self.index + 10];
                            trace!("reg {}", reg);
                            self.set_register_8(reg, ret);
                            self.index += 10;
                        }
                        Err(err) => trace!("{:?}", err),
                    }
                }

                (100, _) => {
                    if self.call_stack.len() > self.config.call_stack_depth {
                        trace!("Callstack {}", self.call_stack.len());
                        break;
                    }

                    let mut addr_array = [0; 8];

                    for (index, byte) in self.program[self.index + 1..self.index + 9]
                        .into_iter()
                        .enumerate()
                    {
                        addr_array[index] = *byte;
                    }

                    let addr = usize::from_be_bytes(addr_array);
                    if addr > self.program.len() {
                        panic!("Invalid jump address {}", addr)
                    } else {
                        let call = FnCall { ret: self.index };
                        self.call_stack.push(call);

                        self.index = addr;
                        trace!("Jumping to {}", addr);

                        self.dump();
                        // panic!();
                    }
                }

                (255, int) => {
                    trace!("Enviroment Call {}", int);
                    let ret = self.enviroment_call_table[int as usize](self);
                    match ret {
                        Ok(eng) => {
                            trace!("Resuming execution at {}", eng.index);
                        }
                        Err(err) => {
                            return Err(HostError(err));
                        }
                    }
                    self.index += 2;
                }

                _op_pair => {
                    // println!("OP Pair {}", op_pair.0);
                    self.index += 1;
                }
            }

            // Finish step

            if self.timer_callback.is_some() {
                let ret = self.timer_callback.unwrap()();
                if (ret - self.last_timer_count) >= self.config.quantum {
                    return Ok(Running);
                }
            }
        }
        Ok(Halted)
    }

    pub fn set_register_8(&mut self, register: u8, value: u8) {
        match register {
            0xA0 => self.registers.a0 = value,
            0xA1 => self.registers.a1 = value,
            0xA2 => self.registers.a2 = value,
            0xA3 => self.registers.a3 = value,
            0xA4 => self.registers.a4 = value,
            0xA5 => self.registers.a5 = value,
            0xA6 => self.registers.a6 = value,
            0xA7 => self.registers.a7 = value,
            0xA8 => self.registers.a8 = value,
            0xA9 => self.registers.a9 = value,
            //
            0xB0 => self.registers.b0 = value,
            0xB1 => self.registers.b1 = value,
            0xB2 => self.registers.b2 = value,
            0xB3 => self.registers.b3 = value,
            0xB4 => self.registers.b4 = value,
            0xB5 => self.registers.b5 = value,
            0xB6 => self.registers.b6 = value,
            0xB7 => self.registers.b7 = value,
            0xB8 => self.registers.b8 = value,
            0xB9 => self.registers.b9 = value,
            //
            0xC0 => self.registers.c0 = value,
            0xC1 => self.registers.c1 = value,
            0xC2 => self.registers.c2 = value,
            0xC3 => self.registers.c3 = value,
            0xC4 => self.registers.c4 = value,
            0xC5 => self.registers.c5 = value,
            0xC6 => self.registers.c6 = value,
            0xC7 => self.registers.c7 = value,
            0xC8 => self.registers.c8 = value,
            0xC9 => self.registers.c9 = value,

            _ => {
                panic!("Unlikely you are here if everyone behaved.\nThis register is not 8 bit")
            }
        }
    }
    pub fn set_register_64(&mut self, register: u8, value: u64) {
        match register {
            0xD0 => self.registers.d0 = value,
            0xD1 => self.registers.d1 = value,
            0xD2 => self.registers.d2 = value,
            0xD3 => self.registers.d3 = value,
            0xD4 => self.registers.d4 = value,
            0xD5 => self.registers.d5 = value,
            0xD6 => self.registers.d6 = value,
            0xD7 => self.registers.d7 = value,
            0xD8 => self.registers.d8 = value,
            0xD9 => self.registers.d9 = value,
            //
            0xE0 => self.registers.f0 = value,
            0xE1 => self.registers.f1 = value,
            0xE2 => self.registers.f2 = value,
            0xE3 => self.registers.f3 = value,
            0xE4 => self.registers.f4 = value,
            0xE5 => self.registers.f5 = value,
            0xE6 => self.registers.f6 = value,
            0xE7 => self.registers.f7 = value,
            0xE8 => self.registers.f8 = value,
            0xE9 => self.registers.f9 = value,

            //
            0xF0 => self.registers.f0 = value,
            0xF1 => self.registers.f1 = value,
            0xF2 => self.registers.f2 = value,
            0xF3 => self.registers.f3 = value,
            0xF4 => self.registers.f4 = value,
            0xF5 => self.registers.f5 = value,
            0xF6 => self.registers.f6 = value,
            0xF7 => self.registers.f7 = value,
            0xF8 => self.registers.f8 = value,
            0xF9 => self.registers.f9 = value,

            _ => {
                panic!("Unlikely you are here if everyone behaved.\nThis register is not 64 bit")
            }
        }
    }
}
