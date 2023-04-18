pub mod call_stack;
pub mod config;
pub mod regs;

use crate::bytecode::ops::*;
use crate::bytecode::types::*;

use crate::HaltStatus;
use crate::Page;
use crate::RuntimeErrors;
use config::EngineConfig;
use regs::Registers;

use self::call_stack::CallStack;

pub struct Engine {
    pub index: usize,
    program: Vec<u8>,
    registers: Registers,
    config: EngineConfig,

    /// BUG: This DOES NOT account for overflowing
    last_timer_count: u32,
    timer_callback: Option<fn() -> u32>,
    memory: Vec<Page>,
    call_stack: CallStack,
}

impl Engine {
    pub fn new(program: Vec<u8>) -> Self {
        Self {
            index: 0,
            program,
            registers: Registers::new(),
            config: EngineConfig::default(),
            last_timer_count: 0,
            timer_callback: None,
            memory: vec![],
            call_stack: vec![],
        }
    }
    pub fn dump(&self) {
        println!("Registers");
        println!(
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
        println!(
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
        println!(
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

        println!(
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
        println!(
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
        println!(
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
        use HaltStatus::*;
        use RuntimeErrors::*;
        loop {
            // Break out of the loop
            if self.index == self.program.len() {
                break;
            }
            let op = self.program[self.index];
            println!("OP {} INDEX {}", self.program[self.index], self.index);
            match op {
                NOP => self.index += 1,
                ADD => {
                    print!("Add");
                    self.index += 1;
                    let mut lhs: Vec<u8> = vec![];
                    let mut rhs: Vec<u8> = vec![];
                    let mut lhs_signed = false;
                    let mut rhs_signed = false;

                    match self.program[self.index] {
                        CONST_U8 => {
                            self.index += 1;
                            lhs.push(self.program[self.index]);
                            print!(" constant {:?}", lhs[0]);
                            lhs_signed = false;
                            self.index += 1;
                        }
                        CONST_U64 => {
                            self.index += 1;
                            lhs.push(self.program[self.index]);
                            self.index += 1;
                            lhs.push(self.program[self.index]);
                            self.index += 1;
                            lhs.push(self.program[self.index]);
                            self.index += 1;
                            lhs.push(self.program[self.index]);

                            lhs_signed = false;
                            self.index += 1;
                        }
                        0xA0..=0xC9 => {
                            println!("TRACE: 8 bit lhs");
                        }
                        0xD0..=0xFF => {
                            println!("TRACE: 64 bit lhs");
                        }
                        op => return Err(InvalidOpcode(op)),
                    }

                    match self.program[self.index] {
                        CONST_U8 => {
                            self.index += 1;
                            rhs.push(self.program[self.index]);
                            rhs_signed = false;
                            print!(" constant {:?}", rhs[0]);
                            self.index += 1;
                        }

                        CONST_U64 => {
                            self.index += 1;
                            rhs.push(self.program[self.index]);
                            self.index += 1;
                            rhs.push(self.program[self.index]);
                            self.index += 1;
                            rhs.push(self.program[self.index]);
                            self.index += 1;
                            rhs.push(self.program[self.index]);

                            print!(" constant {:?}", rhs[0]);
                            rhs_signed = false;
                            self.index += 1;
                        }
                        0xA0..=0xC9 => {
                            println!("TRACE: 8 bit rhs");
                        }
                        0xD0..=0xFF => {
                            println!("TRACE: 64 bit rhs");
                        }

                        _ => {
                            panic!()
                        }
                    }
                    match self.program[self.index] {
                        0xA0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in A0");

                            let sum = lhs[0] + rhs[0];
                            self.registers.a0 = sum;
                            self.index += 1;
                        }
                        0xB0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in B0");
                            let sum = lhs[0] + rhs[0];
                            self.registers.b0 = sum;
                            self.index += 1;
                        }
                        0xC0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in C0");
                            let sum = lhs[0] + rhs[0];
                            self.registers.c8 = sum;
                            self.index += 1;
                        }
                        0xD0 => {
                            if lhs.len() > 4 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 4 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in D0");
                            let lhs: u64 = Into::<u64>::into(lhs[3]) << 60;
                            println!("{}", lhs);
                            println!("{}", 2);
                            // let rhs: u64 = (rhs[4] << 16).into();
                            let rhs: u64 = 0;
                            let sum = lhs + rhs;
                            self.registers.d0 = sum;
                            self.index += 1;
                        }

                        op => {
                            println!("{}", op)
                        }
                    }
                }
                SUB => {
                    print!("Sub");
                    self.index += 1;
                    let mut lhs: Vec<u8> = vec![];
                    let mut rhs: Vec<u8> = vec![];
                    let mut lhs_signed = false;
                    let mut rhs_signed = false;

                    match self.program[self.index] {
                        0xA0 => {
                            lhs.push(self.registers.a8);
                            lhs_signed = false;
                            print!(" constant {:?}", self.registers.a8);
                            self.index += 1;
                        }
                        0xB0 => {
                            lhs.push(self.registers.b8);
                            lhs_signed = false;
                            print!(" constant {:?}", self.registers.b8);
                            self.index += 1;
                        }
                        CONST_U8 => {
                            self.index += 1;
                            lhs.push(self.program[self.index]);
                            print!(" constant {:?}", lhs[0]);
                            lhs_signed = false;
                            self.index += 1;
                        }
                        op => return Err(InvalidOpcode(op)),
                    }

                    match self.program[self.index] {
                        0xB0 => {
                            rhs.push(self.registers.b8);
                            rhs_signed = false;
                            print!(" constant {:?}", self.registers.b8);
                            self.index += 1;
                        }
                        _ => {
                            panic!()
                        }
                    }
                    match self.program[self.index] {
                        0xA0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            println!(" store in A8");

                            let sum = lhs[0] - rhs[0];
                            self.registers.a8 = sum;
                            self.index += 1;
                        }
                        0xB0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in B8");
                            let sum = lhs[0] - rhs[0];
                            self.registers.b8 = sum;
                            self.index += 1;
                        }
                        0xC0 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in B8");
                            let sum = lhs[0] - rhs[0];
                            self.registers.c8 = sum;
                            self.index += 1;
                        }
                        _ => {
                            panic!()
                        }
                    }
                }
                op => {
                    println!("INVALID OPCODE {}", op);
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
}
