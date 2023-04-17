#![allow(unused_assignments)]
#![allow(unused_must_use)]

pub mod bytecode;
use bytecode::*;

pub struct Page {
    data: [u8; 4096],
}

#[repr(u16)]
/// All instructions will use `op src dest` format (if they take arguments)
/// most `src`s can be replaced with a constant
pub enum Instructions {
    Nop = 0, // No Operation or nothing to do
    // Add takes two `src`s and adds into a `dest`
    // add
    Add,
    Sub,
    Mul,
    Div, //HTML Div not division for that refer to ~~Blam~~

    Store,
    // Load a value from one src to one dest
    // `load a8 c8`
    Load,
}

// A `src` may be any of the following
// Memory Address
// Register
// Constant
// ~~Port~~

// A `dest` may be any of the following
// Memory Address
// Register

// 0000 ;; NOP
// // Once we get to an instruction that makes sense we can interpret the next bytes relative to that instruction
// 0001 ;; add

// // EXAMPLE
// 0001
// // This grouping is called a pairing and is useful as a short little guide for how to read AOB
// (0xC0 0x03)/*0xC0 represents a constant that is size u8, 0x03 is 3*/
// (0xC1 0x01)/*0xC1 represents a constant that is size i8, 0x01 is 1*/
// (0xA0 0x01)/* 0xA0 represents a the a8 register as an unsigned number, 0x01 is 1*/
// 0002
// (0xC3 0x01 0x00)/*0xC3 represents a constant that is size u16 , 0x01 0x00 is 256*/
#[test]
fn main() {
    #[rustfmt::skip]
    let prog: Vec<u8> = vec![
        NOP,
        ADD, CONST_U8, 1, CONST_U8, 20, RegisterA8,
        ADD, CONST_U8, 1, CONST_U8,  0, RegisterB8,
        SUB, CONST_U8, 3, RegisterA8,   RegisterC8,
    ];
    let mut eng = Engine::new(prog);
    // eng.timer_callback = Some(time);
    eng.run();
    eng.dump();
}
pub fn time() -> u32 {
    9
}

pub enum RuntimeErrors {
    InvalidOpcode(u8),
    RegisterTooSmall,
}

// If you solve the halting problem feel free to remove this
pub enum HaltStatus {
    Halted,
    Running,
}

#[rustfmt::skip]
pub struct Registers{
    a8: u8, a16: u16, a32: u32, a64: u64,
    b8: u8, b16: u16, b32: u32, b64: u64,
    c8: u8, c16: u16, c32: u32, c64: u64,
    d8: u8, d16: u16, d32: u32, d64: u64,
    e8: u8, e16: u16, e32: u32, e64: u64,
    f8: u8, f16: u16, f32: u32, f64: u64,
}
impl Registers {
    #[rustfmt::skip]
    pub fn new() -> Self{
        Self{
            a8: 0, a16: 0, a32: 0, a64: 0,
            b8: 0, b16: 0, b32: 0, b64: 0,
            c8: 0, c16: 0, c32: 0, c64: 0,
            d8: 0, d16: 0, d32: 0, d64: 0,
            e8: 0, e16: 0, e32: 0, e64: 0,
            f8: 0, f16: 0, f32: 0, f64: 0,
        }
    }
}

pub const ENGINE_DELTA: u32 = 1;

pub struct Engine {
    index: usize,
    program: Vec<u8>,
    registers: Registers,
    /// BUG: This DOES NOT account for overflowing
    last_timer_count: u32,
    timer_callback: Option<fn() -> u32>,
}
impl Engine {
    pub fn new(program: Vec<u8>) -> Self {
        Self {
            index: 0,
            program,
            registers: Registers::new(),
            last_timer_count: 0,
            timer_callback: None,
        }
    }
    pub fn dump(&self) {
        println!("Reg A8 {}", self.registers.a8);
        println!("Reg B8 {}", self.registers.b8);
        println!("Reg C8 {}", self.registers.c8);
        println!("Reg D8 {}", self.registers.d8);
        println!("Reg E8 {}", self.registers.e8);
        println!("Reg F8 {}", self.registers.f8);
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
                        _ => {
                            panic!()
                        }
                    }
                    match self.program[self.index] {
                        RegisterA8 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in A8");

                            let sum = lhs[0] + rhs[0];
                            self.registers.a8 = sum;
                            self.index += 1;
                        }
                        RegisterB8 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in B8");
                            let sum = lhs[0] + rhs[0];
                            self.registers.b8 = sum;
                            self.index += 1;
                        }
                        RegisterC8 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in C8");
                            let sum = lhs[0] + rhs[0];
                            self.registers.c8 = sum;
                            self.index += 1;
                        }

                        RegisterD8 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            if rhs.len() > 1 {
                                panic!("RHS is not an 8 bit number")
                            }
                            println!(" store in D8");
                            let sum = lhs[0] + rhs[0];
                            self.registers.d8 = sum;
                            self.index += 1;
                        }

                        _ => {
                            panic!()
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
                        RegA8 => {
                            lhs.push(self.registers.a8);
                            lhs_signed = false;
                            print!(" constant {:?}", self.registers.a8);
                            self.index += 1;
                        }
                        RegB8 => {
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
                        RegB8 => {
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
                        RegisterA8 => {
                            if lhs.len() > 1 {
                                panic!("LHS is not an 8 bit number")
                            }
                            println!(" store in A8");

                            let sum = lhs[0] - rhs[0];
                            self.registers.a8 = sum;
                            self.index += 1;
                        }
                        RegisterB8 => {
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
                        RegisterC8 => {
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
                if (ret - self.last_timer_count) >= ENGINE_DELTA {
                    return Ok(Running);
                }
            }
        }
        Ok(Halted)
    }
}

pub struct HandSide {
    signed: bool,
    num8: Option<u8>,
    num64: Option<u64>,
}

pub fn math_handler(math_op: u8, lhs: (bool, [u8; 4]), rhs: [u8; 4]) {
    match math_op {
        ADD => {}
        _ => {}
    }
}
