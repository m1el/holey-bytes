#![no_std]
#![feature(error_in_core)]

extern crate alloc;

pub mod text;

mod macros;

use {alloc::vec::Vec, hashbrown::HashSet};

#[derive(Default)]
pub struct Assembler {
    pub buf: Vec<u8>,
    sub:     HashSet<usize>,
}

impl Assembler {
    macros::impl_asm!(
        bbbb(p0: u8, p1: u8, p2: u8, p3: u8)
            => [DIR, DIRF],
        bbb(p0: u8, p1: u8, p2: u8)
            => [ADD, SUB, MUL, AND, OR, XOR, SL, SRS, CMP, CMPU, BRC, ADDF, MULF],
        bbdh(p0: u8, p1: u8, p2: impl Imm, p3: u16)
            => [LD, ST],
        bbd(p0: u8, p1: u8, p2: impl Imm)
            => [ADDI, MULI, ANDI, ORI, XORI, SLI, SRI, SRSI, CMPI, CMPUI,
                JEQ, JNE, JLT, JGT, JLTU, JGTU, ADDFI, MULFI],
        bb(p0: u8, p1: u8)
            => [NEG, NOT, CP, SWA],
        bd(p0: u8, p1: impl Imm)
            => [LI, JMP],
        n()
            => [NOP, ECALL],
    );
}

pub trait Imm {
    fn insert(self, asm: &mut Assembler);
}

impl Imm for u64 {
    #[inline(always)]
    fn insert(self, asm: &mut Assembler) {
        asm.buf.extend(self.to_le_bytes());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Symbol(pub u64);
impl Imm for Symbol {
    #[inline(always)]
    fn insert(self, asm: &mut Assembler) {
        asm.sub.insert(asm.buf.len());
        asm.buf.extend(self.0.to_le_bytes());
    }
}
