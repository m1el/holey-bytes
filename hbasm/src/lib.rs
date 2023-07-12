#![no_std]

extern crate alloc;

mod macros;

use {alloc::vec::Vec, hashbrown::HashSet};

#[derive(Default)]
pub struct Assembler {
    pub buf: Vec<u8>,
    pub sub: HashSet<usize>,
}

macros::impl_both!(
    bbbb(p0: R, p1: R, p2: R, p3: R)
        => [DIR, DIRF, FMAF],
    bbb(p0: R, p1: R, p2: R)
        => [ADD, SUB, MUL, AND, OR, XOR, SL, SR, SRS, CMP, CMPU, BRC, ADDF, SUBF, MULF],
    bbdh(p0: R, p1: R, p2: I, p3: u16)
        => [LD, ST],
    bbd(p0: R, p1: R, p2: I)
        => [ADDI, MULI, ANDI, ORI, XORI, SLI, SRI, SRSI, CMPI, CMPUI,
            BMC, JEQ, JNE, JLT, JGT, JLTU, JGTU, ADDFI, MULFI],
    bb(p0: R, p1: R)
        => [NEG, NOT, CP, SWA, NEGF, ITF, FTI],
    bd(p0: R, p1: I)
        => [LI, JMP],
    n()
        => [NOP, ECALL],
);

pub trait Imm {
    fn insert(&self, asm: &mut Assembler);
}

macro_rules! impl_imm_le_bytes {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Imm for $ty {
                #[inline(always)]
                fn insert(&self, asm: &mut Assembler) {
                    asm.buf.extend(self.to_le_bytes());
                }
            }
        )*
    };
}

impl_imm_le_bytes!(u64, i64, f64);
