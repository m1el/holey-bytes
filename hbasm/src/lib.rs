#![no_std]

extern crate alloc;

mod macros;

use {alloc::vec::Vec, hashbrown::HashSet};

/// Assembler
/// 
/// - Opcode-generic, instruction-type-specific methods are named `i_param_<type>`
///     - You likely won't need to use them, but they are here, just in case :)
/// - Instruction-specific methods are named `i_<instruction>`
#[derive(Default)]
pub struct Assembler {
    pub buf: Vec<u8>,
    pub sub: HashSet<usize>,
}


// Implement both assembler and generate module for text-code-based one
macros::impl_both!(
    bbbb(p0: R, p1: R, p2: R, p3: R)
        => [DIR, DIRF, FMAF],
    bbb(p0: R, p1: R, p2: R)
        => [ADD, SUB, MUL, AND, OR, XOR, SL, SR, SRS, CMP, CMPU, /*BRC,*/ ADDF, SUBF, MULF],
    bbdh(p0: R, p1: R, p2: I, p3: u16)
        => [LD, ST],
    bbd(p0: R, p1: R, p2: I)
        => [ADDI, MULI, ANDI, ORI, XORI, SLI, SRI, SRSI, CMPI, CMPUI,
            BMC, JAL, JEQ, JNE, JLT, JGT, JLTU, JGTU, ADDFI, MULFI],
    bb(p0: R, p1: R)
        => [NEG, NOT, CP, SWA, NEGF, ITF, FTI],
    bd(p0: R, p1: I)
        => [LI],
    n()
        => [UN, NOP, ECALL],
);

impl Assembler {
    // Special-cased for text-assembler
    //
    // `p2` is not a register, but the instruction is still BBB
    #[inline(always)]
    pub fn i_brc(&mut self, p0: u8, p1: u8, p2: u8) {
        self.i_param_bbb(hbbytecode::opcode::BRC, p0, p1, p2)
    }

    /// Append 12 zeroes (UN) at the end
    /// 
    /// # HBVM lore
    ///
    /// In reference HBVM implementation checks are done in
    /// a separate phase before execution.
    ///
    /// This way execution will be much faster as they have to
    /// be done only once.
    ///
    /// There was an issue. You cannot statically check register values and
    /// `JAL` instruction could hop at the end of program to some byte, which
    /// will be interpreted as opcode and VM in attempt to decode the instruction
    /// performed out-of-bounds read which leads to undefined behaviour.
    ///
    /// Several options were considered to overcome this, but inserting some data at
    /// program's end which when executed would lead to undesired behaviour, though
    /// not undefined behaviour.
    ///
    /// Newly created `UN` (as UNreachable) was chosen as
    /// - It was a good idea to add some equivalent to `ud2` anyways
    /// - It was chosen to be zero
    /// - What if you somehow reached that code, it will appropriately bail :)
    /// - (yes, originally `NOP` was considered)
    ///
    /// Why 12 bytes? That's the size of largest instruction parameter part.
    pub fn finalise(&mut self) {
        self.buf.extend([0; 12]);
    }
}

/// Immediate value
/// 
/// # Implementor notice
/// It should insert exactly 8 bytes, otherwise output will be malformed.
/// This is not checked in any way
pub trait Imm {
    /// Insert immediate value
    fn insert(&self, asm: &mut Assembler);
}

/// Implement immediate values
macro_rules! impl_imm_le_bytes {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Imm for $ty {
                #[inline(always)]
                fn insert(&self, asm: &mut Assembler) {
                    // Convert to little-endian bytes, insert.
                    asm.buf.extend(self.to_le_bytes());
                }
            }
        )*
    };
}

impl_imm_le_bytes!(u64, i64, f64);
