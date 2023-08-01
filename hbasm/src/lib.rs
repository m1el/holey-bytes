//! Holey Bytes Assembler
//!
//! Some people claim:
//! > Write programs to handle text streams, because that is a universal interface.
//!
//! We at AbleCorp believe that nice programatic API is nicer than piping some text
//! into a program. It's less error-prone and faster.
//!
//! So this crate contains both assembleer with API for programs and a text assembler
//! for humans to write

#![no_std]

extern crate alloc;

mod macros;

use {
    alloc::{vec, vec::Vec},
    hashbrown::HashSet,
};

/// Assembler
///
/// - Opcode-generic, instruction-type-specific methods are named `i_param_<type>`
///     - You likely won't need to use them, but they are here, just in case :)
/// - Instruction-specific methods are named `i_<instruction>`
pub struct Assembler {
    pub buf: Vec<u8>,
    pub sub: HashSet<usize>,
}

impl Default for Assembler {
    fn default() -> Self {
        Self {
            buf: vec![0; 3],
            sub: Default::default(),
        }
    }
}

hbbytecode::invoke_with_def!(macros::text::gen_text);

impl Assembler {
    hbbytecode::invoke_with_def!(macros::asm::impl_asm);

    /// Append 12 zeroes (UN) at the end and add magic to the begining
    ///
    /// # HoleyBytes lore
    ///
    /// In reference HBVM implementation checks are done in
    /// a separate phase before execution.
    ///
    /// This way execution will be much faster as they have to
    /// be done only once.
    ///
    /// There was an issue. You cannot statically check register values and
    /// `JAL` instruction could hop at the end of program to some byte, which
    /// will be interpreted as some valid opcode and VM in attempt to decode
    /// the instruction performed out-of-bounds read which leads to undefined behaviour.
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
        self.buf[0..4].copy_from_slice(&0xAB1E0B_u32.to_le_bytes());
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
