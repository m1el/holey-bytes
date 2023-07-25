//! Validate if program is sound to execute

/// Program validation error kind
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Unknown opcode
    InvalidInstruction,
    /// VM doesn't implement this valid opcode
    Unimplemented,
    /// Attempted to copy over register boundary
    RegisterArrayOverflow,
    /// Program is not validly terminated
    InvalidEnd,
}

/// Error
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error {
    /// Kind
    pub kind:  ErrorKind,
    /// Location in bytecode
    pub index: usize,
}

/// Perform bytecode validation. If it passes, the program should be
/// sound to execute.
pub fn validate(mut program: &[u8]) -> Result<(), Error> {
    // Program has to end with 12 zeroes, if there is less than
    // 12 bytes, program is invalid.
    if program.len() < 12 {
        return Err(Error {
            kind:  ErrorKind::InvalidEnd,
            index: 0,
        });
    }

    // Verify that program ends with 12 zeroes
    for (index, item) in program.iter().enumerate().skip(program.len() - 12) {
        if *item != 0 {
            return Err(Error {
                kind: ErrorKind::InvalidEnd,
                index,
            });
        }
    }

    let start = program;
    loop {
        use hbbytecode::opcode::*;
        // Match on instruction types and perform necessary checks
        program = match program {
            // End of program
            [] => return Ok(()),

            // Memory load/store cannot go out-of-bounds register array
            [LD..=ST, reg, _, _, _, _, _, _, _, _, count_0, count_1, ..]
                if usize::from(*reg) * 8
                    + usize::from(u16::from_le_bytes([*count_1, *count_0]))
                    > 2048 =>
            {
                return Err(Error {
                    kind:  ErrorKind::RegisterArrayOverflow,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }

            // Block register copy cannot go out-of-bounds register array
            [BRC, src, dst, count, ..]
                if src.checked_add(*count).is_none() || dst.checked_add(*count).is_none() =>
            {
                return Err(Error {
                    kind:  ErrorKind::RegisterArrayOverflow,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }

            // Valid instructions
            [DIR | DIRF | FMAF, _, _, _, _, rest @ ..] // BBBB
            | [ADD | SUB | MUL | AND | OR | XOR | SL | SR | SRS | CMP | CMPU | BRC | ADDF | SUBF | MULF, _, _, _, rest @ ..]
            | [LD | ST, _, _, _, _, _, _, _, _, _, _, _, rest @ ..] // BBDH
            | [
                ADDI | MULI | ANDI | ORI | XORI | CMPI | CMPUI | BMC | JAL | JEQ | JNE | JLT | JGT | JLTU | JGTU | ADDFI | MULFI, _, _, _, _, _, _, _, _, _, _, rest @ ..] // BBD
            | [SLI | SRI | SRSI, _, _, _, _, _, _, rest @ ..] // BBW
            | [NEG | NOT | CP | SWA | NEGF | ITF | FTI, _, _, rest @ ..] // BB
            | [LI, _, _, _, _, _, _, _, _, _, rest @ ..] // BD
            | [UN | NOP | ECALL, rest @ ..] // N
            => rest,

            // The rest
            _ => {
                return Err(Error {
                    kind:  ErrorKind::InvalidInstruction,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }
        }
    }
}
