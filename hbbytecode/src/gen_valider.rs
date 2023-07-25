//! Generate HoleyBytes code validator

macro_rules! gen_valider {
    (
        $(
            $ityn:ident
            ($($param_i:ident: $param_ty:ident),* $(,)?)
            => [$($opcode:ident),* $(,)?],
        )*
    ) => {
        #[allow(unreachable_code)]
        pub mod valider {
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
                    use crate::opcode::*;
                    extern crate std;
                    program = match program {
                        // End of program
                        [] => return Ok(()),

                        // Memory load/store cannot go out-of-bounds register array
                        //         B   B  D1 D2 D3 D4 D5 D6 D7 D8    H1      H2
                        [LD..=ST, reg, _, _, _, _, _, _, _, _, _, count_0, count_1, ..]
                            if usize::from(*reg) * 8
                                + usize::from(u16::from_le_bytes([*count_0, *count_1]))
                                > 2048 =>
                        {
                            return Err(Error {
                                kind:  ErrorKind::RegisterArrayOverflow,
                                index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                            });
                        }

                        // Block register copy cannot go out-of-bounds register array
                        [BRC, src, dst, count, ..]
                            if src.checked_add(*count).is_none()
                                || dst.checked_add(*count).is_none() =>
                        {
                            return Err(Error {
                                kind:  ErrorKind::RegisterArrayOverflow,
                                index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                            });
                        }

                        $(
                            $crate::gen_valider::inst_chk!(
                                rest, $ityn, $($opcode),*
                            )
                        )|* => rest,

                        // The plebs
                        _ => {
                            return Err(Error {
                                kind:  ErrorKind::InvalidInstruction,
                                index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                            })
                        }
                    }
                }
            }
        }
    };
}

/// Generate instruction check pattern
macro_rules! inst_chk {
    // Sadly this has hardcoded instruction types,
    // as I cannot generate parts of patterns+

    ($rest:ident, bbbb, $($opcode:ident),*) => {
        //             B  B  B  B
        [$($opcode)|*, _, _, _, _, $rest @ ..]
    };

    ($rest:ident, bbb, $($opcode:ident),*) => {
        //             B  B  B
        [$($opcode)|*, _, _, _, $rest @ ..]
    };

    ($rest:ident, bbdh, $($opcode:ident),*) => {
        //             B  B  D1 D2 D3 D4 D5 D6 D7 D8 H1 H2
        [$($opcode)|*, _, _, _, _, _, _, _, _, _, _, _, _, $rest @ ..]
    };

    ($rest:ident, bbd, $($opcode:ident),*) => {
        //             B  B  D1 D2 D3 D4 D5 D6 D7 D8
        [$($opcode)|*, _, _, _, _, _, _, _, _, _, _, $rest @ ..]
    };

    ($rest:ident, bbw, $($opcode:ident),*) => {
        //             B  B  W1 W2 W3 W4
        [$($opcode)|*, _, _, _, _, _, _, $rest @ ..]
    };

    ($rest:ident, bb, $($opcode:ident),*) => {
        //             B  B
        [$($opcode)|*, _, _, $rest @ ..]
    };

    ($rest:ident, bd, $($opcode:ident),*) => {
        //             B  D1 D2 D3 D4 D5 D6 D7 D8
        [$($opcode)|*, _, _, _, _, _, _, _, _, _, $rest @ ..]
    };

    ($rest:ident, n, $($opcode:ident),*) => {
        [$($opcode)|*, $rest @ ..]
    };

    ($_0:ident, $($_1:ident),*) => {
        compile_error!("Invalid instruction type");
    }
}

pub(crate) use {gen_valider, inst_chk};
