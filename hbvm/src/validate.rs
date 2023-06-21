#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidInstruction,
    Unimplemented,
    RegisterArrayOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub index: usize,
}

pub fn validate(mut program: &[u8]) -> Result<(), Error> {
    use hbbytecode::opcode::*;

    let start = program;
    loop {
        program = match program {
            [] => return Ok(()),
            [LD..=ST, reg, _, _, _, _, _, _, _, _, _, count, ..]
                if usize::from(*reg) * 8 + usize::from(*count) > 2048 =>
            {
                return Err(Error {
                    kind: ErrorKind::RegisterArrayOverflow,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }
            [BRC, src, dst, count, ..]
                if src.checked_add(*count).is_none() || dst.checked_add(*count).is_none() =>
            {
                return Err(Error {
                    kind: ErrorKind::RegisterArrayOverflow,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }
            [NOP | ECALL, rest @ ..]
            | [DIR | DIRF, _, _, _, _, rest @ ..]
            | [ADD..=CMPU | BRC | ADDF..=MULF, _, _, _, rest @ ..]
            | [NEG..=NOT | CP..=SWA, _, _, rest @ ..]
            | [LI | JMP, _, _, _, _, _, _, _, _, _, rest @ ..]
            | [ADDI..=CMPUI | BMC | JEQ..=JGTU | ADDFI..=MULFI, _, _, _, _, _, _, _, _, _, _, rest @ ..]
            | [LD..=ST, _, _, _, _, _, _, _, _, _, _, _, _, rest @ ..] => rest,
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidInstruction,
                    index: (program.as_ptr() as usize) - (start.as_ptr() as usize),
                })
            }
        }
    }
}
