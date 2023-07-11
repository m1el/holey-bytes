extern crate alloc;
use alloc::vec::Vec;

use {
    core::fmt::{Display, Formatter},
    hashbrown::HashMap,
    lasso::{Rodeo, Spur},
    logos::{Lexer, Logos, Span},
};

macro_rules! tokendef {
    ($($opcode:literal),* $(,)?) => {
        paste::paste! {
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Logos)]
            #[logos(extras = Rodeo)]
            #[logos(skip r"[ \t\f]+")]
            #[logos(skip r"-- .*")]
            pub enum Token {
                $(#[token($opcode, |_| hbbytecode::opcode::[<$opcode:upper>])])*
                OpCode(u8),

                #[regex("[0-9]+", |l| l.slice().parse().ok())]
                #[regex(
                    "-[0-9]+",
                    |lexer| {
                        Some(u64::from_ne_bytes(lexer.slice().parse::<i64>().ok()?.to_ne_bytes()))
                    },
                )] Integer(u64),

                #[regex(
                    "r[0-9]+",
                    |lexer| match lexer.slice()[1..].parse() {
                        Ok(n) => Some(n),
                        _ => None
                    },
                )] Register(u8),

                #[regex(
                    r"\p{XID_Start}\p{XID_Continue}*:",
                    |lexer| lexer.extras.get_or_intern(&lexer.slice()[..lexer.slice().len() - 1]),
                )] Label(Spur),

                #[regex(
                    r"\p{XID_Start}\p{XID_Continue}*",
                    |lexer| lexer.extras.get_or_intern(lexer.slice()),
                )] Symbol(Spur),

                #[token("\n")]
                #[token(";")] ISep,
                #[token(",")] PSep,
            }
        }
    };
}

#[rustfmt::skip]
tokendef![
    "nop", "add", "sub", "mul", "and", "or", "xor", "sl", "sr", "srs", "cmp", "cmpu",
    "dir", "neg", "not", "addi", "muli", "andi", "ori", "xori", "sli", "sri", "srsi",
    "cmpi", "cmpui", "cp", "swa", "li", "ld", "st", "bmc", "brc", "jmp", "jeq", "jne",
    "jlt", "jgt", "jltu", "jgtu", "ecall", "addf", "subf", "mulf", "dirf", "fmaf", "negf",
    "itf", "fti", "addfi", "mulfi",
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    UnexpectedToken,
    InvalidToken,
    UnexpectedEnd,
    InvalidSymbol,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub span: Span,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Error {:?} at {:?}", self.kind, self.span)
    }
}

impl core::error::Error for Error {}

macro_rules! expect_matches {
    ($self:expr, $($pat:pat),* $(,)?) => {$(
        let $pat = $self.next()?
            else { return Err(ErrorKind::UnexpectedToken) };
    )*}
}

pub fn assembly(code: &str, buf: &mut Vec<u8>) -> Result<(), Error> {
    struct Assembler<'a> {
        lexer: Lexer<'a, Token>,
        buf: &'a mut Vec<u8>,
        label_map: HashMap<Spur, u64>,
        to_sub_label: HashMap<usize, Spur>,
    }

    impl<'a> Assembler<'a> {
        fn next(&mut self) -> Result<Token, ErrorKind> {
            match self.lexer.next() {
                Some(Ok(t)) => Ok(t),
                Some(Err(())) => Err(ErrorKind::InvalidToken),
                None => Err(ErrorKind::UnexpectedEnd),
            }
        }

        fn assemble(&mut self) -> Result<(), ErrorKind> {
            use hbbytecode::opcode::*;
            loop {
                match self.lexer.next() {
                    Some(Ok(Token::OpCode(op))) => {
                        self.buf.push(op);
                        match op {
                            NOP | ECALL => Ok(()),
                            DIR | DIRF => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                    Token::PSep,
                                    Token::Register(r2),
                                    Token::PSep,
                                    Token::Register(r3),
                                );
                                self.buf.extend([r0, r1, r2, r3]);
                                Ok(())
                            }
                            ADD..=CMPU | ADDF..=MULF => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                    Token::PSep,
                                    Token::Register(r2),
                                );
                                self.buf.extend([r0, r1, r2]);
                                Ok(())
                            }
                            BRC => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                    Token::PSep,
                                    Token::Integer(count),
                                );
                                self.buf.extend([
                                    r0,
                                    r1,
                                    u8::try_from(count).map_err(|_| ErrorKind::UnexpectedToken)?,
                                ]);
                                Ok(())
                            }
                            NEG..=NOT | CP..=SWA | NEGF..=FTI => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                );
                                self.buf.extend([r0, r1]);
                                Ok(())
                            }
                            LI | JMP => {
                                expect_matches!(self, Token::Register(r0), Token::PSep);
                                self.buf.push(r0);
                                self.insert_imm()?;
                                Ok(())
                            }
                            ADDI..=CMPUI | BMC | JEQ..=JGTU | ADDFI..=MULFI => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                    Token::PSep,
                                );
                                self.buf.extend([r0, r1]);
                                self.insert_imm()?;
                                Ok(())
                            }
                            LD..=ST => {
                                expect_matches!(
                                    self,
                                    Token::Register(r0),
                                    Token::PSep,
                                    Token::Register(r1),
                                    Token::PSep,
                                    Token::Integer(offset),
                                    Token::PSep,
                                    Token::Integer(len),
                                );
                                self.buf.extend([r0, r1]);
                                self.buf.extend(offset.to_le_bytes());
                                self.buf.extend(
                                    u16::try_from(len)
                                        .map_err(|_| ErrorKind::InvalidToken)?
                                        .to_le_bytes(),
                                );
                                Ok(())
                            }
                            _ => unreachable!(),
                        }?;
                        match self.next() {
                            Ok(Token::ISep) => (),
                            Ok(_) => return Err(ErrorKind::UnexpectedToken),
                            Err(ErrorKind::UnexpectedEnd) => return Ok(()),
                            Err(e) => return Err(e),
                        }
                    }
                    Some(Ok(Token::Label(lbl))) => {
                        self.label_map.insert(lbl, self.buf.len() as u64);
                    }
                    Some(Ok(Token::ISep)) => (),
                    Some(Ok(_)) => return Err(ErrorKind::UnexpectedToken),
                    Some(Err(())) => return Err(ErrorKind::InvalidToken),
                    None => return Ok(()),
                }
            }
        }

        fn link_local_syms(&mut self) -> Result<(), ErrorKind> {
            for (ix, sym) in &self.to_sub_label {
                self.label_map
                    .get(sym)
                    .ok_or(ErrorKind::InvalidSymbol)?
                    .to_le_bytes()
                    .iter()
                    .enumerate()
                    .for_each(|(i, b)| {
                        self.buf[ix + i] = *b;
                    });
            }

            Ok(())
        }

        fn insert_imm(&mut self) -> Result<(), ErrorKind> {
            let imm = match self.next()? {
                Token::Integer(i) => i.to_le_bytes(),
                Token::Symbol(s) => {
                    self.to_sub_label.insert(self.buf.len(), s);
                    [0; 8]
                }
                _ => return Err(ErrorKind::UnexpectedToken),
            };
            self.buf.extend(imm);
            Ok(())
        }
    }

    let mut asm = Assembler {
        lexer: Token::lexer(code),
        label_map: Default::default(),
        to_sub_label: Default::default(),
        buf,
    };

    asm.assemble().map_err(|kind| Error {
        kind,
        span: asm.lexer.span(),
    })?;

    asm.link_local_syms()
        .map_err(|kind| Error { kind, span: 0..0 })
}
