macro_rules! gen_text {
    (
        $(
            $ityn:ident
            ($($param_i:ident: $param_ty:ident),* $(,)?)
            => [$($opcode:ident),* $(,)?],
        )*
    ) => {
        pub mod text {
            use {
                crate::{
                    Assembler,
                    macros::text::*,
                },
                hashbrown::HashMap,
                lasso::{Key, Rodeo, Spur},
                logos::{Lexer, Logos, Span},
            };

            paste::paste!(literify::literify! {
                #[derive(Clone, Copy, Debug, PartialEq, Eq, Logos)]
                #[logos(extras = Rodeo)]
                #[logos(skip r"[ \t\t]+")]
                #[logos(skip r"-- .*")]
                pub enum Token {
                    $($(#[token(~([<$opcode:lower>]), |_| hbbytecode::opcode::[<$opcode:upper>])])*)*
                    Opcode(u8),

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
            });

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

            pub fn assemble(asm: &mut Assembler, code: &str) -> Result<(), Error> {
                pub struct TextAsm<'a> {
                    asm: &'a mut Assembler,
                    lexer: Lexer<'a, Token>,
                    symloc: HashMap<Spur, usize>,
                }

                impl<'a> TextAsm<'a> {
                    fn next(&mut self) -> Result<Token, ErrorKind> {
                        match self.lexer.next() {
                            Some(Ok(t))   => Ok(t),
                            Some(Err(())) => Err(ErrorKind::InvalidToken),
                            None          => Err(ErrorKind::UnexpectedEnd),
                        }
                    }

                    #[inline(always)]
                    fn run(&mut self) -> Result<(), ErrorKind> {
                        loop {
                            match self.lexer.next() {
                                Some(Ok(Token::Opcode(op))) => {
                                    match op {
                                        $(
                                            $(hbbytecode::opcode::$opcode)|* => paste::paste!({
                                                param_extract_itm!(self, $($param_i: $param_ty),*);
                                                self.asm.[<i_param_ $ityn>](op, $($param_i),*);
                                            }),
                                        )*
                                        _ => unreachable!(),
                                    }
                                }
                                Some(Ok(Token::Label(lbl))) => {
                                    self.symloc.insert(lbl, self.asm.buf.len());
                                }
                                Some(Ok(Token::ISep)) => (),
                                Some(Ok(_))           => return Err(ErrorKind::UnexpectedToken),
                                Some(Err(()))         => return Err(ErrorKind::InvalidToken),
                                None                  => return Ok(()),
                            }
                        }
                    }
                }

                let mut asm = TextAsm {
                    asm,
                    lexer: Token::lexer(code),
                    symloc: HashMap::default(),
                };

                asm.run()
                    .map_err(|kind| Error { kind, span: asm.lexer.span() })?;
                
                for &loc in &asm.asm.sub {
                    let val = asm.symloc
                        .get(
                            &Spur::try_from_usize(bytemuck::pod_read_unaligned::<u64>(&asm.asm.buf[loc..loc+core::mem::size_of::<u64>()]) as _)
                                .unwrap()
                        )
                        .ok_or(Error { kind: ErrorKind::InvalidSymbol, span: 0..0 })?
                        .to_le_bytes();

                    asm.asm.buf[loc..]
                        .iter_mut()
                        .zip(val)
                        .for_each(|(dst, src)| *dst = src);
                }

                Ok(())
            }

            enum InternalImm {
                Const(u64),
                Named(Spur),
            }

            impl $crate::Imm for InternalImm {
                #[inline]
                fn insert(&self, asm: &mut Assembler) {
                    match self {
                        Self::Const(a) => a.insert(asm),
                        Self::Named(a) => {
                            asm.sub.insert(asm.buf.len());
                            asm.buf.extend((a.into_usize() as u64).to_le_bytes());
                        },
                    }
                }
            }
        }
    };
}

macro_rules! extract_pat {
    ($self:expr, $pat:pat) => {
        let $pat = $self.next()?
            else { return Err(ErrorKind::UnexpectedToken) };
    };
}

macro_rules! extract {
    ($self:expr, R, $id:ident) => {
        extract_pat!($self, Token::Register($id));
    };

    ($self:expr, I, $id:ident) => {
        let $id = match $self.next()? {
            Token::Integer(a) => InternalImm::Const(a),
            Token::Symbol(a) => InternalImm::Named(a),
            _ => return Err(ErrorKind::UnexpectedToken),
        };
    };

    ($self:expr, u16, $id:ident) => {
        extract_pat!($self, Token::Integer($id));
        let $id = u16::try_from($id).map_err(|_| ErrorKind::InvalidToken)?;
    };
}

macro_rules! param_extract_itm {
    ($self:expr, $($id:ident: $ty:ident)? $(, $($tt:tt)*)?) => {
        $(extract!($self, $ty, $id);)?
        $(
            extract_pat!($self, Token::PSep);
            param_extract_itm!($self, $($tt)*);
        )?
    };
}

pub(crate) use {extract, extract_pat, gen_text, param_extract_itm};
