//! Macros to generate text-code assembler at [`crate::text`]
// Refering in module which generates a module to that module — is that even legal? :D

/// Generate text code based assembler
macro_rules! gen_text {
    (
        $(
            $ityn:ident
            ($($param_i:ident: $param_ty:ident),* $(,)?)
            => [$($opcode:ident),* $(,)?],
        )*
    ) => {
        /// Text code based assembler
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
                /// Assembly token
                #[derive(Clone, Copy, Debug, PartialEq, Eq, Logos)]
                #[logos(extras = Rodeo)]
                #[logos(skip r"[ \t\t]+")]
                #[logos(skip r"-- .*")]
                pub enum Token {
                    $($(#[token(~([<$opcode:lower>]), |_| hbbytecode::opcode::[<$opcode:upper>])])*)*
                    #[token("brc", |_| hbbytecode::opcode::BRC)] // Special-cased
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

            /// Type of error
            #[derive(Copy, Clone, Debug, PartialEq, Eq)]
            pub enum ErrorKind {
                UnexpectedToken,
                InvalidToken,
                UnexpectedEnd,
                InvalidSymbol,
            }

            /// Text assembly error
            #[derive(Clone, Debug, PartialEq, Eq)]
            pub struct Error {
                pub kind: ErrorKind,
                pub span: Span,
            }

            /// Parse code and insert instructions
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
                                // Got an opcode
                                Some(Ok(Token::Opcode(op))) => {
                                    match op {
                                        // Take all the opcodes and match them to their corresponding functions
                                        $(
                                            $(hbbytecode::opcode::$opcode)|* => paste::paste!({
                                                param_extract_itm!(self, $($param_i: $param_ty),*);
                                                self.asm.[<i_param_ $ityn>](op, $($param_i),*);
                                            }),
                                        )*
                                        // Special-cased
                                        hbbytecode::opcode::BRC => {
                                            param_extract_itm!(
                                                self,
                                                p0: R,
                                                p1: R,
                                                p2: u8
                                            );

                                            self.asm.i_param_bbb(op, p0, p1, p2);
                                        }
                                        // Already matched in Logos, should not be able to obtain
                                        // invalid opcode.
                                        _ => unreachable!(),
                                    }
                                }
                                // Insert label to table
                                Some(Ok(Token::Label(lbl))) => {
                                    self.symloc.insert(lbl, self.asm.buf.len());
                                }
                                // Instruction separator (LF, ;)
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

                // Walk table and substitute labels
                // for their addresses
                for &loc in &asm.asm.sub {
                    // Extract indices from the code and get addresses from table
                    let val = asm.symloc
                        .get(
                            &Spur::try_from_usize(bytemuck::pod_read_unaligned::<u64>(
                                &asm.asm.buf[loc..loc + core::mem::size_of::<u64>()]) as _
                            ).unwrap()
                        )
                        .ok_or(Error { kind: ErrorKind::InvalidSymbol, span: 0..0 })?
                        .to_le_bytes();

                    // New address
                    asm.asm.buf[loc..]
                        .iter_mut()
                        .zip(val)
                        .for_each(|(dst, src)| *dst = src);
                }

                Ok(())
            }

            // Fun fact: this is a little hack
            // It may slow the things a little bit down, but
            // it made the macro to be made pretty nice.
            // 
            // If you have any idea how to get rid of this,
            // contributions are welcome :)
            // I *likely* won't try anymore.
            enum InternalImm {
                Const(u64),
                Named(Spur),
            }

            impl $crate::Imm for InternalImm {
                #[inline]
                fn insert(&self, asm: &mut Assembler) {
                    match self {
                        // Constant immediate, just put it in
                        Self::Const(a) => a.insert(asm),
                        // Label
                        Self::Named(a) => {
                            // Insert to the sub table that substitution will be
                            // requested
                            asm.sub.insert(asm.buf.len());
                            // Insert value from interner in place
                            asm.buf.extend((a.into_usize() as u64).to_le_bytes());
                        },
                    }
                }
            }
        }
    };
}

/// Extract item by pattern, otherwise return [`ErrorKind::UnexpectedToken`]
macro_rules! extract_pat {
    ($self:expr, $pat:pat) => {
        let $pat = $self.next()?
            else { return Err(ErrorKind::UnexpectedToken) };
    };
}

/// Generate extract macro
macro_rules! gen_extract {
    // Integer types have same body
    ($($int:ident),* $(,)?) => {
        /// Extract operand from code
        macro_rules! extract {
            // Register (require prefixing with r)
            ($self:expr, R, $id:ident) => {
                extract_pat!($self, Token::Register($id));
            };

            ($self:expr, L, $id:ident) => {
                extract_pat!($self, Token::Integer($id));
                if $id > 2048 {
                    return Err(ErrorKind::InvalidToken);
                }

                let $id = u16::try_from($id).unwrap();
            };

            // Immediate
            ($self:expr, I, $id:ident) => {
                let $id = match $self.next()? {
                    // Either straight up integer
                    Token::Integer(a) => InternalImm::Const(a),
                    // …or a label
                    Token::Symbol(a) => InternalImm::Named(a),
                    _ => return Err(ErrorKind::UnexpectedToken),
                };
            };

            // Get $int, if not fitting, the token is claimed invalid
            $(($self:expr, $int, $id:ident) => {
                extract_pat!($self, Token::Integer($id));
                let $id = $int::try_from($id).map_err(|_| ErrorKind::InvalidToken)?;
            });*;
        }
    };
}

gen_extract!(u8, u16, u32);

/// Parameter extract incremental token-tree muncher
/// 
/// What else would it mean?
macro_rules! param_extract_itm {
    ($self:expr, $($id:ident: $ty:ident)? $(, $($tt:tt)*)?) => {
        // Extract pattern
        $(extract!($self, $ty, $id);)?
        $(
            // Require operand separator
            extract_pat!($self, Token::PSep);
            // And go to the next (recursive)
            // …munch munch… yummy token trees.
            param_extract_itm!($self, $($tt)*);
        )?
    };
}

pub(crate) use {extract, extract_pat, gen_text, param_extract_itm};
