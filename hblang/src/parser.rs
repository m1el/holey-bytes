use std::{cell::Cell, ops::Not};

use crate::lexer::{Lexer, Token, TokenKind};

type Ptr<T> = &'static T;

fn ptr<T>(val: T) -> Ptr<T> {
    Box::leak(Box::new(val))
}

pub struct Parser<'a> {
    path:  &'a std::path::Path,
    lexer: Lexer<'a>,
    token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str, path: &'a std::path::Path) -> Self {
        let mut lexer = Lexer::new(input);
        let token = lexer.next();
        Self { lexer, token, path }
    }

    fn next(&mut self) -> Token {
        std::mem::replace(&mut self.token, self.lexer.next())
    }

    pub fn file(&mut self) -> Vec<Expr> {
        std::iter::from_fn(|| (self.token.kind != TokenKind::Eof).then(|| self.expr())).collect()
    }

    fn ptr_expr(&mut self) -> Ptr<Expr> {
        ptr(self.expr())
    }

    pub fn expr(&mut self) -> Expr {
        let token = self.next();
        let expr = match token.kind {
            TokenKind::Ident => {
                let name = self.lexer.slice(token).to_owned().leak();
                if self.advance_if(TokenKind::Decl) {
                    let val = self.ptr_expr();
                    Expr::Decl { name, val }
                } else {
                    Expr::Ident { name }
                }
            }
            TokenKind::Return => Expr::Return {
                val: (self.token.kind != TokenKind::Semi).then(|| self.ptr_expr()),
            },
            TokenKind::Or => {
                self.expect_advance(TokenKind::Colon);
                let ret = self.ptr_expr();
                let body = self.ptr_expr();
                Expr::Closure { ret, body }
            }
            TokenKind::LBrace => Expr::Block {
                stmts: std::iter::from_fn(|| {
                    self.advance_if(TokenKind::RBrace)
                        .not()
                        .then(|| self.expr())
                })
                .collect::<Vec<_>>(),
            },
            TokenKind::Number => Expr::Number {
                value: match self.lexer.slice(token).parse() {
                    Ok(value) => value,
                    Err(e) => self.report(format_args!("invalid number: {e}")),
                },
            },
            tok => self.report(format_args!("unexpected token: {:?}", tok)),
        };

        self.advance_if(TokenKind::Semi);

        expr
    }

    fn advance_if(&mut self, kind: TokenKind) -> bool {
        if self.token.kind == kind {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_advance(&mut self, kind: TokenKind) {
        if self.token.kind != kind {
            self.report(format_args!(
                "expected {:?}, found {:?}",
                kind, self.token.kind
            ));
        }
        self.next();
    }

    fn report(&self, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.lexer.line_col(self.token.start);
        eprintln!("{}:{}:{} => {}", self.path.display(), line, col, msg);
        unreachable!();
    }
}

#[derive(Debug)]
pub enum Expr {
    Decl { name: Ptr<str>, val: Ptr<Expr> },
    Closure { ret: Ptr<Expr>, body: Ptr<Expr> },
    Return { val: Option<Ptr<Expr>> },
    Ident { name: Ptr<str> },
    Block { stmts: Vec<Expr> },
    Number { value: u64 },
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = Cell::new(0);
        }

        match self {
            Self::Decl { name, val } => write!(f, "{} := {}", name, val),
            Self::Closure { ret, body } => write!(f, "||: {} {}", ret, body),
            Self::Return { val: Some(val) } => write!(f, "return {};", val),
            Self::Return { val: None } => write!(f, "return;"),
            Self::Ident { name } => write!(f, "{}", name),
            Self::Block { stmts } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| i.set(i.get() + 1));
                let res = crate::try_block(|| {
                    for stmt in stmts {
                        for _ in 0..INDENT.with(|i| i.get()) {
                            write!(f, "    ")?;
                        }
                        writeln!(f, "{}", stmt)?;
                    }
                    Ok(())
                });
                INDENT.with(|i| i.set(i.get() - 1));
                write!(f, "}}")?;
                res
            }
            Self::Number { value } => write!(f, "{}", value),
        }
    }
}

#[cfg(test)]
mod tests {
    fn parse(input: &'static str, output: &mut String) {
        use std::fmt::Write;
        let mut parser = super::Parser::new(input, std::path::Path::new("test"));
        for expr in parser.file() {
            writeln!(output, "{}", expr).unwrap();
        }
    }

    crate::run_tests! { parse:
        example => include_str!("../examples/main_fn.hb");
    }
}
