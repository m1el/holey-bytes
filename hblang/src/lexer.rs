use std::{iter::Peekable, str::Chars};

#[derive(Debug, PartialEq)]
pub struct Token {
    pub kind:  TokenKind,
    pub start: u32,
    pub end:   u32,
}

impl Token {
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start as usize..self.end as usize
    }
}

#[derive(Debug, PartialEq)]
pub enum TokenKind {
    Ident,
    Number,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBrack,
    RBrack,
    Decl,
    Or,
    Semi,
    Colon,
    Return,
    Eof,
    Error,
}

pub struct Lexer<'a> {
    pos:   u32,
    bytes: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            pos:   0,
            bytes: input.as_bytes(),
        }
    }

    pub fn slice(&self, tok: Token) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[tok.range()]) }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos as usize).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    pub fn next(&mut self) -> Token {
        Iterator::next(self).unwrap_or(Token {
            kind:  TokenKind::Eof,
            start: self.pos,
            end:   self.pos,
        })
    }

    fn advance_if(&mut self, arg: u8) -> bool {
        if self.peek() == Some(arg) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn line_col(&self, mut start: u32) -> (usize, usize) {
        self.bytes
            .split(|&b| b == b'\n')
            .enumerate()
            .find_map(|(i, line)| {
                if start < line.len() as u32 {
                    return Some((i + 1, start as usize + 1));
                }
                start -= line.len() as u32 + 1;
                None
            })
            .unwrap_or((1, 1))
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        use TokenKind as T;
        loop {
            let start = self.pos;
            let kind = match self.advance()? {
                b'\n' | b'\r' | b'\t' | b' ' => continue,
                b'0'..=b'9' => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = self.peek() {
                        self.advance();
                    }

                    let ident = &self.bytes[start as usize..self.pos as usize];
                    match ident {
                        b"return" => T::Return,
                        _ => T::Ident,
                    }
                }
                b':' => match self.advance_if(b'=') {
                    true => T::Decl,
                    false => T::Colon,
                },
                b';' => T::Semi,
                b'|' => match self.advance_if(b'|') {
                    true => T::Or,
                    false => T::Error,
                },
                b'(' => T::LParen,
                b')' => T::RParen,
                b'{' => T::LBrace,
                b'}' => T::RBrace,
                b'[' => T::LBrack,
                b']' => T::RBrack,
                _ => T::Error,
            };

            return Some(Token {
                kind,
                start,
                end: self.pos,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    fn lex(input: &'static str, output: &mut String) {
        use {
            super::{Lexer, TokenKind as T},
            std::fmt::Write,
        };
        let mut lexer = Lexer::new(input);
        loop {
            let token = lexer.next();
            writeln!(output, "{:?} {:?}", token.kind, &input[token.range()],).unwrap();
            if token.kind == T::Eof {
                break;
            }
        }
    }

    crate::run_tests! { lex:
        empty => "";
        whitespace => " \t\n\r";
        examples => include_str!("../examples/main_fn.hb");
    }
}
