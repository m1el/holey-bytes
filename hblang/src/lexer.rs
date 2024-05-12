#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Token {
    pub kind:  TokenKind,
    pub start: u32,
    pub end:   u32,
}

impl Token {
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start as usize..self.end as usize
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }
}

macro_rules! gen_token_kind {
    ($(
        #[$atts:meta])*
        $vis:vis enum $name:ident {
            #[patterns] $(
                $pattern:ident,
            )*
            #[keywords] $(
                $keyword:ident = $keyword_lit:literal,
            )*
            #[punkt] $(
                $punkt:ident = $punkt_lit:literal,
            )*
            #[ops] $(
                #[prec = $prec:literal] $(
                    $op:ident = $op_lit:literal,
                )*
            )*
        }
    ) => {
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let s = match *self {
                    $( Self::$pattern => concat!('<', stringify!($pattern), '>'), )*

                    $( Self::$keyword => stringify!($keyword_lit), )*
                    $( Self::$punkt   => stringify!($punkt_lit),   )*
                    $($( Self::$op    => $op_lit,                )*)*
                };
                f.write_str(s)
            }
        }

        impl $name {
            #[inline(always)]
            pub fn precedence(&self) -> Option<u8> {
                Some(match self {
                    $($(Self::$op)|* => $prec,)*
                    _ => return None,
                })
            }

            #[inline(always)]
            fn from_ident(ident: &[u8]) -> Self {
                match ident {
                    $($keyword_lit => Self::$keyword,)*
                    _ => Self::Ident,
                }
            }
        }

        #[derive(Debug, PartialEq, Eq, Clone, Copy)]
        $vis enum $name {
            $( $pattern, )*
            $( $keyword, )*
            $( $punkt,   )*
            $($( $op,  )*)*
        }
    };
}

gen_token_kind! {
    pub enum TokenKind {
        #[patterns]
        Ident,
        Number,
        Eof,
        Error,
        #[keywords]
        Return   = b"return",
        If       = b"if",
        Else     = b"else",
        Loop     = b"loop",
        Break    = b"break",
        Continue = b"continue",
        Fn       = b"fn",
        #[punkt]
        LParen = b'(',
        RParen = b')',
        LBrace = b'{',
        RBrace = b'}',
        Semi =   b';',
        Colon =  b':',
        Comma =  b',',
        #[ops]
        #[prec = 1]
        Decl =   ":=",
        Assign = "=",
        #[prec = 21]
        Le = "<=",
        Eq = "==",
        #[prec = 22]
        Amp = "&",
        #[prec = 23]
        Plus =  "+",
        Minus = "-",
        #[prec = 24]
        Star =   "*",
        FSlash = "/",
    }
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

    pub fn slice(&self, tok: std::ops::Range<usize>) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[tok]) }
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
                    T::from_ident(ident)
                }
                b':' if self.advance_if(b'=') => T::Decl,
                b':' => T::Colon,
                b',' => T::Comma,
                b';' => T::Semi,
                b'=' if self.advance_if(b'=') => T::Eq,
                b'=' => T::Assign,
                b'<' if self.advance_if(b'=') => T::Le,
                b'+' => T::Plus,
                b'-' => T::Minus,
                b'*' => T::Star,
                b'/' => T::FSlash,
                b'&' => T::Amp,
                b'(' => T::LParen,
                b')' => T::RParen,
                b'{' => T::LBrace,
                b'}' => T::RBrace,
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
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
    }
}
