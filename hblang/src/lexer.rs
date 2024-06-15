use std::simd::cmp::SimdPartialEq;

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
                #[$prec:ident] $(
                    $op:ident = $op_lit:literal $(=> $assign:ident)?,
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
                    $($( Self::$op    => $op_lit,
                      $(Self::$assign => concat!($op_lit, "="),)?)*)*
                };
                f.write_str(s)
            }
        }

        impl $name {
            #[inline(always)]
            pub fn precedence(&self) -> Option<u8> {
                Some(match self {
                    $($(Self::$op => ${ignore($prec)} ${index(1)},
                      $(Self::$assign => 0,)?)*)*
                    _ => return None,
                } + 1)
            }

            #[inline(always)]
            fn from_ident(ident: &[u8]) -> Self {
                match ident {
                    $($keyword_lit => Self::$keyword,)*
                    _ => Self::Ident,
                }
            }

            pub fn assign_op(&self) -> Option<Self> {
                Some(match self {
                    $($($(Self::$assign => Self::$op,)?)*)*
                    _ => return None,
                })
            }
        }

        #[derive(Debug, PartialEq, Eq, Clone, Copy)]
        $vis enum $name {
            $( $pattern, )*
            $( $keyword, )*
            $( $punkt,   )*
            $($( $op, $($assign,)?  )*)*
        }
    };
}

gen_token_kind! {
    pub enum TokenKind {
        #[patterns]
        CtIdent,
        Ident,
        Number,
        Eof,
        Error,
        Driective,
        String,
        #[keywords]
        Return   = b"return",
        If       = b"if",
        Else     = b"else",
        Loop     = b"loop",
        Break    = b"break",
        Continue = b"continue",
        Fn       = b"fn",
        Struct   = b"struct",
        True     = b"true",
        #[punkt]
        LParen = "(",
        RParen = ")",
        LBrace = "{",
        RBrace = "}",
        Semi   = ";",
        Colon  = ":",
        Comma  = ",",
        Dot    = ".",
        Ctor   = ".{",
        Tupl   = ".(",
        #[ops]
        #[prec]
        Decl   = ":=",
        Assign = "=",
        #[prec]
        Or = "||",
        #[prec]
        And = "&&",
        #[prec]
        Bor = "|" => BorAss,
        #[prec]
        Xor = "^" => XorAss,
        #[prec]
        Band = "&" => BandAss,
        #[prec]
        Eq = "==",
        Ne = "!=",
        #[prec]
        Le = "<=",
        Ge = ">=",
        Lt = "<",
        Gt = ">",
        #[prec]
        Shl = "<<" => ShlAss,
        Shr = ">>" => ShrAss,
        #[prec]
        Add = "+" => AddAss,
        Sub = "-" => SubAss,
        #[prec]
        Mul = "*" => MulAss,
        Div = "/" => DivAss,
        Mod = "%" => ModAss,
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
        use TokenKind as T;
        loop {
            let mut start = self.pos;

            let Some(c) = self.advance() else {
                return Token {
                    kind: T::Eof,
                    start,
                    end: self.pos,
                };
            };

            let advance_ident = |s: &mut Self| {
                while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = s.peek() {
                    s.advance();
                }
            };

            let kind = match c {
                b'\n' | b'\r' | b'\t' | b' ' => continue,
                b'0'..=b'9' => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'@' => {
                    start += 1;
                    advance_ident(self);
                    T::Driective
                }
                b'$' => {
                    start += 1;
                    advance_ident(self);
                    T::CtIdent
                }
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                    advance_ident(self);
                    let ident = &self.bytes[start as usize..self.pos as usize];
                    T::from_ident(ident)
                }
                b'"' => {
                    while let Some(c) = self.advance() {
                        match c {
                            b'"' => break,
                            b'\\' => _ = self.advance(),
                            _ => {}
                        }
                    }
                    T::String
                }
                b':' if self.advance_if(b'=') => T::Decl,
                b':' => T::Colon,
                b',' => T::Comma,
                b'.' if self.advance_if(b'{') => T::Ctor,
                b'.' if self.advance_if(b'(') => T::Tupl,
                b'.' => T::Dot,
                b';' => T::Semi,
                b'!' if self.advance_if(b'=') => T::Ne,
                b'=' if self.advance_if(b'=') => T::Eq,
                b'=' => T::Assign,
                b'<' if self.advance_if(b'=') => T::Le,
                b'<' if self.advance_if(b'<') => match self.advance_if(b'=') {
                    true => T::ShlAss,
                    false => T::Shl,
                },
                b'<' => T::Lt,
                b'>' if self.advance_if(b'=') => T::Ge,
                b'>' if self.advance_if(b'>') => match self.advance_if(b'=') {
                    true => T::ShrAss,
                    false => T::Shr,
                },
                b'>' => T::Gt,
                b'+' if self.advance_if(b'=') => T::AddAss,
                b'+' => T::Add,
                b'-' if self.advance_if(b'=') => T::SubAss,
                b'-' => T::Sub,
                b'*' if self.advance_if(b'=') => T::MulAss,
                b'*' => T::Mul,
                b'/' if self.advance_if(b'=') => T::DivAss,
                b'/' => T::Div,
                b'%' if self.advance_if(b'=') => T::ModAss,
                b'%' => T::Mod,
                b'&' if self.advance_if(b'=') => T::BandAss,
                b'&' if self.advance_if(b'&') => T::And,
                b'&' => T::Band,
                b'^' if self.advance_if(b'=') => T::XorAss,
                b'^' => T::Xor,
                b'|' if self.advance_if(b'=') => T::BorAss,
                b'|' if self.advance_if(b'|') => T::Or,
                b'|' => T::Bor,
                b'(' => T::LParen,
                b')' => T::RParen,
                b'{' => T::LBrace,
                b'}' => T::RBrace,
                _ => T::Error,
            };

            return Token {
                kind,
                start,
                end: self.pos,
            };
        }
    }

    fn advance_if(&mut self, arg: u8) -> bool {
        if self.peek() == Some(arg) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn line_col(&self, pos: u32) -> (usize, usize) {
        line_col(self.bytes, pos)
    }
}

pub fn line_col(bytes: &[u8], pos: u32) -> (usize, usize) {
    bytes[..pos as usize]
        .split(|&b| b == b'\n')
        .map(<[u8]>::len)
        .enumerate()
        .last()
        .map(|(line, col)| (line + 1, col + 1))
        .unwrap_or((1, 1))
}

pub struct LineMap {
    lines: Box<[u8]>,
}

impl LineMap {
    pub fn line_col(&self, mut pos: u32) -> (usize, usize) {
        let mut line = 1;

        let mut iter = self.lines.iter().copied();

        while let Some(mut len) = iter.next() {
            let mut acc = 0;
            while len & 0x80 != 0 {
                acc = (acc << 7) | (len & 0x7F) as u32;
                len = iter.next().unwrap();
            }
            acc += len as u32;

            if pos < acc {
                break;
            }
            pos = pos.saturating_sub(acc);
            line += 1;
        }

        (line, pos as usize + 1)
    }

    pub fn new(input: &str) -> Self {
        let bytes = input.as_bytes();
        let (start, simd_mid, end) = bytes.as_simd::<16>();

        let query = std::simd::u8x16::splat(b'\n');

        let nl_count = start.iter().map(|&b| (b == b'\n') as usize).sum::<usize>()
            + simd_mid
                .iter()
                .map(|s| s.simd_eq(query).to_bitmask().count_ones())
                .sum::<u32>() as usize
            + end.iter().map(|&b| (b == b'\n') as usize).sum::<usize>();

        let mut lines = Vec::with_capacity(nl_count);
        let mut last_nl = 0;

        let handle_rem = |offset: usize, bytes: &[u8], last_nl: &mut usize, lines: &mut Vec<u8>| {
            bytes
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, b)| (b == b'\n').then_some(i + offset))
                .for_each(|i| {
                    lines.push((i - *last_nl + 1) as u8);
                    *last_nl = i + 1;
                });
        };

        handle_rem(0, start, &mut last_nl, &mut lines);

        for (i, simd) in simd_mid.iter().enumerate() {
            let mask = simd.simd_eq(query);
            let mut mask = mask.to_bitmask();
            while mask != 0 {
                let idx = mask.trailing_zeros() as usize + i * 16 + start.len();
                let mut len = idx - last_nl + 1;
                while len >= 0x80 {
                    lines.push((0x80 | (len & 0x7F)) as u8);
                    len >>= 7;
                }
                lines.push(len as u8);
                last_nl = idx + 1;
                mask &= mask - 1;
            }
        }

        handle_rem(bytes.len() - end.len(), end, &mut last_nl, &mut lines);

        Self {
            lines: Box::from(lines),
        }
    }
}
