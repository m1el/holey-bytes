use std::simd::cmp::SimdPartialEq;

const fn ascii_mask(chars: &[u8]) -> u128 {
    let mut eq = 0;
    let mut i = 0;
    while i < chars.len() {
        let b = chars[i];
        eq |= 1 << b;
        i += 1;
    }
    eq
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Token {
    pub kind: TokenKind,
    pub start: u32,
    pub end: u32,
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
                let sf = *self as u8;
                f.write_str(match *self {
                    $( Self::$pattern => concat!('<', stringify!($pattern), '>'), )*
                    $( Self::$keyword => stringify!($keyword_lit), )*
                    $( Self::$punkt   => stringify!($punkt_lit),   )*
                    $($( Self::$op    => $op_lit,
                      $(Self::$assign => concat!($op_lit, "="),)?)*)*
                    _ => unsafe { std::str::from_utf8_unchecked(std::slice::from_ref(&sf)) },
                })
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
        }
    };
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum TokenKind {
    Not = b'!',
    DQuote = b'"',
    Pound = b'#',
    CtIdent = b'$',
    Mod = b'%',
    Band = b'&',
    Quote = b'\'',
    LParen = b'(',
    RParen = b')',
    Mul = b'*',
    Add = b'+',
    Comma = b',',
    Sub = b'-',
    Dot = b'.',
    Div = b'/',
    // Unused = 2-6
    Shl = b'<' - 5,
    // Unused = 8
    Shr = b'>' - 5,
    Colon = b':',
    Semi = b';',
    Lt = b'<',
    Assign = b'=',
    Gt = b'>',
    Que = b'?',
    Directive = b'@',

    Comment,

    Ident,
    Number,
    Eof,

    Ct,

    Return,
    If,
    Else,
    Loop,
    Break,
    Continue,
    Fn,
    Struct,
    True,

    Ctor,
    Tupl,

    Or,
    And,

    // Unused = R-Z
    LBrack = b'[',
    BSlash = b'\\',
    RBrack = b']',
    Xor = b'^',
    Tick = b'`',
    // Unused = a-z
    LBrace = b'{',
    Bor = b'|',
    RBrace = b'}',
    Tilde = b'~',

    Decl = b':' + 128,
    Eq = b'=' + 128,
    Ne = b'!' + 128,
    Le = b'<' + 128,
    Ge = b'>' + 128,

    BorAss = b'|' + 128,
    AddAss = b'+' + 128,
    SubAss = b'-' + 128,
    MulAss = b'*' + 128,
    DivAss = b'/' + 128,
    ModAss = b'%' + 128,
    XorAss = b'^' + 128,
    BandAss = b'&' + 128,
    ShrAss = b'>' - 5 + 128,
    ShlAss = b'<' - 5 + 128,
}

impl TokenKind {
    pub fn ass_op(self) -> Option<Self> {
        let id = (self as u8).saturating_sub(128);
        if ascii_mask(b"|+-*/%^&79") & (1u128 << id) == 0 {
            return None;
        }
        Some(unsafe { std::mem::transmute::<u8, Self>(id) })
    }
}

gen_token_kind! {
    pub enum TokenKind {
        #[patterns]
        CtIdent,
        Ident,
        Number,
        Eof,
        Directive,
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
    pos: u32,
    bytes: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self::restore(input, 0)
    }

    pub fn restore(input: &'a str, pos: u32) -> Self {
        Self { pos, bytes: input.as_bytes() }
    }

    pub fn slice(&self, tok: std::ops::Range<usize>) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[tok]) }
    }

    fn peek(&self) -> Option<u8> {
        if std::intrinsics::unlikely(self.pos >= self.bytes.len() as u32) {
            None
        } else {
            Some(unsafe { *self.bytes.get_unchecked(self.pos as usize) })
        }
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    pub fn last(&mut self) -> Token {
        let mut token = self.next();
        loop {
            let next = self.next();
            if next.kind == TokenKind::Eof {
                break;
            }
            token = next;
        }
        token
    }

    pub fn next(&mut self) -> Token {
        use TokenKind as T;
        loop {
            let mut start = self.pos;

            let Some(c) = self.advance() else {
                return Token { kind: T::Eof, start, end: self.pos };
            };

            let advance_ident = |s: &mut Self| {
                while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | 127..) = s.peek() {
                    s.advance();
                }
            };

            let identity = |s: u8| unsafe { std::mem::transmute::<u8, T>(s) };

            let kind = match c {
                ..=b' ' => continue,
                b'0' if self.advance_if(b'x') => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0' if self.advance_if(b'b') => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0' if self.advance_if(b'o') => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0'..=b'9' => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'a'..=b'z' | b'A'..=b'Z' | b'_' | 127.. => {
                    advance_ident(self);
                    let ident = &self.bytes[start as usize..self.pos as usize];
                    T::from_ident(ident)
                }
                b'"' | b'\'' => loop {
                    match self.advance() {
                        Some(b'\\') => _ = self.advance(),
                        Some(nc) if nc == c => break identity(c),
                        Some(_) => {}
                        None => break T::Eof,
                    }
                },
                b'/' if self.advance_if(b'/') => {
                    while let Some(l) = self.advance()
                        && l != b'\n'
                    {}
                    T::Comment
                }
                b'/' if self.advance_if(b'*') => {
                    let mut depth = 1;
                    while let Some(l) = self.advance() {
                        match l {
                            b'/' if self.advance_if(b'*') => depth += 1,
                            b'*' if self.advance_if(b'/') => match depth {
                                1 => break,
                                _ => depth -= 1,
                            },
                            _ => {}
                        }
                    }
                    T::Comment
                }
                b'.' if self.advance_if(b'{') => T::Ctor,
                b'.' if self.advance_if(b'(') => T::Tupl,
                b'&' if self.advance_if(b'&') => T::And,
                b'|' if self.advance_if(b'|') => T::Or,
                b'$' if self.advance_if(b':') => T::Ct,
                b'@' | b'$' => {
                    start += 1;
                    advance_ident(self);
                    identity(c)
                }
                b'<' | b'>' if self.advance_if(c) => {
                    identity(c - 5 + 128 * self.advance_if(b'=') as u8)
                }
                b':' | b'=' | b'!' | b'<' | b'>' | b'|' | b'+' | b'-' | b'*' | b'/' | b'%'
                | b'^' | b'&'
                    if self.advance_if(b'=') =>
                {
                    identity(c + 128)
                }
                _ => identity(c),
            };

            return Token { kind, start, end: self.pos };
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

        loop {
            let mut acc = 0;
            let mut idx = 0;
            loop {
                let len = iter.next().unwrap();
                acc |= ((len & 0x7F) as u32) << (7 * idx);
                idx += 1;
                if len & 0x80 == 0 {
                    break;
                }
            }

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
            + simd_mid.iter().map(|s| s.simd_eq(query).to_bitmask().count_ones()).sum::<u32>()
                as usize
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
                    lines.push(0x80 | (len & 0x7F) as u8);
                    len >>= 7;
                }
                lines.push(len as u8);
                last_nl = idx + 1;
                mask &= mask - 1;
            }
        }

        handle_rem(bytes.len() - end.len(), end, &mut last_nl, &mut lines);

        Self { lines: Box::from(lines) }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_smh() {
        let example = include_str!("../README.md");

        let nlines = super::LineMap::new(example);

        fn slow_nline_search(str: &str, mut pos: usize) -> (usize, usize) {
            (
                str.lines()
                    .take_while(|l| match pos.checked_sub(l.len() + 1) {
                        Some(nl) => (pos = nl, true).1,
                        None => false,
                    })
                    .count()
                    + 1,
                pos + 1,
            )
        }

        for i in 0..example.len() {
            assert_eq!(slow_nline_search(example, i), nlines.line_col(i as _));
        }
    }
}
