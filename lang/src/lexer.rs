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
    pub fn range(&self) -> core::ops::Range<usize> {
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
        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.write_str(self.name())
            }
        }

        impl $name {
            pub const OPS: &[Self] = &[$($(Self::$op),*),*];

            pub fn name(&self) -> &str {
                let sf = unsafe { &*(self as *const _ as *const u8) } ;
                match *self {
                    $( Self::$pattern => concat!('<', stringify!($pattern), '>'), )*
                    $( Self::$keyword => stringify!($keyword_lit), )*
                    $( Self::$punkt   => stringify!($punkt_lit),   )*
                    $($( Self::$op    => $op_lit,
                      $(Self::$assign => concat!($op_lit, "="),)?)*)*
                    _ => unsafe { core::str::from_utf8_unchecked(core::slice::from_ref(&sf)) },
                }
            }

            #[inline(always)]
            pub fn precedence(&self) -> Option<u8> {
                Some(match self {
                    $($(Self::$op => ${ignore($prec)} ${index(1)},
                      $(Self::$assign => 0,)?)*)*
                    _ => return None,
                } + 1)
            }

            fn from_ident(ident: &[u8]) -> Self {
                match ident {
                    $($keyword_lit => Self::$keyword,)*
                    _ => Self::Ident,
                }
            }
        }
    };
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
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
    Float,
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
    Packed,
    True,
    False,
    Null,
    Idk,
    Die,

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
    Under = b'_',
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

impl core::fmt::Debug for TokenKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

impl TokenKind {
    pub fn ass_op(self) -> Option<Self> {
        let id = (self as u8).saturating_sub(128);
        if ascii_mask(b"|+-*/%^&79") & (1u128 << id) == 0 {
            return None;
        }
        Some(unsafe { core::mem::transmute::<u8, Self>(id) })
    }

    pub fn is_comutative(self) -> bool {
        use TokenKind as S;
        matches!(self, S::Eq | S::Ne | S::Bor | S::Xor | S::Band | S::Add | S::Mul)
    }

    pub fn is_supported_float_op(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Eq
                | Self::Ne
                | Self::Le
                | Self::Ge
                | Self::Lt
                | Self::Gt
        )
    }

    pub fn apply_binop(self, a: i64, b: i64, float: bool) -> i64 {
        if float {
            debug_assert!(self.is_supported_float_op());
            let [a, b] = [f64::from_bits(a as _), f64::from_bits(b as _)];
            let res = match self {
                Self::Add => a + b,
                Self::Sub => a - b,
                Self::Mul => a * b,
                Self::Div => a / b,
                Self::Eq => return (a == b) as i64,
                Self::Ne => return (a != b) as i64,
                Self::Lt => return (a < b) as i64,
                Self::Gt => return (a > b) as i64,
                Self::Le => return (a >= b) as i64,
                Self::Ge => return (a <= b) as i64,
                _ => todo!("floating point op: {self}"),
            };

            return res.to_bits() as _;
        }

        match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Div if b == 0 => 0,
            Self::Div => a.wrapping_div(b),
            Self::Shl => a.wrapping_shl(b as _),
            Self::Eq => (a == b) as i64,
            Self::Ne => (a != b) as i64,
            Self::Lt => (a < b) as i64,
            Self::Gt => (a > b) as i64,
            Self::Le => (a >= b) as i64,
            Self::Ge => (a <= b) as i64,
            Self::Band => a & b,
            Self::Bor => a | b,
            Self::Xor => a ^ b,
            Self::Mod if b == 0 => 0,
            Self::Mod => a.wrapping_rem(b),
            Self::Shr => a.wrapping_shr(b as _),
            s => todo!("{s}"),
        }
    }

    pub fn is_homogenous(&self) -> bool {
        self.precedence() != Self::Eq.precedence()
            && self.precedence() != Self::Gt.precedence()
            && self.precedence() != Self::Eof.precedence()
    }

    pub fn apply_unop(&self, value: i64, float: bool) -> i64 {
        match self {
            Self::Sub if float => (-f64::from_bits(value as _)).to_bits() as _,
            Self::Sub => value.wrapping_neg(),
            Self::Not => (value == 0) as _,
            Self::Float if float => value,
            Self::Float => (value as f64).to_bits() as _,
            Self::Number if float => f64::from_bits(value as _) as _,
            Self::Number => value,
            s => todo!("{s}"),
        }
    }

    pub fn closing(&self) -> Option<TokenKind> {
        Some(match self {
            Self::Ctor => Self::RBrace,
            Self::Tupl => Self::RParen,
            Self::LParen => Self::RParen,
            Self::LBrack => Self::RBrack,
            Self::LBrace => Self::RBrace,
            _ => return None,
        })
    }
}

gen_token_kind! {
    pub enum TokenKind {
        #[patterns]
        CtIdent,
        Ident,
        Number,
        Float,
        Eof,
        Directive,
        #[keywords]
        Return    = b"return",
        If        = b"if",
        Else      = b"else",
        Loop      = b"loop",
        Break     = b"break",
        Continue  = b"continue",
        Fn        = b"fn",
        Struct    = b"struct",
        Packed    = b"packed",
        True      = b"true",
        False     = b"false",
        Null      = b"null",
        Idk       = b"idk",
        Die       = b"die",
        Under     = b"_",
        #[punkt]
        Ctor   = ".{",
        Tupl   = ".(",
        // #define OP: each `#[prec]` delimeters a level of precedence from lowest to highest
        #[ops]
        #[prec]
        // this also includess all `<op>=` tokens
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
    source: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self::restore(input, 0)
    }

    pub fn uses(input: &'a str) -> impl Iterator<Item = &'a str> {
        let mut s = Self::new(input);
        core::iter::from_fn(move || loop {
            let t = s.eat();
            if t.kind == TokenKind::Eof {
                return None;
            }
            if t.kind == TokenKind::Directive
                && s.slice(t.range()) == "use"
                && s.eat().kind == TokenKind::LParen
            {
                let t = s.eat();
                if t.kind == TokenKind::DQuote {
                    return Some(&s.slice(t.range())[1..t.range().len() - 1]);
                }
            }
        })
    }

    pub fn restore(input: &'a str, pos: u32) -> Self {
        Self { pos, source: input.as_bytes() }
    }

    pub fn source(&self) -> &'a str {
        unsafe { core::str::from_utf8_unchecked(self.source) }
    }

    pub fn slice(&self, tok: core::ops::Range<usize>) -> &'a str {
        unsafe { core::str::from_utf8_unchecked(&self.source[tok]) }
    }

    fn peek(&self) -> Option<u8> {
        if core::intrinsics::unlikely(self.pos >= self.source.len() as u32) {
            None
        } else {
            Some(unsafe { *self.source.get_unchecked(self.pos as usize) })
        }
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    pub fn last(&mut self) -> Token {
        let mut token = self.eat();
        loop {
            let next = self.eat();
            if next.kind == TokenKind::Eof {
                break;
            }
            token = next;
        }
        token
    }

    pub fn eat(&mut self) -> Token {
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

            let identity = |s: u8| unsafe { core::mem::transmute::<u8, T>(s) };

            let kind = match c {
                ..=b' ' => continue,
                b'0' if self.advance_if(b'x') => {
                    while let Some(b'0'..=b'9' | b'A'..=b'F' | b'a'..=b'f') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0' if self.advance_if(b'b') => {
                    while let Some(b'0' | b'1') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0' if self.advance_if(b'o') => {
                    while let Some(b'0'..=b'7') = self.peek() {
                        self.advance();
                    }
                    T::Number
                }
                b'0'..=b'9' => {
                    while let Some(b'0'..=b'9') = self.peek() {
                        self.advance();
                    }

                    if self.advance_if(b'.') {
                        while let Some(b'0'..=b'9') = self.peek() {
                            self.advance();
                        }
                        T::Float
                    } else {
                        T::Number
                    }
                }
                b'a'..=b'z' | b'A'..=b'Z' | b'_' | 127.. => {
                    advance_ident(self);
                    let ident = &self.source[start as usize..self.pos as usize];
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
                    while let Some(l) = self.peek()
                        && l != b'\n'
                    {
                        self.pos += 1;
                    }

                    let end = self.source[..self.pos as usize]
                        .iter()
                        .rposition(|&b| !b.is_ascii_whitespace())
                        .map_or(self.pos, |i| i as u32 + 1);

                    return Token { kind: T::Comment, start, end };
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
