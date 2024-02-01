use logos::Logos;

macro_rules! gen_token {
    ($name:ident {
        keywords: {
            $($keyword:ident = $lit:literal,)*
        },
        operators: $op_name:ident {
            $($prec:literal: {$(
                $op:ident = $op_lit:literal,
            )*},)*
        },
        types: $ty_type:ident {
            $($ty:ident = $ty_lit:literal,)*
        },
        regexes: {
            $($regex:ident = $regex_lit:literal,)*
        },
    }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Copy, Logos)]
        #[logos(skip "[ \t\n]+")]
        pub enum $name {
            $(#[token($lit)] $keyword,)*
            $($(#[token($op_lit, |_| $op_name::$op)])*)*
            Op($op_name),
            $(#[token($ty_lit, |_| $ty_type::$ty)])*
            Ty($ty_type),
            $(#[regex($regex_lit)] $regex,)*
        }

        #[derive(Debug, Clone, PartialEq, Eq, Copy)]
        pub enum $op_name {
            $($($op,)*)*
        }

        #[derive(Debug, Clone, PartialEq, Eq, Copy)]
        pub enum $ty_type {
            $($ty,)*
        }

        impl $op_name {
            pub fn prec(&self) -> u8 {
                match self {
                    $($($op_name::$op => $prec,)*)*
                }
            }
        }
    };
}

gen_token! {
    TokenKind {
        keywords: {
            Use = "use",
            Fn = "fn",
            Let = "let",
            If = "if",
            Else = "else",
            For = "for",
            Return = "return",
            Break = "break",
            Continue = "continue",
            Struct = "struct",

            True = "true",
            False = "false",

            LBrace = "{",
            RBrace = "}",
            LParen = "(",
            RParen = ")",
            LBracket = "[",
            RBracket = "]",

            Colon = ":",
            Semicolon = ";",
            Comma = ",",
            Dot = ".",
        },
        operators: Op {
            14: {
                Assign = "=",
                AddAssign = "+=",
                SubAssign = "-=",
                MulAssign = "*=",
                DivAssign = "/=",
                ModAssign = "%=",
                AndAssign = "&=",
                OrAssign = "|=",
                XorAssign = "^=",
                ShlAssign = "<<=",
                ShrAssign = ">>=",
            },
            12: {
                Or = "||",
            },
            11: {
                And = "&&",
            },
            10: {
                Bor = "|",
            },
            9: {
                Xor = "^",
            },
            8: {
                Band = "&",
            },
            7: {
                Eq = "==",
                Neq = "!=",
            },
            6: {
                Lt = "<",
                Gt = ">",
                Le = "<=",
                Ge = ">=",
            },
            5: {
                Shl = "<<",
                Shr = ">>",
            },
            4: {
                Add = "+",
                Sub = "-",
            },
            3: {
                Mul = "*",
                Div = "/",
                Mod = "%",
            },
        },
        types: Ty {
            U8 = "u8",
            U16 = "u16",
            U32 = "u32",
            U64 = "u64",
            I8 = "i8",
            I16 = "i16",
            I32 = "i32",
            I64 = "i64",
            Bool = "bool",
            Void = "void",
        },
        regexes: {
            Ident = "[a-zA-Z_][a-zA-Z0-9_]*",
            Number = "[0-9]+",
        },
    }
}
