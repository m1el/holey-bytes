use {core::panic, std::iter};

use std::array;

use logos::{Lexer, Logos};

use crate::lexer::{Op, TokenKind, Ty};

#[derive(Clone, Debug)]
pub enum Item {
    Import(String),
    Struct(Struct),
    Function(Function),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Builtin(Ty),
    Struct(String),
    Pinter(Box<Type>),
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub name:   String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty:   Type,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub args: Vec<Arg>,
    pub ret:  Type,
    pub body: Vec<Exp>,
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: String,
    pub ty:   Type,
}

#[derive(Clone, Debug)]
pub struct CtorField {
    pub name:  String,
    pub value: Exp,
}

#[derive(Clone, Debug)]
pub enum Exp {
    Literal(Literal),
    Variable(String),
    Call {
        name: String,
        args: Vec<Exp>,
    },
    Ctor {
        name:   Option<Box<Exp>>,
        fields: Vec<CtorField>,
    },
    Index {
        base:  Box<Exp>,
        index: Box<Exp>,
    },
    Field {
        base:  Box<Exp>,
        field: String,
    },
    Unary {
        op:  Op,
        exp: Box<Exp>,
    },
    Binary {
        op:    Op,
        left:  Box<Exp>,
        right: Box<Exp>,
    },
    If {
        cond:  Box<Exp>,
        then:  Box<Exp>,
        else_: Option<Box<Exp>>,
    },
    Let {
        name:  String,
        ty:    Option<Type>,
        value: Box<Exp>,
    },
    For {
        init:  Option<Box<Exp>>,
        cond:  Option<Box<Exp>>,
        step:  Option<Box<Exp>>,
        block: Box<Exp>,
    },
    Block(Vec<Exp>),
    Return(Option<Box<Exp>>),
    Break,
    Continue,
}

#[derive(Clone, Debug)]
pub enum Literal {
    Int(u64),
    Bool(bool),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub kind:  TokenKind,
    pub span:  std::ops::Range<usize>,
    pub value: String,
}

struct Parser<'a> {
    next_token: Option<Token>,
    lexer:      logos::Lexer<'a, TokenKind>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = TokenKind::lexer(input);
        let next_token = Self::next_token(&mut lexer);
        Self { next_token, lexer }
    }

    pub fn next(&mut self) -> Option<Token> {
        let token = self.next_token.clone();
        self.next_token = Self::next_token(&mut self.lexer);
        token
    }

    pub fn next_token(lexer: &mut Lexer<TokenKind>) -> Option<Token> {
        lexer.next().map(|r| {
            r.map(|e| Token {
                kind:  e,
                span:  lexer.span(),
                value: lexer.slice().to_owned(),
            })
            .unwrap_or_else(|e| {
                let (line, col) = Self::pos_to_line_col_low(lexer.source(), lexer.span().start);
                println!("Lexer error: {}:{}: {:?}", line, col, e);
                std::process::exit(1);
            })
        })
    }

    pub fn pos_to_line_col(&self, pos: usize) -> (usize, usize) {
        Self::pos_to_line_col_low(self.lexer.source(), pos)
    }

    pub fn pos_to_line_col_low(source: &str, pos: usize) -> (usize, usize) {
        let line = source[..pos].lines().count();
        let col = source[..pos].lines().last().map(|l| l.len()).unwrap_or(0);
        (line, col)
    }

    pub fn expect(&mut self, kind: TokenKind) -> Token {
        let token = self.expect_any();
        if token.kind == kind {
            token
        } else {
            let (line, col) = self.pos_to_line_col(token.span.start);
            panic!(
                "Expected {:?} at {}:{}, found {:?}",
                kind, line, col, token.kind
            )
        }
    }

    pub fn expect_any(&mut self) -> Token {
        self.next().unwrap_or_else(|| panic!("Unexpected EOF"))
    }

    pub fn peek(&self) -> Option<&Token> {
        self.next_token.as_ref()
    }

    pub fn try_advance(&mut self, kind: TokenKind) -> bool {
        if self.peek().is_some_and(|t| t.kind == kind) {
            self.next();
            true
        } else {
            false
        }
    }

    pub fn parse(&mut self) -> Vec<Item> {
        iter::from_fn(|| self.parse_item()).collect()
    }

    fn parse_item(&mut self) -> Option<Item> {
        let token = self.next()?;
        match token.kind {
            TokenKind::Struct => Some(self.parse_struct()),
            TokenKind::Fn => Some(self.parse_function()),
            TokenKind::Use => Some(Item::Import(self.expect(TokenKind::String).value)),
            tkn => {
                let (line, col) = self.pos_to_line_col(token.span.start);
                panic!("Unexpected {:?} at {}:{}", tkn, line, col)
            }
        }
    }

    fn parse_struct(&mut self) -> Item {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::LBrace);
        let fields = self.sequence(TokenKind::Comma, TokenKind::RBrace, Self::parse_field);
        Item::Struct(Struct { name, fields })
    }

    fn parse_field(&mut self) -> Field {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::Colon);
        let ty = self.type_();

        Field { name, ty }
    }

    fn type_(&mut self) -> Type {
        let token = self.next().unwrap();
        match token.kind {
            TokenKind::Ty(ty) => Type::Builtin(ty),
            TokenKind::Ident => Type::Struct(token.value),
            TokenKind::Op(Op::Band) => {
                let ty = self.type_();
                Type::Pinter(Box::new(ty))
            }
            tkn => {
                let (line, col) = self.pos_to_line_col(token.span.start);
                panic!("Unexpected {:?} at {}:{}", tkn, line, col)
            }
        }
    }

    fn parse_function(&mut self) -> Item {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::LParen);
        let args = self.sequence(TokenKind::Comma, TokenKind::RParen, Self::parse_arg);
        self.expect(TokenKind::Colon);
        let ret = self.type_();
        Item::Function(Function {
            name,
            args,
            ret,
            body: self.parse_block(),
        })
    }

    fn parse_arg(&mut self) -> Arg {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::Colon);
        let ty = self.type_();
        self.try_advance(TokenKind::Comma);
        Arg { name, ty }
    }

    fn parse_expr(&mut self) -> Exp {
        self.parse_binary_expr(255)
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> Exp {
        let mut lhs = self.parse_unit_expr();

        while let Some(TokenKind::Op(op)) = self.peek().map(|t| t.kind) {
            let prec = op.prec();
            if prec > min_prec {
                break;
            }

            self.next();
            let rhs = self.parse_binary_expr(prec);

            lhs = Exp::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
            };
        }

        lhs
    }

    fn parse_unit_expr(&mut self) -> Exp {
        let token = self.next().unwrap();
        let mut expr = match token.kind {
            TokenKind::True => Exp::Literal(Literal::Bool(true)),
            TokenKind::False => Exp::Literal(Literal::Bool(false)),
            TokenKind::Ident => Exp::Variable(token.value),
            TokenKind::LBrace => {
                Exp::Block(self.sequence(TokenKind::Semicolon, TokenKind::LBrace, Self::parse_expr))
            }
            TokenKind::LParen => {
                let expr = self.parse_expr();
                self.expect(TokenKind::RParen);
                expr
            }
            TokenKind::Number => {
                let value = token.value.parse().unwrap();
                Exp::Literal(Literal::Int(value))
            }
            TokenKind::Let => {
                let name = self.expect(TokenKind::Ident).value;
                let ty = self.try_advance(TokenKind::Colon).then(|| self.type_());
                self.expect(TokenKind::Op(Op::Assign));
                let value = self.parse_expr();
                Exp::Let {
                    name,
                    ty,
                    value: Box::new(value),
                }
            }
            TokenKind::If => {
                let cond = self.parse_expr();
                let then = Exp::Block(self.parse_block());
                let else_ = self
                    .try_advance(TokenKind::Else)
                    .then(|| {
                        if self.peek().is_some_and(|t| t.kind == TokenKind::If) {
                            self.parse_expr()
                        } else {
                            Exp::Block(self.parse_block())
                        }
                    })
                    .map(Box::new);
                Exp::If {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    else_,
                }
            }
            TokenKind::For => {
                let params =
                    self.sequence(TokenKind::Semicolon, TokenKind::LBrace, Self::parse_expr);
                let mut exprs = Vec::new();
                while !self.try_advance(TokenKind::RBrace) {
                    exprs.push(self.parse_expr());
                    self.try_advance(TokenKind::Semicolon);
                }
                let block = Exp::Block(exprs);
                let len = params.len();
                let mut exprs = params.into_iter();
                let [init, consd, step] = array::from_fn(|_| exprs.next());
                match len {
                    0 => Exp::For {
                        init:  None,
                        cond:  None,
                        step:  None,
                        block: Box::new(block),
                    },
                    1 => Exp::For {
                        init:  None,
                        cond:  init.map(Box::new),
                        step:  None,
                        block: Box::new(block),
                    },
                    3 => Exp::For {
                        init:  init.map(Box::new),
                        cond:  consd.map(Box::new),
                        step:  step.map(Box::new),
                        block: Box::new(block),
                    },
                    _ => {
                        let (line, col) = self.pos_to_line_col(token.span.start);
                        panic!("Invalid loop syntax at {}:{}, loop accepts 1 (while), 0 (loop), or 3 (for) statements separated by semicolon", line, col)
                    }
                }
            }
            TokenKind::Return => {
                let value = self
                    .peek()
                    .is_some_and(|t| {
                        !matches!(
                            t.kind,
                            TokenKind::Semicolon
                                | TokenKind::RBrace
                                | TokenKind::RParen
                                | TokenKind::Comma
                        )
                    })
                    .then(|| Box::new(self.parse_expr()));
                Exp::Return(value)
            }
            TokenKind::Op(op) => Exp::Unary {
                op,
                exp: Box::new(self.parse_expr()),
            },
            TokenKind::Dot => {
                let token = self.expect_any();
                match token.kind {
                    TokenKind::LBrace => {
                        let fields = self.sequence(
                            TokenKind::Comma,
                            TokenKind::RBrace,
                            Self::parse_ctor_field,
                        );
                        Exp::Ctor { name: None, fields }
                    }
                    tkn => {
                        let (line, col) = self.pos_to_line_col(token.span.start);
                        panic!("Unexpected {:?} at {}:{}", tkn, line, col)
                    }
                }
            }

            TokenKind::Ty(_)
            | TokenKind::String
            | TokenKind::Use
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::Struct
            | TokenKind::RBrace
            | TokenKind::RParen
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::Colon
            | TokenKind::Semicolon
            | TokenKind::Comma
            | TokenKind::Fn
            | TokenKind::Else => {
                let (line, col) = self.pos_to_line_col(token.span.start);
                panic!("Unexpected {:?} at {}:{}", token.kind, line, col)
            }
        };

        loop {
            match self.peek().map(|t| t.kind) {
                Some(TokenKind::LParen) => {
                    self.next();
                    expr = Exp::Call {
                        name: Box::new(expr),
                        args: self.sequence(TokenKind::Comma, TokenKind::RParen, Self::parse_expr),
                    };
                }
                Some(TokenKind::LBracket) => {
                    self.next();
                    let index = self.parse_expr();
                    self.expect(TokenKind::RBracket);
                    expr = Exp::Index {
                        base:  Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Some(TokenKind::Dot) => {
                    self.next();

                    let token = self.expect_any();
                    match token.kind {
                        TokenKind::Ident => {
                            expr = Exp::Field {
                                base:  Box::new(expr),
                                field: token.value,
                            };
                        }
                        TokenKind::LBrace => {
                            let fields = self.sequence(
                                TokenKind::Comma,
                                TokenKind::RBrace,
                                Self::parse_ctor_field,
                            );
                            expr = Exp::Ctor {
                                name: Some(Box::new(expr)),
                                fields,
                            };
                        }
                        tkn => {
                            let (line, col) = self.pos_to_line_col(token.span.start);
                            panic!("Unexpected {:?} at {}:{}", tkn, line, col)
                        }
                    }
                }
                _ => break expr,
            }
        }
    }

    pub fn parse_ctor_field(&mut self) -> CtorField {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::Colon);
        let value = self.parse_expr();
        CtorField { name, value }
    }

    pub fn parse_block(&mut self) -> Vec<Exp> {
        self.expect(TokenKind::LBrace);
        let mut exprs = Vec::new();
        while !self.try_advance(TokenKind::RBrace) {
            exprs.push(self.parse_expr());
            self.try_advance(TokenKind::Semicolon);
        }
        exprs
    }

    pub fn sequence<T>(
        &mut self,
        sep: TokenKind,
        term: TokenKind,
        mut parser: impl FnMut(&mut Self) -> T,
    ) -> Vec<T> {
        let mut items = Vec::new();
        while !self.try_advance(term) {
            items.push(parser(self));
            if self.try_advance(term) {
                break;
            }
            self.expect(sep);
        }
        items
    }
}

pub fn parse(input: &str) -> Vec<Item> {
    Parser::new(input).parse()
}

#[cfg(test)]
mod test {
    #[test]
    fn sanity() {
        let input = r#"
            struct Foo {
                x: i32,
                y: i32,
            }

            fn main(): void {
                let foo = Foo.{ x: 1, y: 2 };
                if foo.x > 0 {
                    return foo.x;
                } else {
                    return foo.y;
                }
                for i < 10 {
                    i = i + 1;
                }
                for let i = 0; i < 10; i = i + 1 {
                    i = i + 1;
                }
                i + 1 * 3 / 4 % 5 == 2 + 3 - 4 * 5 / 6 % 7;
                fomething();
                pahum(&foo);
                lupa(*soo);
                return foo.x + foo.y;
            }

            fn lupa(x: i32): i32 {
                return x;
            }

            fn pahum(x: &Foo): void {
                return;
            }
        "#;
        let _ = super::parse(input);
    }
}
