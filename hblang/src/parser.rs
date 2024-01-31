use {core::panic, std::iter};

use logos::{Lexer, Logos};

use crate::lexer::{Op, TokenKind, Ty};

pub enum Item {
    Struct(Struct),
    Function(Function),
}

pub enum Type {
    Builtin(Ty),
    Struct(String),
}

pub struct Struct {
    pub name:   String,
    pub fields: Vec<Field>,
}

pub struct Field {
    pub name: String,
    pub ty:   Type,
}

pub struct Function {
    pub name: String,
    pub args: Vec<Arg>,
    pub ret:  Type,
    pub body: Vec<Exp>,
}

pub struct Arg {
    pub name: String,
    pub ty:   Type,
}

pub enum Exp {
    Literal(Literal),
    Variable(String),
    Call {
        name: Box<Exp>,
        args: Vec<Exp>,
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
    Block(Vec<Exp>),
    Return(Box<Exp>),
    Break,
    Continue,
}

pub enum Literal {
    Int(i64),
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
                panic!("Lexer error: {}:{}", line, col,)
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
        let token = self.next().unwrap_or_else(|| panic!("Unexpected EOF"));
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
            tkn => {
                let (line, col) = self.pos_to_line_col(token.span.start);
                panic!("Unexpected {:?} at {}:{}", tkn, line, col)
            }
        }
    }

    fn parse_struct(&mut self) -> Item {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::LBrace);
        let fields = iter::from_fn(|| self.parse_field()).collect();
        self.expect(TokenKind::RBrace);
        Item::Struct(Struct { name, fields })
    }

    fn parse_field(&mut self) -> Option<Field> {
        if self.peek()?.kind == TokenKind::RBrace {
            return None;
        }

        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::Colon);
        let ty = self.type_();
        self.try_advance(TokenKind::Comma);

        Some(Field { name, ty })
    }

    fn type_(&mut self) -> Type {
        let token = self.next().unwrap();
        match token.kind {
            TokenKind::Ty(ty) => Type::Builtin(ty),
            TokenKind::Ident => Type::Struct(token.value),
            tkn => {
                let (line, col) = self.pos_to_line_col(token.span.start);
                panic!("Unexpected {:?} at {}:{}", tkn, line, col)
            }
        }
    }

    fn parse_function(&mut self) -> Item {
        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::LParen);
        let args = iter::from_fn(|| self.parse_arg()).collect();
        self.expect(TokenKind::RParen);
        self.expect(TokenKind::Colon);
        let ret = self.type_();
        self.expect(TokenKind::LBrace);
        let body = iter::from_fn(|| self.parse_stmt()).collect();
        self.expect(TokenKind::RBrace);
        Item::Function(Function {
            name,
            args,
            ret,
            body,
        })
    }

    fn parse_arg(&mut self) -> Option<Arg> {
        if self.peek()?.kind == TokenKind::RParen {
            return None;
        }

        let name = self.expect(TokenKind::Ident).value;
        self.expect(TokenKind::Colon);
        let ty = self.type_();
        self.try_advance(TokenKind::Comma);

        Some(Arg { name, ty })
    }

    fn parse_stmt(&mut self) -> Option<Exp> {
        if self.peek()?.kind == TokenKind::RBrace {
            return None;
        }

        let expr = self.parse_expr();
        self.expect(TokenKind::Semicolon);

        Some(expr)
    }

    fn parse_expr(&mut self) -> Exp {
        self.parse_binary_expr(255)
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> Exp {
        let mut lhs = self.parse_unit_expr();

        while let Some(TokenKind::Op(op)) = self.peek().map(|t| t.kind) {
            let prec = op.prec();
            if prec <= min_prec {
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
                let body = iter::from_fn(|| self.parse_stmt()).collect();
                self.expect(TokenKind::RBrace);
                Exp::Block(body)
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
            TokenKind::Fn => todo!(),
            TokenKind::Let => todo!(),
            TokenKind::If => todo!(),
            TokenKind::Else => todo!(),
            TokenKind::For => todo!(),
            TokenKind::Return => todo!(),
            TokenKind::Break => todo!(),
            TokenKind::Continue => todo!(),
            TokenKind::Struct => todo!(),
            TokenKind::RBrace => todo!(),
            TokenKind::RParen => todo!(),
            TokenKind::LBracket => todo!(),
            TokenKind::RBracket => todo!(),
            TokenKind::Colon => todo!(),
            TokenKind::Semicolon => todo!(),
            TokenKind::Comma => todo!(),
            TokenKind::Op(_) => todo!(),
            TokenKind::Ty(_) => todo!(),
            TokenKind::Dot => todo!(),
        };

        loop {
            match self.peek().map(|t| t.kind) {
                Some(TokenKind::LParen) => {
                    self.next();
                    let args = iter::from_fn(|| self.parse_call_arg()).collect();
                    self.expect(TokenKind::RParen);
                    expr = Exp::Call {
                        name: Box::new(expr),
                        args,
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
                    let field = self.expect(TokenKind::Ident).value;
                    expr = Exp::Field {
                        base: Box::new(expr),
                        field,
                    };
                }
                _ => break expr,
            }
        }
    }

    pub fn parse_call_arg(&mut self) -> Option<Exp> {
        if self.peek()?.kind == TokenKind::RParen {
            return None;
        }

        let expr = self.parse_expr();
        self.try_advance(TokenKind::Comma);

        Some(expr)
    }
}

pub fn parse(input: &str) -> Vec<Item> {
    Parser::new(input).parse()
}
