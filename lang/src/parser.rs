use {
    crate::{
        fmt::Formatter,
        ident::{self, Ident},
        lexer::{self, Lexer, Token, TokenKind},
    },
    alloc::{boxed::Box, string::String, vec::Vec},
    core::{
        alloc::Layout,
        cell::UnsafeCell,
        fmt::{self},
        intrinsics::unlikely,
        marker::PhantomData,
        ops::Deref,
        ptr::NonNull,
        sync::atomic::AtomicUsize,
    },
};

pub type Pos = u32;
pub type IdentFlags = u32;
pub type Symbols = Vec<Symbol>;
pub type FileId = u32;
pub type IdentIndex = u16;
pub type LoaderError = String;
pub type Loader<'a> = &'a mut (dyn FnMut(&str, &str) -> Result<FileId, LoaderError> + 'a);

pub const SOURCE_TO_AST_FACTOR: usize = 7 * (core::mem::size_of::<usize>() / 4) + 1;

pub mod idfl {
    use super::*;

    macro_rules! flags {
        ($($name:ident,)*) => {
            $(pub const $name: IdentFlags = 1 << (core::mem::size_of::<IdentFlags>() * 8 - 1 - ${index(0)});)*
            pub const ALL: IdentFlags = 0 $(| $name)*;
        };
    }

    flags! {
        MUTABLE,
        REFERENCED,
        COMPTIME,
    }
}

pub fn no_loader(_: &str, _: &str) -> Result<FileId, LoaderError> {
    Err(String::new())
}

#[derive(Debug)]
pub struct Symbol {
    pub name: Ident,
    pub flags: IdentFlags,
}

#[derive(Clone, Copy)]
struct ScopeIdent {
    ident: Ident,
    declared: bool,
    ordered: bool,
    flags: IdentFlags,
}

pub struct Parser<'a, 'b> {
    path: &'b str,
    loader: Loader<'b>,
    lexer: Lexer<'a>,
    arena: &'a Arena,
    ctx: &'b mut ParserCtx,
    token: Token,
    ns_bound: usize,
    trailing_sep: bool,
    packed: bool,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn parse(
        ctx: &'b mut ParserCtx,
        input: &'a str,
        path: &'b str,
        loader: Loader<'b>,
        arena: &'a Arena,
    ) -> &'a [Expr<'a>] {
        let mut lexer = Lexer::new(input);
        Self {
            loader,
            token: lexer.next(),
            lexer,
            path,
            ctx,
            arena,
            ns_bound: 0,
            trailing_sep: false,
            packed: false,
        }
        .file()
    }

    fn file(&mut self) -> &'a [Expr<'a>] {
        let f = self.collect_list(TokenKind::Semi, TokenKind::Eof, |s| s.expr_low(true));

        self.pop_scope(0);

        if !self.ctx.idents.is_empty() {
            // TODO: we need error recovery
            log::error!("{}", {
                let mut errors = String::new();
                for id in self.ctx.idents.drain(..) {
                    report_to(
                        self.lexer.source(),
                        self.path,
                        ident::pos(id.ident),
                        format_args!(
                            "undeclared identifier: {}",
                            self.lexer.slice(ident::range(id.ident))
                        ),
                        &mut errors,
                    );
                }
                errors
            });
            unreachable!();
        }

        f
    }

    fn next(&mut self) -> Token {
        core::mem::replace(&mut self.token, self.lexer.next())
    }

    fn ptr_expr(&mut self) -> &'a Expr<'a> {
        self.arena.alloc(self.expr())
    }

    fn expr_low(&mut self, top_level: bool) -> Expr<'a> {
        let left = self.unit_expr();
        self.bin_expr(left, 0, top_level)
    }

    fn expr(&mut self) -> Expr<'a> {
        self.expr_low(false)
    }

    fn bin_expr(&mut self, mut fold: Expr<'a>, min_prec: u8, top_level: bool) -> Expr<'a> {
        loop {
            let Some(prec) = self.token.kind.precedence() else {
                break;
            };

            if prec <= min_prec {
                break;
            }

            let op = self.next().kind;

            if op == TokenKind::Decl {
                self.declare_rec(&fold, top_level);
            }

            let right = self.unit_expr();
            let right = self.bin_expr(right, prec, false);
            let right = self.arena.alloc(right);
            let left = self.arena.alloc(fold);

            if let Some(op) = op.ass_op() {
                self.flag_idents(*left, idfl::MUTABLE);
                let right = Expr::BinOp { left: self.arena.alloc(fold), op, right };
                fold = Expr::BinOp { left, op: TokenKind::Assign, right: self.arena.alloc(right) };
            } else {
                fold = Expr::BinOp { left, right, op };
                if op == TokenKind::Assign {
                    self.flag_idents(*left, idfl::MUTABLE);
                }
            }
        }

        fold
    }

    fn declare_rec(&mut self, expr: &Expr, top_level: bool) {
        match *expr {
            Expr::Ident { pos, id, is_first, .. } => {
                self.declare(pos, id, !top_level, is_first || top_level)
            }
            Expr::Ctor { fields, .. } => {
                for CtorField { value, .. } in fields {
                    self.declare_rec(value, top_level)
                }
            }
            _ => self.report(expr.pos(), "cant declare this shit (yet)"),
        }
    }

    fn declare(&mut self, pos: Pos, id: Ident, ordered: bool, valid_order: bool) {
        if !valid_order {
            self.report(
                pos,
                format_args!(
                    "out of order declaration not allowed: {}",
                    self.lexer.slice(ident::range(id))
                ),
            );
        }

        let index = self.ctx.idents.binary_search_by_key(&id, |s| s.ident).expect("fck up");
        if core::mem::replace(&mut self.ctx.idents[index].declared, true) {
            self.report(
                pos,
                format_args!("redeclaration of identifier: {}", self.lexer.slice(ident::range(id))),
            )
        }

        self.ctx.idents[index].ordered = ordered;
    }

    fn resolve_ident(&mut self, token: Token) -> (Ident, bool) {
        let is_ct = token.kind == TokenKind::CtIdent;
        let name = self.lexer.slice(token.range());

        if let Some(builtin) = crate::ty::from_str(name) {
            return (builtin, false);
        }

        let (i, id, bl) = match self
            .ctx
            .idents
            .iter_mut()
            .enumerate()
            .rfind(|(_, elem)| self.lexer.slice(ident::range(elem.ident)) == name)
        {
            Some((i, elem)) => (i, elem, false),
            None => {
                let id = ident::new(token.start, name.len() as _);
                self.ctx.idents.push(ScopeIdent {
                    ident: id,
                    declared: false,
                    ordered: false,
                    flags: 0,
                });
                (self.ctx.idents.len() - 1, self.ctx.idents.last_mut().unwrap(), true)
            }
        };

        id.flags |= idfl::COMPTIME * is_ct as u32;
        if id.declared && id.ordered && self.ns_bound > i {
            id.flags |= idfl::COMPTIME;
            self.ctx.captured.push(id.ident);
        }

        (id.ident, bl)
    }

    fn tok_str(&mut self, range: Token) -> &'a str {
        self.lexer.slice(range.range())
    }

    fn unit_expr(&mut self) -> Expr<'a> {
        use {Expr as E, TokenKind as T};
        let frame = self.ctx.idents.len();
        let token @ Token { start: pos, .. } = self.next();
        let prev_boundary = self.ns_bound;
        let prev_captured = self.ctx.captured.len();
        let mut expr = match token.kind {
            T::Ct => E::Ct { pos, value: self.ptr_expr() },
            T::Directive if self.lexer.slice(token.range()) == "use" => {
                self.expect_advance(TokenKind::LParen);
                let str = self.expect_advance(TokenKind::DQuote);
                self.expect_advance(TokenKind::RParen);
                let path = self.lexer.slice(str.range());
                let path = &path[1..path.len() - 1];

                E::Mod {
                    pos,
                    path,
                    id: match (self.loader)(path, self.path) {
                        Ok(id) => id,
                        Err(e) => {
                            self.report(str.start, format_args!("error loading dependency: {e:#}"))
                        }
                    },
                }
            }
            T::Directive => E::Directive {
                pos: pos - 1, // need to undo the directive shift
                name: self.tok_str(token),
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, Self::expr)
                },
            },
            T::True => E::Bool { pos, value: true },
            T::False => E::Bool { pos, value: false },
            T::Idk => E::Idk { pos },
            T::DQuote => E::String { pos, literal: self.tok_str(token) },
            T::Packed => {
                self.packed = true;
                let expr = self.unit_expr();
                if self.packed {
                    self.report(
                        expr.pos(),
                        "this can not be packed \
                        (unlike your mom that gets packed every day by me)",
                    );
                }
                expr
            }
            T::Struct => E::Struct {
                packed: core::mem::take(&mut self.packed),
                fields: {
                    self.ns_bound = self.ctx.idents.len();
                    self.expect_advance(T::LBrace);
                    self.collect_list(T::Comma, T::RBrace, |s| {
                        let tok = s.token;
                        if s.advance_if(T::Comment) {
                            CommentOr::Comment { literal: s.tok_str(tok), pos: tok.start }
                        } else {
                            let name = s.expect_advance(T::Ident);
                            s.expect_advance(T::Colon);
                            CommentOr::Or(StructField {
                                pos: name.start,
                                name: s.tok_str(name),
                                ty: s.expr(),
                            })
                        }
                    })
                },
                captured: {
                    self.ns_bound = prev_boundary;
                    let mut captured = &mut self.ctx.captured[prev_captured..];
                    while let Some(it) = captured.take_first_mut() {
                        for ot in &mut *captured {
                            if it > ot {
                                core::mem::swap(it, ot);
                            }
                        }
                    }
                    debug_assert!(captured.is_sorted());
                    let preserved = self.ctx.captured[prev_captured..].partition_dedup().0.len();
                    self.ctx.captured.truncate(prev_captured + preserved);
                    self.arena.alloc_slice(&self.ctx.captured[prev_captured..])
                },
                pos: {
                    if self.ns_bound == 0 {
                        // we might save some memory
                        self.ctx.captured.clear();
                    }
                    pos
                },
                trailing_comma: core::mem::take(&mut self.trailing_sep),
            },
            T::Ident | T::CtIdent => {
                let (id, is_first) = self.resolve_ident(token);
                let name = self.tok_str(token);
                E::Ident { pos, is_ct: token.kind == T::CtIdent, name, id, is_first }
            }
            T::If => E::If {
                pos,
                cond: self.ptr_expr(),
                then: self.ptr_expr(),
                else_: self.advance_if(T::Else).then(|| self.ptr_expr()),
            },
            T::Loop => E::Loop { pos, body: self.ptr_expr() },
            T::Break => E::Break { pos },
            T::Continue => E::Continue { pos },
            T::Return => E::Return {
                pos,
                val: (!matches!(
                    self.token.kind,
                    T::Semi | T::RBrace | T::RBrack | T::RParen | T::Comma
                ))
                .then(|| self.ptr_expr()),
            },
            T::Fn => E::Closure {
                pos,
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, |s| {
                        let name = s.advance_ident();
                        let (id, _) = s.resolve_ident(name);
                        s.declare(name.start, id, true, true);
                        s.expect_advance(T::Colon);
                        Arg {
                            pos: name.start,
                            name: s.tok_str(name),
                            is_ct: name.kind == T::CtIdent,
                            id,
                            ty: s.expr(),
                        }
                    })
                },
                ret: {
                    self.expect_advance(T::Colon);
                    self.ptr_expr()
                },
                body: self.ptr_expr(),
            },
            T::Ctor => self.ctor(pos, None),
            T::Tupl => self.tupl(pos, None),
            T::LBrack => E::Slice {
                item: self.ptr_unit_expr(),
                size: self.advance_if(T::Semi).then(|| self.ptr_expr()),
                pos: {
                    self.expect_advance(T::RBrack);
                    pos
                },
            },
            T::Band | T::Mul | T::Xor | T::Sub => E::UnOp {
                pos,
                op: token.kind,
                val: {
                    let expr = self.ptr_unit_expr();
                    if token.kind == T::Band {
                        self.flag_idents(*expr, idfl::REFERENCED);
                    }
                    expr
                },
            },
            T::LBrace => E::Block { pos, stmts: self.collect_list(T::Semi, T::RBrace, Self::expr) },
            T::Number => {
                let slice = self.lexer.slice(token.range());
                let (slice, radix) = match &slice.get(0..2) {
                    Some("0x") => (&slice[2..], Radix::Hex),
                    Some("0b") => (&slice[2..], Radix::Binary),
                    Some("0o") => (&slice[2..], Radix::Octal),
                    _ => (slice, Radix::Decimal),
                };
                E::Number {
                    pos,
                    value: match u64::from_str_radix(slice, radix as u32) {
                        Ok(value) => value,
                        Err(e) => self.report(token.start, format_args!("invalid number: {e}")),
                    } as i64,
                    radix,
                }
            }
            T::LParen => {
                let expr = self.expr();
                self.expect_advance(T::RParen);
                expr
            }
            T::Comment => Expr::Comment { pos, literal: self.tok_str(token) },
            tok => self.report(token.start, format_args!("unexpected token: {tok}")),
        };

        loop {
            let token = self.token;
            if matches!(token.kind, T::LParen | T::Ctor | T::Dot | T::Tupl | T::LBrack) {
                self.next();
            }

            expr = match token.kind {
                T::LParen => Expr::Call {
                    func: self.arena.alloc(expr),
                    args: self.collect_list(T::Comma, T::RParen, Self::expr),
                    trailing_comma: core::mem::take(&mut self.trailing_sep),
                },
                T::Ctor => self.ctor(token.start, Some(expr)),
                T::Tupl => self.tupl(token.start, Some(expr)),
                T::LBrack => E::Index {
                    base: self.arena.alloc(expr),
                    index: {
                        let index = self.expr();
                        self.expect_advance(T::RBrack);
                        self.arena.alloc(index)
                    },
                },
                T::Dot => E::Field {
                    target: self.arena.alloc(expr),
                    pos: token.start,
                    name: {
                        let token = self.expect_advance(T::Ident);
                        self.tok_str(token)
                    },
                },
                _ => break,
            }
        }

        if matches!(token.kind, T::Loop | T::LBrace | T::Fn) {
            self.pop_scope(frame);
        }

        expr
    }

    fn tupl(&mut self, pos: Pos, ty: Option<Expr<'a>>) -> Expr<'a> {
        Expr::Tupl {
            pos,
            ty: ty.map(|ty| self.arena.alloc(ty)),
            fields: self.collect_list(TokenKind::Comma, TokenKind::RParen, Self::expr),
            trailing_comma: core::mem::take(&mut self.trailing_sep),
        }
    }

    fn ctor(&mut self, pos: Pos, ty: Option<Expr<'a>>) -> Expr<'a> {
        Expr::Ctor {
            pos,
            ty: ty.map(|ty| self.arena.alloc(ty)),
            fields: self.collect_list(TokenKind::Comma, TokenKind::RBrace, |s| {
                let name_tok = s.advance_ident();
                let name = s.tok_str(name_tok);
                CtorField {
                    pos: name_tok.start,
                    name,
                    value: if s.advance_if(TokenKind::Colon) {
                        s.expr()
                    } else {
                        let (id, is_first) = s.resolve_ident(name_tok);
                        Expr::Ident { pos: name_tok.start, is_ct: false, id, name, is_first }
                    },
                }
            }),
            trailing_comma: core::mem::take(&mut self.trailing_sep),
        }
    }

    fn advance_ident(&mut self) -> Token {
        if matches!(self.token.kind, TokenKind::Ident | TokenKind::CtIdent) {
            self.next()
        } else {
            self.report(
                self.token.start,
                format_args!("expected identifier, found {}", self.token.kind),
            )
        }
    }

    fn pop_scope(&mut self, frame: usize) {
        let mut undeclared_count = frame;
        for i in frame..self.ctx.idents.len() {
            if !&self.ctx.idents[i].declared {
                self.ctx.idents.swap(i, undeclared_count);
                undeclared_count += 1;
            }
        }

        self.ctx
            .idents
            .drain(undeclared_count..)
            .map(|ident| Symbol { name: ident.ident, flags: ident.flags })
            .collect_into(&mut self.ctx.symbols);
    }

    fn ptr_unit_expr(&mut self) -> &'a Expr<'a> {
        self.arena.alloc(self.unit_expr())
    }

    fn collect_list<T: Copy>(
        &mut self,
        delim: TokenKind,
        end: TokenKind,
        mut f: impl FnMut(&mut Self) -> T,
    ) -> &'a [T] {
        let mut view = self.ctx.stack.view();
        while !self.advance_if(end) {
            let val = f(self);
            self.trailing_sep = self.advance_if(delim);
            unsafe { self.ctx.stack.push(&mut view, val) };
        }
        self.arena.alloc_slice(unsafe { self.ctx.stack.finalize(view) })
    }

    fn advance_if(&mut self, kind: TokenKind) -> bool {
        if self.token.kind == kind {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_advance(&mut self, kind: TokenKind) -> Token {
        if self.token.kind != kind {
            self.report(
                self.token.start,
                format_args!("expected {}, found {}", kind, self.token.kind),
            );
        }
        self.next()
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl fmt::Display) -> ! {
        log::error!("{}", Report::new(self.lexer.source(), self.path, pos, msg));
        unreachable!();
    }

    fn flag_idents(&mut self, e: Expr<'a>, flags: IdentFlags) {
        match e {
            Expr::Ident { id, .. } => find_ident(&mut self.ctx.idents, id).flags |= flags,
            Expr::Field { target, .. } => self.flag_idents(*target, flags),
            _ => {}
        }
    }
}

fn find_ident(idents: &mut [ScopeIdent], id: Ident) -> &mut ScopeIdent {
    idents.binary_search_by_key(&id, |si| si.ident).map(|i| &mut idents[i]).unwrap()
}

pub fn find_symbol(symbols: &[Symbol], id: Ident) -> &Symbol {
    symbols.binary_search_by_key(&id, |s| s.name).map(|i| &symbols[i]).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arg<'a> {
    pub pos: u32,
    pub name: &'a str,
    pub id: Ident,
    pub is_ct: bool,
    pub ty: Expr<'a>,
}

impl Poser for Arg<'_> {
    fn posi(&self) -> Pos {
        self.pos
    }
}

macro_rules! generate_expr {
    ($(#[$meta:meta])* $vis:vis enum $name:ident<$lt:lifetime> {$(
        $(#[$field_meta:meta])*
        $variant:ident {

            $($field:ident: $ty:ty,)*
        },
    )*}) => {
        $(#[$meta])*
        $vis enum $name<$lt> {$(
            $(#[$field_meta])*
            $variant {
                $($field: $ty,)*
            },
        )*}

        impl<$lt> $name<$lt> {
            pub fn used_bytes(&self) -> usize {
                match self {
                    $(Self::$variant { $($field),* } => {
                        0 $(.max($field as *const _ as usize - self as *const _ as usize
                            + core::mem::size_of::<$ty>()))*
                    })*
                }
            }

            pub fn pos(&self) -> Pos {
                #[allow(unused_variables)]
                match self {
                    $(Self::$variant { $($field),* } => generate_expr!(@first $(($field),)*).posi(),)*
                }
            }
        }
    };

    (@filed_names $variant:ident $ident1:ident) => { Self::$variant { $ident1: a } };

    (@first ($($first:tt)*), $($rest:tt)*) => { $($first)* };
    (@last ($($ign:tt)*), $($rest:tt)*) => { $($rest)* };
    (@last ($($last:tt)*),) => { $($last)* };
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Radix {
    Hex = 16,
    Octal = 8,
    Binary = 2,
    Decimal = 10,
}

generate_expr! {
    /// `LIST(start, sep, end, elem) => start { elem sep } [elem] end`
    /// `OP := grep for `#define OP:`
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Expr<'a> {
        /// `'ct' Expr`
        Ct {
            pos: Pos,
            value: &'a Self,
        },
        /// `'"([^"]|\\")"'`
        String {
            pos: Pos,
            literal: &'a str,
        },
        /// `'//[^\n]' | '/*' { '([^/*]|*/)*' | Comment } '*/'
        Comment {
            pos: Pos,
            literal: &'a str,
        },
        /// `'break'`
        Break {
            pos: Pos,
        },
        /// `'continue'`
        Continue {
            pos: Pos,
        },
        /// `'fn' LIST('(', ',', ')', Ident ':' Expr) ':' Expr Expr`
        Closure {
            pos:  Pos,
            args: &'a [Arg<'a>],
            ret:  &'a Self,
            body: &'a Self,
        },
        /// `Expr LIST('(', ',', ')', Expr)`
        Call {
            func: &'a Self,
            args: &'a [Self],
            trailing_comma: bool,
        },
        /// `'return' [Expr]`
        Return {
            pos: Pos,
            val: Option<&'a Self>,
        },
        /// note: ':unicode:' is any utf-8 character except ascii
        /// `'[a-zA-Z_:unicode:][a-zA-Z0-9_:unicode:]*'`
        Ident {
            pos: Pos,
            is_ct: bool,
            is_first: bool,
            id: Ident,
            name: &'a str,
        },
        /// `LIST('{', [';'], '}', Expr)`
        Block {
            pos:   Pos,
            stmts: &'a [Self],
        },
        /// `'0b[01]+' | '0o[0-7]+' | '[0-9]+' | '0b[01]+'`
        Number {
            pos:   Pos,
            value: i64,
            radix: Radix,
        },
        /// node: precedence defined in `OP` applies
        /// `Expr OP Expr`
        BinOp {
            left:  &'a Self,
            op:    TokenKind,
            right: &'a Self,
        },
        /// `'if' Expr Expr [else Expr]`
        If {
            pos:   Pos,
            cond:  &'a Self,
            then:  &'a Self,
            else_: Option<&'a Self>,
        },
        /// `'loop' Expr`
        Loop {
            pos:  Pos,
            body: &'a Self,
        },
        /// `('&' | '*' | '^') Expr`
        UnOp {
            pos: Pos,
            op:  TokenKind,
            val: &'a Self,
        },
        /// `'struct' LIST('{', ',', '}', Ident ':' Expr)`
        Struct {
            pos:      Pos,
            fields:   &'a [CommentOr<'a, StructField<'a>>],
            captured: &'a [Ident],
            trailing_comma: bool,
            packed: bool,
        },
        /// `[Expr] LIST('.{', ',', '}', Ident [':' Expr])`
        Ctor {
            pos:    Pos,
            ty:     Option<&'a Self>,
            fields: &'a [CtorField<'a>],
            trailing_comma: bool,
        },
        /// `[Expr] LIST('.(', ',', ')', Ident [':' Expr])`
        Tupl {
            pos:    Pos,
            ty:     Option<&'a Self>,
            fields: &'a [Self],
            trailing_comma: bool,
        },
        /// `'[' Expr [';' Expr] ']'`
        Slice {
            pos: Pos,
            size: Option<&'a Self>,
            item: &'a Self,
        },
        /// `Expr '[' Expr ']'`
        Index {
            base: &'a Self,
            index: &'a Self,
        },
        /// `Expr '.' Ident`
        Field {
            target: &'a Self,
            // we put it second place because its the pos of '.'
            pos: Pos,
            name:  &'a str,
        },
        /// `'true' | 'false'`
        Bool {
            pos:   Pos,
            value: bool,
        },
        /// `'idk'`
        Idk {
            pos: Pos,
        },
        /// `'@' Ident List('(', ',', ')', Expr)`
        Directive {
            pos:  Pos,
            name: &'a str,
            args: &'a [Self],
        },
        /// `'@use' '(' String ')'`
        Mod {
            pos:  Pos,
            id:   FileId,
            path: &'a str,
        },
    }
}

impl<'a> Expr<'a> {
    pub fn declares(&self, iden: Result<Ident, &str>) -> Option<Ident> {
        match *self {
            Self::Ident { id, name, .. } if iden == Ok(id) || iden == Err(name) => Some(id),
            Self::Ctor { fields, .. } => fields.iter().find_map(|f| f.value.declares(iden)),
            _ => None,
        }
    }

    pub fn has_ct(&self, symbols: &[Symbol]) -> bool {
        match *self {
            Self::Ident { id, .. } => find_symbol(symbols, id).flags & idfl::COMPTIME != 0,
            Self::Ctor { fields, .. } => fields.iter().any(|f| f.value.has_ct(symbols)),
            _ => false,
        }
    }

    pub fn find_pattern_path<T, F: FnOnce(&Expr) -> T>(
        &self,
        ident: Ident,
        target: &Expr,
        mut with_final: F,
    ) -> Result<T, F> {
        match *self {
            Self::Ident { id, .. } if id == ident => Ok(with_final(target)),
            Self::Ctor { fields, .. } => {
                for &CtorField { name, value, pos } in fields {
                    match value.find_pattern_path(
                        ident,
                        &Expr::Field { pos, target, name },
                        with_final,
                    ) {
                        Ok(value) => return Ok(value),
                        Err(e) => with_final = e,
                    }
                }
                Err(with_final)
            }
            _ => Err(with_final),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StructField<'a> {
    pub pos: Pos,
    pub name: &'a str,
    pub ty: Expr<'a>,
}

impl Poser for StructField<'_> {
    fn posi(&self) -> Pos {
        self.pos
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtorField<'a> {
    pub pos: Pos,
    pub name: &'a str,
    pub value: Expr<'a>,
}

impl Poser for CtorField<'_> {
    fn posi(&self) -> Pos {
        self.pos
    }
}

pub trait Poser {
    fn posi(&self) -> Pos;
}

impl Poser for Pos {
    fn posi(&self) -> Pos {
        *self
    }
}

impl<'a> Poser for Expr<'a> {
    fn posi(&self) -> Pos {
        self.pos()
    }
}

impl<'a, T: Poser> Poser for CommentOr<'a, T> {
    fn posi(&self) -> Pos {
        match self {
            CommentOr::Or(expr) => expr.posi(),
            CommentOr::Comment { pos, .. } => *pos,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommentOr<'a, T> {
    Or(T),
    Comment { literal: &'a str, pos: Pos },
}

impl<'a, T: Copy> CommentOr<'a, T> {
    pub fn or(&self) -> Option<T> {
        match *self {
            CommentOr::Or(v) => Some(v),
            CommentOr::Comment { .. } => None,
        }
    }
}

pub struct Display<'a> {
    source: &'a str,
    expr: &'a Expr<'a>,
}

impl<'a> Display<'a> {
    pub fn new(source: &'a str, expr: &'a Expr<'a>) -> Self {
        Self { source, expr }
    }
}

impl core::fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Formatter::new(self.source).fmt(self.expr, f)
    }
}

#[derive(Default)]
pub struct ParserCtx {
    symbols: Symbols,
    stack: StackAlloc,
    idents: Vec<ScopeIdent>,
    captured: Vec<Ident>,
}

#[repr(C)]
pub struct AstInner<T: ?Sized> {
    ref_count: AtomicUsize,
    pub mem: ArenaChunk,
    exprs: *const [Expr<'static>],

    pub path: Box<str>,
    pub file: Box<str>,
    pub symbols: T,
}

impl AstInner<[Symbol]> {
    fn layout(syms: usize) -> core::alloc::Layout {
        core::alloc::Layout::new::<AstInner<()>>()
            .extend(core::alloc::Layout::array::<Symbol>(syms).unwrap())
            .unwrap()
            .0
    }

    fn new(file: Box<str>, path: &str, ctx: &mut ParserCtx, loader: Loader) -> NonNull<Self> {
        let arena = Arena::with_capacity(file.len() * SOURCE_TO_AST_FACTOR);
        let exprs =
            unsafe { core::mem::transmute(Parser::parse(ctx, &file, path, loader, &arena)) };

        ctx.symbols.sort_unstable_by_key(|s| s.name);

        let layout = Self::layout(ctx.symbols.len());

        unsafe {
            let ptr = alloc::alloc::alloc(layout);
            let inner: *mut Self = core::ptr::from_raw_parts_mut(ptr as *mut _, ctx.symbols.len());

            core::ptr::write(inner as *mut AstInner<()>, AstInner {
                ref_count: AtomicUsize::new(1),
                mem: arena.chunk.into_inner(),
                exprs,
                path: path.into(),
                file,
                symbols: (),
            });
            core::ptr::addr_of_mut!((*inner).symbols)
                .as_mut_ptr()
                .copy_from_nonoverlapping(ctx.symbols.as_ptr(), ctx.symbols.len());

            NonNull::new_unchecked(inner)
        }
    }

    pub fn report<D>(&self, pos: Pos, msg: D) -> Report<D> {
        Report::new(&self.file, &self.path, pos, msg)
    }
}

fn report_to(file: &str, path: &str, pos: Pos, msg: impl fmt::Display, out: &mut impl fmt::Write) {
    let (line, mut col) = lexer::line_col(file.as_bytes(), pos);
    #[cfg(feature = "std")]
    let disp = crate::fs::display_rel_path(path);
    #[cfg(not(feature = "std"))]
    let disp = path;
    _ = writeln!(out, "{}:{}:{}: {}", disp, line, col, msg);

    let line = &file[file[..pos as usize].rfind('\n').map_or(0, |i| i + 1)
        ..file[pos as usize..].find('\n').unwrap_or(file.len()) + pos as usize];
    col += line.matches('\t').count() * 3;

    _ = writeln!(out, "{}", line.replace("\t", "    "));
    _ = writeln!(out, "{}^", " ".repeat(col - 1));
}

pub struct Report<'a, D> {
    file: &'a str,
    path: &'a str,
    pos: Pos,
    msg: D,
}

impl<'a, D> Report<'a, D> {
    pub fn new(file: &'a str, path: &'a str, pos: Pos, msg: D) -> Self {
        Self { file, path, pos, msg }
    }
}

impl<D: core::fmt::Display> core::fmt::Display for Report<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        report_to(self.file, self.path, self.pos, &self.msg, f);
        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct Ast(NonNull<AstInner<[Symbol]>>);

impl Ast {
    pub fn new(path: &str, content: String, ctx: &mut ParserCtx, loader: Loader) -> Self {
        Self(AstInner::new(content.into(), path, ctx, loader))
    }

    pub fn exprs(&self) -> &[Expr] {
        unsafe { &*self.inner().exprs }
    }

    fn inner(&self) -> &AstInner<[Symbol]> {
        unsafe { self.0.as_ref() }
    }

    pub fn find_decl(&self, id: Result<Ident, &str>) -> Option<(&Expr, Ident)> {
        self.exprs().iter().find_map(|expr| match expr {
            Expr::BinOp { left, op: TokenKind::Decl, .. } => left.declares(id).map(|id| (expr, id)),
            _ => None,
        })
    }

    pub fn ident_str(&self, ident: Ident) -> &str {
        &self.file[ident::range(ident)]
    }
}

impl Default for Ast {
    fn default() -> Self {
        Self(AstInner::new("".into(), "", &mut ParserCtx::default(), &mut no_loader))
    }
}

#[derive(Clone, Copy)]
#[repr(packed)]
pub struct ExprRef(NonNull<Expr<'static>>);

impl ExprRef {
    pub fn new(expr: &Expr) -> Self {
        Self(NonNull::from(expr).cast())
    }

    pub fn get<'a>(&self, from: &'a Ast) -> Option<&'a Expr<'a>> {
        from.mem.contains(self.0.as_ptr() as _).then_some(())?;
        // SAFETY: the pointer is or was a valid reference in the past, if it points within one of
        // arenas regions, it muts be walid, since arena does not give invalid pointers to its
        // allocations
        Some(unsafe { { self.0 }.as_ref() })
    }

    pub fn dangling() -> Self {
        Self(NonNull::dangling())
    }
}

impl Default for ExprRef {
    fn default() -> Self {
        Self::dangling()
    }
}

unsafe impl Send for Ast {}
unsafe impl Sync for Ast {}

impl Clone for Ast {
    fn clone(&self) -> Self {
        unsafe { self.0.as_ref() }.ref_count.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for Ast {
    fn drop(&mut self) {
        let inner = unsafe { self.0.as_ref() };
        if inner.ref_count.fetch_sub(1, core::sync::atomic::Ordering::Relaxed) == 1 {
            let inner = unsafe { self.0.as_mut() };
            let len = inner.symbols.len();
            unsafe { core::ptr::drop_in_place(inner) };
            let layout = AstInner::layout(len);
            unsafe {
                alloc::alloc::dealloc(self.0.as_ptr() as _, layout);
            }
        }
    }
}

impl Deref for Ast {
    type Target = AstInner<[Symbol]>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

struct StackAllocView<T> {
    prev: usize,
    base: usize,
    _ph: PhantomData<T>,
}

struct StackAlloc {
    data: *mut u8,
    len: usize,
    cap: usize,
}

impl StackAlloc {
    const MAX_ALIGN: usize = 16;

    fn view<T: Copy>(&mut self) -> StackAllocView<T> {
        let prev = self.len;
        let align = core::mem::align_of::<T>();
        assert!(align <= Self::MAX_ALIGN);
        self.len = (self.len + align - 1) & !(align - 1);
        StackAllocView { base: self.len, prev, _ph: PhantomData }
    }

    unsafe fn push<T: Copy>(&mut self, _view: &mut StackAllocView<T>, value: T) {
        if unlikely(self.len + core::mem::size_of::<T>() > self.cap) {
            let next_cap = self.cap.max(16 * 32).max(core::mem::size_of::<T>()) * 2;
            if self.cap == 0 {
                let layout =
                    core::alloc::Layout::from_size_align_unchecked(next_cap, Self::MAX_ALIGN);
                self.data = alloc::alloc::alloc(layout);
            } else {
                let old_layout =
                    core::alloc::Layout::from_size_align_unchecked(self.cap, Self::MAX_ALIGN);
                self.data = alloc::alloc::realloc(self.data, old_layout, next_cap);
            }
            self.cap = next_cap;
        }

        let dst = self.data.add(self.len) as *mut T;
        debug_assert!(dst.is_aligned(),);
        self.len += core::mem::size_of::<T>();
        core::ptr::write(dst, value);
    }

    unsafe fn finalize<T: Copy>(&mut self, view: StackAllocView<T>) -> &[T] {
        if unlikely(self.cap == 0) {
            return &[];
        }
        let slice = core::slice::from_ptr_range(
            self.data.add(view.base) as *const T..self.data.add(self.len) as *const T,
        );
        self.len = view.prev;
        slice
    }
}

impl Default for StackAlloc {
    fn default() -> Self {
        Self { data: core::ptr::null_mut(), len: 0, cap: 0 }
    }
}

impl Drop for StackAlloc {
    fn drop(&mut self) {
        let layout =
            unsafe { core::alloc::Layout::from_size_align_unchecked(self.cap, Self::MAX_ALIGN) };
        unsafe { alloc::alloc::dealloc(self.data, layout) };
    }
}

#[derive(Default)]
pub struct Arena {
    chunk: UnsafeCell<ArenaChunk>,
}

impl Arena {
    pub fn with_capacity(cap: usize) -> Arena {
        Self { chunk: UnsafeCell::new(ArenaChunk::new(cap, ArenaChunk::default())) }
    }

    pub fn alloc<'a>(&'a self, expr: Expr<'a>) -> &'a Expr<'a> {
        let layout = core::alloc::Layout::from_size_align(
            expr.used_bytes(),
            core::mem::align_of::<Expr<'a>>(),
        )
        .unwrap();
        let ptr = self.alloc_low(layout);
        unsafe {
            ptr.cast::<usize>().copy_from_nonoverlapping(
                NonNull::from(&expr).cast(),
                layout.size() / core::mem::size_of::<usize>(),
            )
        };
        unsafe { ptr.cast::<Expr<'a>>().as_ref() }
    }

    pub fn alloc_slice<'a, T: Copy>(&'a self, slice: &[T]) -> &'a [T] {
        if slice.is_empty() || core::mem::size_of::<T>() == 0 {
            return &mut [];
        }

        let layout = core::alloc::Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_low(layout);
        unsafe { ptr.as_ptr().cast::<T>().copy_from_nonoverlapping(slice.as_ptr(), slice.len()) };
        unsafe { core::slice::from_raw_parts(ptr.as_ptr() as _, slice.len()) }
    }

    fn alloc_low(&self, layout: core::alloc::Layout) -> NonNull<u8> {
        let chunk = unsafe { &mut *self.chunk.get() };

        if let Some(ptr) = chunk.alloc(layout) {
            return ptr;
        }

        unsafe {
            core::ptr::write(
                chunk,
                ArenaChunk::new(
                    1024 * 4 - core::mem::size_of::<ArenaChunk>(),
                    core::ptr::read(chunk),
                ),
            );
        }

        chunk.alloc(layout).unwrap()
    }
}

pub struct ArenaChunk {
    base: *mut u8,
    end: *mut u8,
    size: usize,
}

impl Default for ArenaChunk {
    fn default() -> Self {
        Self {
            base: core::mem::size_of::<Self>() as _,
            end: core::mem::size_of::<Self>() as _,
            size: 0,
        }
    }
}

impl ArenaChunk {
    fn layout(size: usize) -> Layout {
        Layout::new::<Self>().extend(Layout::array::<u8>(size).unwrap()).unwrap().0
    }

    fn new(size: usize, next: Self) -> Self {
        let mut base = unsafe { alloc::alloc::alloc(Self::layout(size)) };
        let end = unsafe { base.add(size) };
        unsafe { core::ptr::write(base.cast(), next) };
        base = unsafe { base.add(core::mem::size_of::<Self>()) };
        Self { base, end, size }
    }

    fn alloc(&mut self, layout: core::alloc::Layout) -> Option<NonNull<u8>> {
        let padding = self.end as usize - (self.end as usize & !(layout.align() - 1));
        let size = layout.size() + padding;
        if size > self.end as usize - self.base as usize {
            return None;
        }
        unsafe { self.end = self.end.sub(size) };
        unsafe { Some(NonNull::new_unchecked(self.end)) }
    }

    fn next(&self) -> Option<&Self> {
        unsafe { self.base.cast::<Self>().sub(1).as_ref() }
    }

    fn contains(&self, arg: *mut u8) -> bool {
        (self.base <= arg && unsafe { self.base.add(self.size) } > arg)
            || self.next().map_or(false, |s| s.contains(arg))
    }

    pub fn size(&self) -> usize {
        self.base as usize + self.size - self.end as usize + self.next().map_or(0, Self::size)
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        if self.size == 0 {
            return;
        }
        _ = self.next().map(|r| unsafe { core::ptr::read(r) });
        unsafe {
            alloc::alloc::dealloc(
                self.base.sub(core::mem::size_of::<Self>()),
                Self::layout(self.size),
            )
        }
    }
}
