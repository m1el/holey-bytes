use std::{
    cell::{Cell, UnsafeCell},
    io,
    ops::{Deref, Not},
    ptr::NonNull,
    sync::atomic::AtomicUsize,
};

use crate::{
    codegen::bt,
    ident::{self, Ident},
    lexer::{Lexer, LineMap, Token, TokenKind},
    log,
};

pub type Pos = u32;
pub type IdentFlags = u32;
pub type Symbols = Vec<Symbol>;
pub type FileId = u32;
pub type Loader<'a> = &'a (dyn Fn(&str, &str) -> io::Result<FileId> + 'a);

pub mod idfl {
    use super::*;

    macro_rules! flags {
        ($($name:ident,)*) => {
            $(pub const $name: IdentFlags = 1 << (std::mem::size_of::<IdentFlags>() * 8 - 1 - ${index(0)});)*
            pub const ALL: IdentFlags = 0 $(| $name)*;
        };
    }

    flags! {
        MUTABLE,
        REFERENCED,
        COMPTIME,
    }

    pub fn index(i: IdentFlags) -> u16 {
        (i & !ALL) as _
    }
}

pub fn no_loader(_: &str, _: &str) -> io::Result<FileId> {
    Err(io::ErrorKind::NotFound.into())
}

#[derive(Debug)]
pub struct Symbol {
    pub name:  Ident,
    pub flags: IdentFlags,
}

#[derive(Clone, Copy)]
struct ScopeIdent {
    ident:    Ident,
    declared: bool,
    flags:    IdentFlags,
}

pub struct Parser<'a, 'b> {
    path:     &'b str,
    loader:   Loader<'b>,
    lexer:    Lexer<'b>,
    arena:    &'b Arena<'a>,
    token:    Token,
    symbols:  &'b mut Symbols,
    ns_bound: usize,
    idents:   Vec<ScopeIdent>,
    captured: Vec<Ident>,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(arena: &'b Arena<'a>, symbols: &'b mut Symbols, loader: Loader<'b>) -> Self {
        let mut lexer = Lexer::new("");
        Self {
            loader,
            token: lexer.next(),
            lexer,
            path: "",
            arena,
            symbols,
            ns_bound: 0,
            idents: Vec::new(),
            captured: Vec::new(),
        }
    }

    pub fn file(&mut self, input: &'b str, path: &'b str) -> &'a [Expr<'a>] {
        self.path = path;
        self.lexer = Lexer::new(input);
        self.token = self.lexer.next();

        let f = self.collect_list(TokenKind::Semi, TokenKind::Eof, Self::expr);

        self.pop_scope(0);
        let has_undeclared = !self.idents.is_empty();
        for id in self.idents.drain(..) {
            let (line, col) = self.lexer.line_col(ident::pos(id.ident));
            eprintln!(
                "{}:{}:{} => undeclared identifier: {}",
                self.path,
                line,
                col,
                self.lexer.slice(ident::range(id.ident))
            );
        }

        if has_undeclared {
            // TODO: we need error recovery
            unreachable!();
        }

        f
    }

    fn next(&mut self) -> Token {
        std::mem::replace(&mut self.token, self.lexer.next())
    }

    fn ptr_expr(&mut self) -> &'a Expr<'a> {
        self.arena.alloc(self.expr())
    }

    fn expr(&mut self) -> Expr<'a> {
        let left = self.unit_expr();
        self.bin_expr(left, 0)
    }

    fn bin_expr(&mut self, mut fold: Expr<'a>, min_prec: u8) -> Expr<'a> {
        loop {
            let Some(prec) = self.token.kind.precedence() else {
                break;
            };

            if prec <= min_prec {
                break;
            }

            let op = self.next().kind;

            let right = self.unit_expr();
            let right = self.bin_expr(right, prec);
            let right = self.arena.alloc(right);
            let left = self.arena.alloc(fold);

            if let Some(op) = op.assign_op() {
                self.flag_idents(*left, idfl::MUTABLE);
                let right = Expr::BinOp { left, op, right };
                fold = Expr::BinOp {
                    left,
                    op: TokenKind::Assign,
                    right: self.arena.alloc(right),
                };
            } else {
                fold = Expr::BinOp { left, right, op };
                if op == TokenKind::Assign {
                    self.flag_idents(*left, idfl::MUTABLE);
                }
            }
        }

        fold
    }

    fn resolve_ident(&mut self, token: Token, decl: bool) -> (Ident, u16) {
        let is_ct = self.token.kind == TokenKind::CtIdent;
        let name = self.lexer.slice(token.range());

        if let Some(builtin) = bt::from_str(name) {
            return (builtin, 0);
        }

        let (i, id) = match self
            .idents
            .iter_mut()
            .enumerate()
            .rfind(|(_, elem)| self.lexer.slice(ident::range(elem.ident)) == name)
        {
            Some((_, elem)) if decl && elem.declared => {
                self.report(format_args!("redeclaration of identifier: {name}"))
            }
            Some((i, elem)) => {
                elem.flags += 1;
                (i, elem)
            }
            None => {
                let id = ident::new(token.start, name.len() as _);
                self.idents.push(ScopeIdent {
                    ident:    id,
                    declared: false,
                    flags:    0,
                });
                (self.idents.len() - 1, self.idents.last_mut().unwrap())
            }
        };

        id.declared |= decl;
        id.flags |= idfl::COMPTIME * is_ct as u32;
        if id.declared && self.ns_bound > i {
            id.flags |= idfl::COMPTIME;
            self.captured.push(id.ident);
        }

        (id.ident, idfl::index(id.flags))
    }

    fn move_str(&mut self, range: Token) -> &'a str {
        self.arena.alloc_str(self.lexer.slice(range.range()))
    }

    fn unit_expr(&mut self) -> Expr<'a> {
        use {Expr as E, TokenKind as T};
        let frame = self.idents.len();
        let token = self.next();
        let prev_boundary = self.ns_bound;
        let prev_captured = self.captured.len();
        let mut expr = match token.kind {
            T::Driective if self.lexer.slice(token.range()) == "use" => {
                self.expect_advance(TokenKind::LParen);
                let str = self.expect_advance(TokenKind::String);
                self.expect_advance(TokenKind::RParen);
                let path = self.lexer.slice(str.range()).trim_matches('"');

                E::Mod {
                    pos:  token.start,
                    path: self.arena.alloc_str(path),
                    id:   match (self.loader)(path, self.path) {
                        Ok(id) => id,
                        Err(e) => self.report(format_args!("error loading dependency: {e:#?}")),
                    },
                }
            }
            T::Driective => E::Directive {
                pos:  token.start,
                name: self.move_str(token),
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, Self::expr)
                },
            },
            T::True => E::Bool {
                pos:   token.start,
                value: true,
            },
            T::Struct => E::Struct {
                fields:   {
                    self.ns_bound = self.idents.len();
                    self.expect_advance(T::LBrace);
                    self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        (s.move_str(name), s.expr())
                    })
                },
                captured: {
                    self.ns_bound = prev_boundary;
                    self.captured[prev_captured..].sort_unstable();
                    let preserved = self.captured[prev_captured..].partition_dedup().0.len();
                    self.captured.truncate(prev_captured + preserved);
                    self.arena.alloc_slice(&self.captured[prev_captured..])
                },
                pos:      {
                    if self.ns_bound == 0 {
                        // we might save some memory
                        self.captured.clear();
                    }
                    token.start
                },
            },
            T::Ident | T::CtIdent => {
                let (id, index) = self.resolve_ident(token, self.token.kind == T::Decl);
                let name = self.move_str(token);
                E::Ident { name, id, index }
            }
            T::If => E::If {
                pos:   token.start,
                cond:  self.ptr_expr(),
                then:  self.ptr_expr(),
                else_: self.advance_if(T::Else).then(|| self.ptr_expr()),
            },
            T::Loop => E::Loop {
                pos:  token.start,
                body: self.ptr_expr(),
            },
            T::Break => E::Break { pos: token.start },
            T::Continue => E::Continue { pos: token.start },
            T::Return => E::Return {
                pos: token.start,
                val: (self.token.kind != T::Semi).then(|| self.ptr_expr()),
            },
            T::Fn => E::Closure {
                pos:  token.start,
                args: {
                    self.expect_advance(T::LParen);
                    self.collect_list(T::Comma, T::RParen, |s| {
                        let name = s.advance_ident();
                        let (id, index) = s.resolve_ident(name, true);
                        s.expect_advance(T::Colon);
                        Arg {
                            name: s.move_str(name),
                            id,
                            index,
                            ty: s.expr(),
                        }
                    })
                },
                ret:  {
                    self.expect_advance(T::Colon);
                    self.ptr_expr()
                },
                body: self.ptr_expr(),
            },
            T::Band | T::Mul | T::Xor => E::UnOp {
                pos: token.start,
                op:  token.kind,
                val: {
                    let expr = if token.kind == T::Xor {
                        let expr = self.expr();
                        self.arena.alloc(expr)
                    } else {
                        self.ptr_unit_expr()
                    };
                    if token.kind == T::Band {
                        self.flag_idents(*expr, idfl::REFERENCED);
                    }
                    expr
                },
            },
            T::LBrace => E::Block {
                pos:   token.start,
                stmts: self.collect_list(T::Semi, T::RBrace, Self::expr),
            },
            T::Number => E::Number {
                pos:   token.start,
                value: match self.lexer.slice(token.range()).parse() {
                    Ok(value) => value,
                    Err(e) => self.report(format_args!("invalid number: {e}")),
                },
            },
            T::LParen => {
                let expr = self.expr();
                self.expect_advance(T::RParen);
                expr
            }
            tok => self.report(format_args!("unexpected token: {tok:?}")),
        };

        loop {
            let token = self.token;
            if matches!(token.kind, T::LParen | T::Ctor | T::Dot | T::Tupl) {
                self.next();
            }

            expr = match token.kind {
                T::LParen => Expr::Call {
                    func: self.arena.alloc(expr),
                    args: self.collect_list(T::Comma, T::RParen, Self::expr),
                },
                T::Ctor => E::Ctor {
                    pos:    token.start,
                    ty:     Some(self.arena.alloc(expr)),
                    fields: self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        let val = s.expr();
                        (Some(s.move_str(name)), val)
                    }),
                },
                T::Tupl => E::Ctor {
                    pos:    token.start,
                    ty:     Some(self.arena.alloc(expr)),
                    fields: self.collect_list(T::Comma, T::RParen, |s| (None, s.expr())),
                },
                T::Dot => E::Field {
                    target: self.arena.alloc(expr),
                    field:  {
                        let token = self.expect_advance(T::Ident);
                        self.move_str(token)
                    },
                },
                _ => break,
            }
        }

        if matches!(token.kind, T::Return) {
            self.expect_advance(T::Semi);
        }

        if matches!(token.kind, T::Loop | T::LBrace | T::Fn) {
            self.pop_scope(frame);
        }

        expr
    }

    fn advance_ident(&mut self) -> Token {
        if matches!(self.token.kind, TokenKind::Ident | TokenKind::CtIdent) {
            self.next()
        } else {
            self.report(format_args!(
                "expected identifier, found {:?}",
                self.token.kind
            ))
        }
    }

    fn pop_scope(&mut self, frame: usize) {
        let mut undeclared_count = frame;
        for i in frame..self.idents.len() {
            if !&self.idents[i].declared {
                self.idents.swap(i, undeclared_count);
                undeclared_count += 1;
            }
        }

        self.idents
            .drain(undeclared_count..)
            .map(|ident| Symbol {
                name:  ident.ident,
                flags: ident.flags,
            })
            .collect_into(self.symbols);
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
        self.collect(|s| {
            s.advance_if(end).not().then(|| {
                let val = f(s);
                s.advance_if(delim);
                val
            })
        })
    }

    fn collect<T: Copy>(&mut self, mut f: impl FnMut(&mut Self) -> Option<T>) -> &'a [T] {
        let vec = std::iter::from_fn(|| f(self)).collect::<Vec<_>>();
        self.arena.alloc_slice(&vec)
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
            self.report(format_args!(
                "expected {:?}, found {:?}",
                kind, self.token.kind
            ));
        }
        self.next()
    }

    #[track_caller]
    fn report(&self, msg: impl std::fmt::Display) -> ! {
        self.report_pos(self.token.start, msg)
    }

    #[track_caller]
    fn report_pos(&self, pos: Pos, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.lexer.line_col(pos);
        eprintln!("{}:{}:{} => {}", self.path, line, col, msg);
        unreachable!();
    }

    fn flag_idents(&mut self, e: Expr<'a>, flags: IdentFlags) {
        match e {
            Expr::Ident { id, .. } => find_ident(&mut self.idents, id).flags |= flags,
            Expr::Field { target, .. } => self.flag_idents(*target, flags),
            _ => {}
        }
    }
}

fn find_ident(idents: &mut [ScopeIdent], id: Ident) -> &mut ScopeIdent {
    idents
        .binary_search_by_key(&id, |si| si.ident)
        .map(|i| &mut idents[i])
        .unwrap()
}

pub fn find_symbol(symbols: &[Symbol], id: Ident) -> &Symbol {
    symbols
        .binary_search_by_key(&id, |s| s.name)
        .map(|i| &symbols[i])
        .unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arg<'a> {
    pub name:  &'a str,
    pub id:    Ident,
    pub index: u16,
    pub ty:    Expr<'a>,
}

macro_rules! generate_expr {
    ($(#[$meta:meta])* $vis:vis enum $name:ident<$lt:lifetime> {$(
        $(#[$field_meta:meta])*
        $variant:ident {
            $($field:ident: $ty:ty,)*
        },
    )*}) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $name<$lt> {$(
            $variant {
                $($field: $ty,)*
            },
        )*}

        impl<$lt> $name<$lt> {
            pub fn pos(&self) -> Pos {
                #[allow(unused_variables)]
                match self {
                    $(Self::$variant { $($field),* } => generate_expr!(@first $(($field),)*).posi(self),)*
                }
            }

           pub fn used_bytes(&self) -> usize {
               match self {$(
                   Self::$variant { $($field,)* } => {
                        let fields = [$(($field as *const _ as usize - self as *const _ as usize, std::mem::size_of_val($field)),)*];
                        let (last, size) = fields.iter().copied().max().unwrap();
                        last + size
                   },
              )*}
           }
        }
    };

    (@first ($($first:tt)*), $($rest:tt)*) => { $($first)* };
    (@last ($($ign:tt)*), $($rest:tt)*) => { $($rest)* };
    (@last ($($last:tt)*),) => { $($last)* };
}

// it would be real nice if we could use relative pointers and still pattern match easily
generate_expr! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Expr<'a> {
        Break {
            pos: Pos,
        },
        Continue {
            pos: Pos,
        },
        Closure {
            pos:  Pos,
            args: &'a [Arg<'a>],
            ret:  &'a Self,
            body: &'a Self,
        },
        Call {
            func: &'a Self,
            args: &'a [Self],
        },
        Return {
            pos: Pos,
            val: Option<&'a Self>,
        },
        Ident {
            id:    Ident,
            name:  &'a str,
            index: u16,
        },
        Block {
            pos:   Pos,
            stmts: &'a [Self],
        },
        Number {
            pos:   Pos,
            value: u64,
        },
        BinOp {
            left:  &'a Self,
            op:    TokenKind,
            right: &'a Self,
        },
        If {
            pos:   Pos,
            cond:  &'a Self,
            then:  &'a Self,
            else_: Option<&'a Self>,
        },
        Loop {
            pos:  Pos,
            body: &'a Self,
        },
        UnOp {
            pos: Pos,
            op:  TokenKind,
            val: &'a Self,
        },
        Struct {
            pos:      Pos,
            fields:   &'a [(&'a str, Self)],
            captured: &'a [Ident],
        },
        Ctor {
            pos:    Pos,
            ty:     Option<&'a Self>,
            fields: &'a [(Option<&'a str>, Self)],
        },
        Field {
            target: &'a Self,
            field:  &'a str,
        },
        Bool {
            pos:   Pos,
            value: bool,
        },
        Directive {
            pos:  u32,
            name: &'a str,
            args: &'a [Self],
        },
        Mod {
            pos:  Pos,
            id:   FileId,
            path: &'a str,
        },
    }
}

trait Poser {
    fn posi(self, expr: &Expr) -> Pos;
}

impl Poser for Pos {
    fn posi(self, expr: &Expr) -> Pos {
        if matches!(expr, Expr::Ident { .. }) {
            ident::pos(self)
        } else {
            self
        }
    }
}

impl<'a> Poser for &Expr<'a> {
    fn posi(self, _: &Expr) -> Pos {
        self.pos()
    }
}

impl<'a> std::fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = const { Cell::new(0) };
        }

        fn fmt_list<T>(
            f: &mut std::fmt::Formatter,
            end: &str,
            list: &[T],
            fmt: impl Fn(&T, &mut std::fmt::Formatter) -> std::fmt::Result,
        ) -> std::fmt::Result {
            let first = &mut true;
            for expr in list {
                if !std::mem::take(first) {
                    write!(f, ", ")?;
                }
                fmt(expr, f)?;
            }
            write!(f, "{end}")
        }

        macro_rules! impl_parenter {
            ($($name:ident => $pat:pat,)*) => {
                $(
                    struct $name<'a>(&'a Expr<'a>);

                    impl<'a> std::fmt::Display for $name<'a> {
                        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                            if matches!(self.0, $pat) {
                                write!(f, "({})", self.0)
                            } else {
                                write!(f, "{}", self.0)
                            }
                        }
                    }
                )*
            };
        }

        impl_parenter! {
            Unary => Expr::BinOp { .. },
            Postfix => Expr::UnOp { .. } | Expr::BinOp { .. },
            Consecutive => Expr::UnOp { .. },
        }

        match *self {
            Self::Mod { path, .. } => write!(f, "@mod(\"{path}\")"),
            Self::Field { target, field } => write!(f, "{}.{field}", Postfix(target)),
            Self::Directive { name, args, .. } => {
                write!(f, "@{name}(")?;
                fmt_list(f, ")", args, std::fmt::Display::fmt)
            }
            Self::Struct { fields, .. } => {
                write!(f, "struct {{")?;
                fmt_list(f, "}", fields, |(name, val), f| write!(f, "{name}: {val}",))
            }
            Self::Ctor { ty, fields, .. } => {
                let (left, rith) = if fields.iter().any(|(name, _)| name.is_some()) {
                    ('{', '}')
                } else {
                    ('(', ')')
                };

                if let Some(ty) = ty {
                    write!(f, "{}", Unary(ty))?;
                }
                write!(f, ".{left}")?;
                let first = &mut true;
                for (name, val) in fields {
                    if !std::mem::take(first) {
                        write!(f, ", ")?;
                    }
                    if let Some(name) = name {
                        write!(f, "{name}: ")?;
                    }
                    write!(f, "{val}")?;
                }
                write!(f, "{rith}")
            }
            Self::UnOp { op, val, .. } => write!(f, "{op}{}", Unary(val)),
            Self::Break { .. } => write!(f, "break;"),
            Self::Continue { .. } => write!(f, "continue;"),
            Self::If {
                cond, then, else_, ..
            } => {
                write!(f, "if {cond} {}", Consecutive(then))?;
                if let Some(else_) = else_ {
                    write!(f, " else {else_}")?;
                }
                Ok(())
            }
            Self::Loop { body, .. } => write!(f, "loop {body}"),
            Self::Closure {
                ret, body, args, ..
            } => {
                write!(f, "fn(")?;
                fmt_list(f, "", args, |arg, f| write!(f, "{}: {}", arg.name, arg.ty))?;
                write!(f, "): {ret} {body}")
            }
            Self::Call { func, args } => {
                write!(f, "{}(", Postfix(func))?;
                fmt_list(f, ")", args, std::fmt::Display::fmt)
            }
            Self::Return { val: Some(val), .. } => write!(f, "return {val};"),
            Self::Return { val: None, .. } => write!(f, "return;"),
            Self::Ident { name, .. } => write!(f, "{name}"),
            Self::Block { stmts, .. } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| i.set(i.get() + 1));
                let res = (|| {
                    for stmt in stmts {
                        for _ in 0..INDENT.with(|i| i.get()) {
                            write!(f, "    ")?;
                        }
                        writeln!(f, "{stmt}")?;
                    }
                    Ok(())
                })();
                INDENT.with(|i| i.set(i.get() - 1));
                write!(f, "}}")?;
                res
            }
            Self::Number { value, .. } => write!(f, "{value}"),
            Self::Bool { value, .. } => write!(f, "{value}"),
            Self::BinOp { left, right, op } => {
                let display_branch = |f: &mut std::fmt::Formatter, expr: &Self| {
                    if let Self::BinOp { op: lop, .. } = expr
                        && op.precedence() > lop.precedence()
                    {
                        write!(f, "({expr})")
                    } else {
                        write!(f, "{expr}")
                    }
                };

                display_branch(f, left)?;
                write!(f, " {op} ")?;
                display_branch(f, right)
            }
        }
    }
}

#[repr(C)]
pub struct AstInner<T: ?Sized> {
    ref_count: AtomicUsize,
    mem:       ArenaChunk,
    exprs:     *const [Expr<'static>],

    pub path:    Box<str>,
    pub nlines:  LineMap,
    pub symbols: T,
}

impl AstInner<[Symbol]> {
    fn layout(syms: usize) -> std::alloc::Layout {
        std::alloc::Layout::new::<AstInner<()>>()
            .extend(std::alloc::Layout::array::<Symbol>(syms).unwrap())
            .unwrap()
            .0
    }

    fn new(content: &str, path: &str, loader: Loader) -> NonNull<Self> {
        let arena = Arena::default();
        let mut syms = Vec::new();
        let mut parser = Parser::new(&arena, &mut syms, loader);
        let exprs = parser.file(content, path) as *const [Expr<'static>];

        syms.sort_unstable_by_key(|s| s.name);

        let layout = Self::layout(syms.len());

        unsafe {
            let ptr = std::alloc::alloc(layout);
            let inner: *mut Self = std::ptr::from_raw_parts_mut(ptr as *mut _, syms.len());

            std::ptr::write(
                inner as *mut AstInner<()>,
                AstInner {
                    ref_count: AtomicUsize::new(1),
                    mem: arena.chunk.into_inner(),
                    exprs,
                    path: path.into(),
                    nlines: LineMap::new(content),
                    symbols: (),
                },
            );
            std::ptr::addr_of_mut!((*inner).symbols)
                .as_mut_ptr()
                .copy_from_nonoverlapping(syms.as_ptr(), syms.len());

            NonNull::new_unchecked(inner)
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct Ast(NonNull<AstInner<[Symbol]>>);

impl Ast {
    pub fn new(path: &str, content: &str, loader: Loader) -> Self {
        Self(AstInner::new(content, path, loader))
    }

    pub fn exprs(&self) -> &[Expr] {
        unsafe { &*self.inner().exprs }
    }

    fn inner(&self) -> &AstInner<[Symbol]> {
        unsafe { self.0.as_ref() }
    }

    pub fn find_decl(&self, id: Result<Ident, &str>) -> Option<(&Expr, Ident)> {
        self.exprs().iter().find_map(|expr| match expr {
            Expr::BinOp {
                left: &Expr::Ident { id: iden, name, .. },
                op: TokenKind::Decl,
                ..
            } if Ok(iden) == id || Err(name) == id => Some((expr, iden)),
            _ => None,
        })
    }
}

impl std::fmt::Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for expr in self.exprs() {
            writeln!(f, "{expr}\n")?;
        }
        Ok(())
    }
}

impl Default for Ast {
    fn default() -> Self {
        Self(AstInner::new("", "", &no_loader))
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
        // SAFETY: the pointer is or was a valid reference in the past, if it points within one of
        // arenas regions, it muts be walid, since arena does not give invalid pointers to its
        // allocations
        ArenaChunk::contains(from.mem.base, self.0.as_ptr() as _)
            .then(|| unsafe { { self.0 }.as_ref() })
    }
}

unsafe impl Send for Ast {}
unsafe impl Sync for Ast {}

impl Clone for Ast {
    fn clone(&self) -> Self {
        unsafe { self.0.as_ref() }
            .ref_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for Ast {
    fn drop(&mut self) {
        let inner = unsafe { self.0.as_ref() };
        if inner
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            == 1
        {
            unsafe { std::ptr::drop_in_place(self.0.as_ptr()) };

            let layout = AstInner::layout(inner.symbols.len());
            unsafe {
                std::alloc::dealloc(self.0.as_ptr() as _, layout);
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

#[derive(Default)]
pub struct Arena<'a> {
    chunk: UnsafeCell<ArenaChunk>,
    ph:    std::marker::PhantomData<&'a ()>,
}

impl<'a> Arena<'a> {
    pub fn alloc_str(&self, token: &str) -> &'a str {
        let ptr = self.alloc_slice(token.as_bytes());
        unsafe { std::str::from_utf8_unchecked(ptr) }
    }

    pub fn alloc(&self, expr: Expr<'a>) -> &'a Expr<'a> {
        let align = std::mem::align_of::<Expr<'a>>();
        let size = expr.used_bytes();
        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(size, align) };
        let ptr = self.alloc_low(layout);
        unsafe {
            ptr.cast::<u64>()
                .copy_from_nonoverlapping(NonNull::from(&expr).cast(), size / 8)
        };
        unsafe { ptr.cast::<Expr<'a>>().as_ref() }
    }

    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &'a [T] {
        if slice.is_empty() || std::mem::size_of::<T>() == 0 {
            return &mut [];
        }

        let layout = std::alloc::Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_low(layout);
        unsafe {
            ptr.as_ptr()
                .cast::<T>()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len())
        };
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as _, slice.len()) }
    }

    fn alloc_low(&self, layout: std::alloc::Layout) -> NonNull<u8> {
        assert!(layout.align() <= ArenaChunk::ALIGN);
        assert!(layout.size() <= ArenaChunk::CHUNK_SIZE);

        let chunk = unsafe { &mut *self.chunk.get() };

        if let Some(ptr) = chunk.alloc(layout) {
            return ptr;
        }

        unsafe {
            std::ptr::write(chunk, ArenaChunk::new(chunk.base));
        }

        chunk.alloc(layout).unwrap()
    }
}

struct ArenaChunk {
    base: *mut u8,
    end:  *mut u8,
}

impl Default for ArenaChunk {
    fn default() -> Self {
        Self {
            base: std::ptr::null_mut(),
            end:  std::ptr::null_mut(),
        }
    }
}

impl ArenaChunk {
    const CHUNK_SIZE: usize = 1 << 16;
    const ALIGN: usize = std::mem::align_of::<Self>();
    const NEXT_OFFSET: usize = Self::CHUNK_SIZE - std::mem::size_of::<*mut u8>();
    const LAYOUT: std::alloc::Layout =
        unsafe { std::alloc::Layout::from_size_align_unchecked(Self::CHUNK_SIZE, Self::ALIGN) };

    fn new(next: *mut u8) -> Self {
        let base = unsafe { std::alloc::alloc(Self::LAYOUT) };
        let end = unsafe { base.add(Self::NEXT_OFFSET) };
        Self::set_next(base, next);
        Self { base, end }
    }

    fn set_next(curr: *mut u8, next: *mut u8) {
        unsafe { std::ptr::write(curr.add(Self::NEXT_OFFSET) as *mut _, next) };
    }

    fn next(curr: *mut u8) -> *mut u8 {
        unsafe { std::ptr::read(curr.add(Self::NEXT_OFFSET) as *mut _) }
    }

    fn alloc(&mut self, layout: std::alloc::Layout) -> Option<NonNull<u8>> {
        let padding = self.end as usize - (self.end as usize & !(layout.align() - 1));
        let size = layout.size() + padding;
        if size > self.end as usize - self.base as usize {
            return None;
        }
        unsafe { self.end = self.end.sub(size) };
        unsafe { Some(NonNull::new_unchecked(self.end)) }
    }

    fn contains(base: *mut u8, arg: *mut u8) -> bool {
        !base.is_null()
            && ((unsafe { base.add(Self::CHUNK_SIZE) } > arg && base <= arg)
                || Self::contains(Self::next(base), arg))
    }
}

impl Drop for ArenaChunk {
    fn drop(&mut self) {
        log::inf!(
            "dropping chunk of size: {}",
            (Self::LAYOUT.size() - (self.end as usize - self.base as usize))
                * !self.end.is_null() as usize
        );
        let mut current = self.base;
        while !current.is_null() {
            let next = Self::next(current);
            unsafe { std::alloc::dealloc(current, Self::LAYOUT) };
            current = next;
            log::dbg!("deallocating full chunk");
        }
    }
}

#[cfg(test)]
mod tests {
    fn parse(input: &'static str, output: &mut String) {
        use std::fmt::Write;
        let arena = super::Arena::default();
        let mut symbols = Vec::new();
        let mut parser = super::Parser::new(&arena, &mut symbols, &super::no_loader);
        for expr in parser.file(input, "test") {
            writeln!(output, "{}", expr).unwrap();
        }
    }

    crate::run_tests! { parse:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
    }
}
