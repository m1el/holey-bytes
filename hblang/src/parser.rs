use std::{cell::Cell, ops::Not, ptr::NonNull};

use crate::{
    codegen::bt,
    ident::{self, Ident},
    lexer::{Lexer, Token, TokenKind},
};

pub type Pos = u32;

pub type IdentFlags = u32;

pub const MUTABLE: IdentFlags = 1 << std::mem::size_of::<IdentFlags>() * 8 - 1;
pub const REFERENCED: IdentFlags = 1 << std::mem::size_of::<IdentFlags>() * 8 - 2;

pub fn ident_flag_index(flag: IdentFlags) -> u32 {
    flag & !(MUTABLE | REFERENCED)
}

struct ScopeIdent<'a> {
    ident:    Ident,
    declared: bool,
    last:     &'a Cell<IdentFlags>,
}

pub struct Parser<'a, 'b> {
    path:       &'a str,
    lexer:      Lexer<'a>,
    arena:      &'b Arena<'a>,
    token:      Token,
    idents:     Vec<ScopeIdent<'a>>,
    referening: bool,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(arena: &'b Arena<'a>) -> Self {
        let mut lexer = Lexer::new("");
        let token = lexer.next();
        Self {
            lexer,
            token,
            path: "",
            arena,
            idents: Vec::new(),
            referening: false,
        }
    }

    pub fn file(&mut self, input: &'a str, path: &'a str) -> &'a [Expr<'a>] {
        self.path = path;
        self.lexer = Lexer::new(input);
        self.token = self.lexer.next();

        let f = self.collect(|s| (s.token.kind != TokenKind::Eof).then(|| s.expr()));
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
            let right = &*self.arena.alloc(right);
            let left = &*self.arena.alloc(fold);

            if let Some(op) = op.assign_op() {
                fold.mark_mut();
                let right = Expr::BinOp { left, op, right };
                fold = Expr::BinOp {
                    left,
                    op: TokenKind::Assign,
                    right: self.arena.alloc(right),
                };
            } else {
                fold = Expr::BinOp { left, right, op };
                if op == TokenKind::Assign {
                    fold.mark_mut();
                }
            }
        }

        fold
    }

    fn try_resolve_builtin(name: &str) -> Option<Ident> {
        // FIXME: we actually do this the second time in the codegen
        Some(match name {
            "int" | "i64" => bt::INT,
            "i8" => bt::I8,
            "i16" => bt::I16,
            "i32" => bt::I32,
            "u8" => bt::U8,
            "u16" => bt::U16,
            "uint" | "u32" => bt::U32,
            "bool" => bt::BOOL,
            "void" => bt::VOID,
            "never" => bt::NEVER,
            _ => return None,
        })
    }

    fn resolve_ident(&mut self, token: Token, decl: bool) -> (Ident, Option<&'a Cell<IdentFlags>>) {
        let name = self.lexer.slice(token.range());

        if let Some(builtin) = Self::try_resolve_builtin(name) {
            return (builtin, None);
        }

        let id = match self
            .idents
            .iter_mut()
            .rfind(|elem| self.lexer.slice(ident::range(elem.ident)) == name)
        {
            Some(elem) if decl && elem.declared => {
                self.report(format_args!("redeclaration of identifier: {name}"))
            }
            Some(elem) => {
                elem.last.set(elem.last.get() + 1);
                elem
            }
            None => {
                let last = self.arena.alloc(Cell::new(0));
                let id = ident::new(token.start, name.len() as _);
                self.idents.push(ScopeIdent {
                    ident: id,
                    declared: false,
                    last,
                });
                self.idents.last_mut().unwrap()
            }
        };

        id.declared |= decl;
        id.last
            .set(id.last.get() | (REFERENCED * self.referening as u32));

        (id.ident, Some(id.last))
    }

    fn unit_expr(&mut self) -> Expr<'a> {
        use {Expr as E, TokenKind as T};
        let frame = self.idents.len();
        let token = self.next();
        let mut expr = match token.kind {
            T::Driective => E::Directive {
                pos:  token.start,
                name: self.lexer.slice(token.range()),
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
                pos:    token.start,
                fields: {
                    self.expect_advance(T::LBrace);
                    self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        let ty = s.expr();
                        (s.lexer.slice(name.range()), ty)
                    })
                },
            },
            T::Ident => {
                let (id, last) = self.resolve_ident(token, self.token.kind == T::Decl);
                E::Ident {
                    name: self.lexer.slice(token.range()),
                    id,
                    last,
                    index: last.map_or(0, |l| ident_flag_index(l.get())),
                }
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
                        let name = s.expect_advance(T::Ident);
                        let (id, last) = s.resolve_ident(name, true);
                        s.expect_advance(T::Colon);
                        Arg {
                            name: s.lexer.slice(name.range()),
                            id,
                            last,
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
            T::Band | T::Mul => E::UnOp {
                pos: token.start,
                op:  token.kind,
                val: match token.kind {
                    T::Band => self.referenced(Self::ptr_unit_expr),
                    _ => self.ptr_unit_expr(),
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
                    args: self
                        .calcel_ref()
                        .collect_list(T::Comma, T::RParen, Self::expr),
                },
                T::Ctor => E::Ctor {
                    pos:    token.start,
                    ty:     Some(self.arena.alloc(expr)),
                    fields: self.collect_list(T::Comma, T::RBrace, |s| {
                        let name = s.expect_advance(T::Ident);
                        s.expect_advance(T::Colon);
                        let val = s.expr();
                        (Some(s.lexer.slice(name.range())), val)
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
                        self.lexer.slice(token.range())
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

    fn referenced<T>(&mut self, f: impl Fn(&mut Self) -> T) -> T {
        if self.referening {
            self.report("cannot take reference of reference, (souwy)");
        }
        self.referening = true;
        let expr = f(self);
        self.referening = false;
        expr
    }

    fn pop_scope(&mut self, frame: usize) {
        let mut undeclared_count = frame;
        for i in frame..self.idents.len() {
            if !self.idents[i].declared {
                self.idents.swap(i, undeclared_count);
                undeclared_count += 1;
            }
        }

        self.idents.drain(undeclared_count..);
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

    fn report(&self, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.lexer.line_col(self.token.start);
        eprintln!("{}:{}:{} => {}", self.path, line, col, msg);
        unreachable!();
    }

    fn calcel_ref(&mut self) -> &mut Self {
        self.referening = false;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arg<'a> {
    pub name: &'a str,
    pub id:   Ident,
    pub last: Option<&'a Cell<IdentFlags>>,
    pub ty:   Expr<'a>,
}

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
        name:  &'a str,
        id:    Ident,
        index: u32,
        last:  Option<&'a Cell<IdentFlags>>,
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
        pos:    Pos,
        fields: &'a [(&'a str, Self)],
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
}

impl<'a> Expr<'a> {
    pub fn pos(&self) -> Pos {
        match self {
            Self::Call { func, .. } => func.pos(),
            Self::Ident { id, .. } => ident::pos(*id),
            Self::Break { pos }
            | Self::Directive { pos, .. }
            | Self::Continue { pos }
            | Self::Closure { pos, .. }
            | Self::Block { pos, .. }
            | Self::Number { pos, .. }
            | Self::Return { pos, .. }
            | Self::If { pos, .. }
            | Self::Loop { pos, .. }
            | Self::UnOp { pos, .. }
            | Self::Struct { pos, .. }
            | Self::Ctor { pos, .. }
            | Self::Bool { pos, .. } => *pos,
            Self::BinOp { left, .. } => left.pos(),
            Self::Field { target, .. } => target.pos(),
        }
    }

    fn mark_mut(&self) {
        match self {
            Self::Ident { last, .. } => _ = last.map(|l| l.set(l.get() | MUTABLE)),
            Self::Field { target, .. } => target.mark_mut(),
            _ => {}
        }
    }
}

impl<'a> std::fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = Cell::new(0);
        }

        fn fmt_list<'a, T>(
            f: &mut std::fmt::Formatter,
            end: &str,
            list: &'a [T],
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
            Self::Field { target, field } => {
                write!(f, "{}.{field}", Postfix(target))
            }
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

#[derive(Default)]
pub struct Arena<'a> {
    chunk: Cell<ArenaChunk>,
    ph:    std::marker::PhantomData<&'a ()>,
}

impl<'a> Arena<'a> {
    pub fn alloc_str(&self, token: &str) -> &'a str {
        let ptr = self.alloc_slice(token.as_bytes());
        unsafe { std::str::from_utf8_unchecked_mut(ptr) }
    }

    pub fn alloc<T>(&self, value: T) -> &'a mut T {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = self.alloc_low(layout);
        unsafe { ptr.cast::<T>().write(value) };
        unsafe { ptr.cast::<T>().as_mut() }
    }

    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &'a mut [T] {
        let layout = std::alloc::Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_low(layout);
        unsafe {
            ptr.as_ptr()
                .cast::<T>()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len())
        };
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as _, slice.len()) }
    }

    pub fn clear(&mut self) {
        let chunk = self.chunk.get_mut();
        if chunk.base.is_null() {
            return;
        }

        loop {
            let prev = ArenaChunk::prev(chunk.base);
            if prev.is_null() {
                break;
            }
            chunk.base = prev;
        }

        chunk.end = unsafe { chunk.base.add(ArenaChunk::PREV_OFFSET) };
    }

    fn with_chunk<R>(&self, f: impl FnOnce(&mut ArenaChunk) -> R) -> R {
        let mut chunk = self.chunk.get();
        let r = f(&mut chunk);
        self.chunk.set(chunk);
        r
    }

    fn alloc_low(&self, layout: std::alloc::Layout) -> NonNull<u8> {
        assert!(layout.align() <= ArenaChunk::ALIGN);
        assert!(layout.size() <= ArenaChunk::CHUNK_SIZE);
        self.with_chunk(|chunk| {
            if let Some(ptr) = chunk.alloc(layout) {
                return ptr;
            }

            if let Some(prev) = ArenaChunk::reset(ArenaChunk::prev(chunk.base)) {
                *chunk = prev;
            } else {
                *chunk = ArenaChunk::new(chunk.base);
            }

            chunk.alloc(layout).unwrap()
        })
    }
}

impl<'a> Drop for Arena<'a> {
    fn drop(&mut self) {
        use ArenaChunk as AC;

        let mut current = self.chunk.get().base;

        let mut prev = AC::prev(current);
        while !prev.is_null() {
            let next = AC::next(prev);
            unsafe { std::alloc::dealloc(prev, AC::LAYOUT) };
            prev = next;
        }

        while !current.is_null() {
            let next = AC::next(current);
            unsafe { std::alloc::dealloc(current, AC::LAYOUT) };
            current = next;
        }
    }
}

#[derive(Clone, Copy)]
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
    const PREV_OFFSET: usize = Self::NEXT_OFFSET - std::mem::size_of::<*mut u8>();
    const LAYOUT: std::alloc::Layout =
        unsafe { std::alloc::Layout::from_size_align_unchecked(Self::CHUNK_SIZE, Self::ALIGN) };

    fn new(next: *mut u8) -> Self {
        let base = unsafe { std::alloc::alloc(Self::LAYOUT) };
        let end = unsafe { base.add(Self::PREV_OFFSET) };
        if !next.is_null() {
            Self::set_prev(next, base);
        }
        Self::set_next(base, next);
        Self::set_prev(base, std::ptr::null_mut());
        Self { base, end }
    }

    fn set_next(curr: *mut u8, next: *mut u8) {
        unsafe { std::ptr::write(curr.add(Self::NEXT_OFFSET) as *mut _, next) };
    }

    fn set_prev(curr: *mut u8, prev: *mut u8) {
        unsafe { std::ptr::write(curr.add(Self::PREV_OFFSET) as *mut _, prev) };
    }

    fn next(curr: *mut u8) -> *mut u8 {
        unsafe { std::ptr::read(curr.add(Self::NEXT_OFFSET) as *mut _) }
    }

    fn prev(curr: *mut u8) -> *mut u8 {
        if curr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe { std::ptr::read(curr.add(Self::PREV_OFFSET) as *mut _) }
    }

    fn reset(prev: *mut u8) -> Option<Self> {
        if prev.is_null() {
            return None;
        }

        Some(Self {
            base: prev,
            end:  unsafe { prev.add(Self::CHUNK_SIZE) },
        })
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
}

#[cfg(test)]
mod tests {
    fn parse(input: &'static str, output: &mut String) {
        use std::fmt::Write;
        let mut arena = super::Arena::default();
        let mut parser = super::Parser::new(&arena);
        for expr in parser.file(input, "test") {
            writeln!(output, "{}", expr).unwrap();
        }
        arena.clear();
    }

    crate::run_tests! { parse:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
    }
}
