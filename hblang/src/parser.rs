use std::{cell::Cell, ops::Not, ptr::NonNull};

use crate::lexer::{Lexer, Token, TokenKind};

pub struct Parser<'a, 'b> {
    path:     &'a std::path::Path,
    lexer:    Lexer<'a>,
    arena:    &'b Arena<'a>,
    expr_buf: &'b mut Vec<Expr<'a>>,
    token:    Token,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(
        input: &'a str,
        path: &'a std::path::Path,
        arena: &'b Arena<'a>,
        expr_buf: &'b mut Vec<Expr<'static>>,
    ) -> Self {
        let mut lexer = Lexer::new(input);
        let token = lexer.next();
        Self {
            lexer,
            token,
            path,
            arena,
            // we ensure its empty before returning form parse
            expr_buf: unsafe { std::mem::transmute(expr_buf) },
        }
    }

    pub fn file(&mut self) -> &'a [Expr<'a>] {
        self.collect(|s| (s.token.kind != TokenKind::Eof).then(|| s.expr()))
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

    fn bin_expr(&mut self, mut left: Expr<'a>, min_prec: u8) -> Expr<'a> {
        loop {
            let Some(prec) = self.token.kind.precedence() else {
                break;
            };

            if prec < min_prec {
                break;
            }

            let op = self.next().kind;
            let right = self.unit_expr();
            let right = self.bin_expr(right, prec);
            left = Expr::BinOp {
                left: self.arena.alloc(left),
                right: self.arena.alloc(right),
                op,
            };
        }

        left
    }

    fn unit_expr(&mut self) -> Expr<'a> {
        let token = self.next();
        let mut expr = match token.kind {
            TokenKind::Ident => {
                let name = self.arena.alloc_str(self.lexer.slice(token));
                if self.advance_if(TokenKind::Decl) {
                    let val = self.ptr_expr();
                    Expr::Decl { name, val }
                } else {
                    Expr::Ident { name }
                }
            }
            TokenKind::If => {
                let cond = self.ptr_expr();
                let then = self.ptr_expr();
                let else_ = self.advance_if(TokenKind::Else).then(|| self.ptr_expr());
                Expr::If { cond, then, else_ }
            }
            TokenKind::Loop => Expr::Loop {
                body: self.ptr_expr(),
            },
            TokenKind::Break => Expr::Break,
            TokenKind::Continue => Expr::Continue,
            TokenKind::Return => Expr::Return {
                val: (self.token.kind != TokenKind::Semi).then(|| self.ptr_expr()),
            },
            TokenKind::Or => {
                self.expect_advance(TokenKind::Colon);
                let ret = self.ptr_expr();
                let body = self.ptr_expr();
                Expr::Closure {
                    ret,
                    body,
                    args: &[],
                }
            }
            TokenKind::Bor => {
                let args = self.collect(|s| {
                    s.advance_if(TokenKind::Bor).not().then(|| {
                        let name = s.expect_advance(TokenKind::Ident);
                        let name = s.arena.alloc_str(s.lexer.slice(name));
                        s.expect_advance(TokenKind::Colon);
                        let val = s.expr();
                        s.advance_if(TokenKind::Comma);
                        (name, val)
                    })
                });
                self.expect_advance(TokenKind::Colon);
                let ret = self.ptr_expr();
                let body = self.ptr_expr();
                Expr::Closure { args, ret, body }
            }
            TokenKind::LBrace => Expr::Block {
                stmts: self.collect(|s| (!s.advance_if(TokenKind::RBrace)).then(|| s.expr())),
            },
            TokenKind::Number => Expr::Number {
                value: match self.lexer.slice(token).parse() {
                    Ok(value) => value,
                    Err(e) => self.report(format_args!("invalid number: {e}")),
                },
            },
            TokenKind::LParen => {
                let expr = self.expr();
                self.expect_advance(TokenKind::RParen);
                expr
            }
            tok => self.report(format_args!("unexpected token: {tok:?}")),
        };

        loop {
            expr = match self.token.kind {
                TokenKind::LParen => {
                    self.next();
                    Expr::Call {
                        func: self.arena.alloc(expr),
                        args: self.collect(|s| {
                            s.advance_if(TokenKind::RParen).not().then(|| {
                                let arg = s.expr();
                                s.advance_if(TokenKind::Comma);
                                arg
                            })
                        }),
                    }
                }
                _ => break,
            }
        }

        self.advance_if(TokenKind::Semi);

        expr
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
        eprintln!("{}:{}:{} => {}", self.path.display(), line, col, msg);
        unreachable!();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr<'a> {
    Break,
    Continue,
    Decl {
        name: &'a str,
        val:  &'a Expr<'a>,
    },
    Closure {
        args: &'a [(&'a str, Expr<'a>)],
        ret:  &'a Expr<'a>,
        body: &'a Expr<'a>,
    },
    Call {
        func: &'a Expr<'a>,
        args: &'a [Expr<'a>],
    },
    Return {
        val: Option<&'a Expr<'a>>,
    },
    Ident {
        name: &'a str,
    },
    Block {
        stmts: &'a [Expr<'a>],
    },
    Number {
        value: u64,
    },
    BinOp {
        left:  &'a Expr<'a>,
        op:    TokenKind,
        right: &'a Expr<'a>,
    },
    If {
        cond:  &'a Expr<'a>,
        then:  &'a Expr<'a>,
        else_: Option<&'a Expr<'a>>,
    },
    Loop {
        body: &'a Expr<'a>,
    },
}

impl<'a> std::fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = Cell::new(0);
        }

        match *self {
            Self::Break => write!(f, "break;"),
            Self::Continue => write!(f, "continue;"),
            Self::If { cond, then, else_ } => {
                write!(f, "if {} {}", cond, then)?;
                if let Some(else_) = else_ {
                    write!(f, " else {}", else_)?;
                }
                Ok(())
            }
            Self::Loop { body } => write!(f, "loop {}", body),
            Self::Decl { name, val } => write!(f, "{} := {}", name, val),
            Self::Closure { ret, body, args } => {
                write!(f, "|")?;
                let first = &mut true;
                for (name, val) in args {
                    if !std::mem::take(first) {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, val)?;
                }
                write!(f, "|: {} {}", ret, body)
            }
            Self::Call { func, args } => {
                write!(f, "{}(", func)?;
                let first = &mut true;
                for arg in args {
                    if !std::mem::take(first) {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Self::Return { val: Some(val) } => write!(f, "return {};", val),
            Self::Return { val: None } => write!(f, "return;"),
            Self::Ident { name } => write!(f, "{}", name),
            Self::Block { stmts } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| i.set(i.get() + 1));
                let res = (|| {
                    for stmt in stmts {
                        for _ in 0..INDENT.with(|i| i.get()) {
                            write!(f, "    ")?;
                        }
                        writeln!(f, "{}", stmt)?;
                    }
                    Ok(())
                })();
                INDENT.with(|i| i.set(i.get() - 1));
                write!(f, "}}")?;
                res
            }
            Self::Number { value } => write!(f, "{}", value),
            Self::BinOp { left, right, op } => {
                let display_branch = |f: &mut std::fmt::Formatter, expr: &Self| {
                    if let Self::BinOp { op: lop, .. } = expr
                        && op.precedence() > lop.precedence()
                    {
                        write!(f, "({})", expr)
                    } else {
                        write!(f, "{}", expr)
                    }
                };

                display_branch(f, left)?;
                write!(f, " {} ", op)?;
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
        let mut buffer = Vec::new();
        let mut parser =
            super::Parser::new(input, std::path::Path::new("test"), &arena, &mut buffer);
        for expr in parser.file() {
            writeln!(output, "{}", expr).unwrap();
        }
        arena.clear();
    }

    crate::run_tests! { parse:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
    }
}
