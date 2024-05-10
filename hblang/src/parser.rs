use std::{cell::Cell, ptr::NonNull};

use crate::lexer::{Lexer, Token, TokenKind};

type Ptr<'a, T> = &'a T;
type Slice<'a, T> = &'a [T];

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

    pub fn file(&mut self) -> Slice<'a, Expr<'a>> {
        self.collect(|s| (s.token.kind != TokenKind::Eof).then(|| s.expr()))
    }

    fn next(&mut self) -> Token {
        std::mem::replace(&mut self.token, self.lexer.next())
    }

    fn ptr_expr(&mut self) -> Ptr<'a, Expr<'a>> {
        self.arena.alloc(self.expr())
    }

    fn expr(&mut self) -> Expr<'a> {
        let token = self.next();
        let expr = match token.kind {
            TokenKind::Ident => {
                let name = self.arena.alloc_str(self.lexer.slice(token));
                if self.advance_if(TokenKind::Decl) {
                    let val = self.ptr_expr();
                    Expr::Decl { name, val }
                } else {
                    Expr::Ident { name }
                }
            }
            TokenKind::Return => Expr::Return {
                val: (self.token.kind != TokenKind::Semi).then(|| self.ptr_expr()),
            },
            TokenKind::Or => {
                self.expect_advance(TokenKind::Colon);
                let ret = self.ptr_expr();
                let body = self.ptr_expr();
                Expr::Closure { ret, body }
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
            tok => self.report(format_args!("unexpected token: {tok:?}")),
        };

        self.advance_if(TokenKind::Semi);

        expr
    }

    fn collect(&mut self, mut f: impl FnMut(&mut Self) -> Option<Expr<'a>>) -> Slice<'a, Expr<'a>> {
        let prev_len = self.expr_buf.len();
        while let Some(v) = f(self) {
            self.expr_buf.push(v);
        }
        let sl = self.arena.alloc_slice(&self.expr_buf[prev_len..]);
        self.expr_buf.truncate(prev_len);
        sl
    }

    fn advance_if(&mut self, kind: TokenKind) -> bool {
        if self.token.kind == kind {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect_advance(&mut self, kind: TokenKind) {
        if self.token.kind != kind {
            self.report(format_args!(
                "expected {:?}, found {:?}",
                kind, self.token.kind
            ));
        }
        self.next();
    }

    fn report(&self, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.lexer.line_col(self.token.start);
        eprintln!("{}:{}:{} => {}", self.path.display(), line, col, msg);
        unreachable!();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Expr<'a> {
    Decl {
        name: Ptr<'a, str>,
        val:  Ptr<'a, Expr<'a>>,
    },
    Closure {
        ret:  Ptr<'a, Expr<'a>>,
        body: Ptr<'a, Expr<'a>>,
    },
    Return {
        val: Option<Ptr<'a, Expr<'a>>>,
    },
    Ident {
        name: Ptr<'a, str>,
    },
    Block {
        stmts: Slice<'a, Expr<'a>>,
    },
    Number {
        value: u64,
    },
}

impl<'a> std::fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        thread_local! {
            static INDENT: Cell<usize> = Cell::new(0);
        }

        match self {
            Self::Decl { name, val } => write!(f, "{} := {}", name, val),
            Self::Closure { ret, body } => write!(f, "||: {} {}", ret, body),
            Self::Return { val: Some(val) } => write!(f, "return {};", val),
            Self::Return { val: None } => write!(f, "return;"),
            Self::Ident { name } => write!(f, "{}", name),
            Self::Block { stmts } => {
                writeln!(f, "{{")?;
                INDENT.with(|i| i.set(i.get() + 1));
                let res = crate::try_block(|| {
                    for stmt in *stmts {
                        for _ in 0..INDENT.with(|i| i.get()) {
                            write!(f, "    ")?;
                        }
                        writeln!(f, "{}", stmt)?;
                    }
                    Ok(())
                });
                INDENT.with(|i| i.set(i.get() - 1));
                write!(f, "}}")?;
                res
            }
            Self::Number { value } => write!(f, "{}", value),
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
    }
}
