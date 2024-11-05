use {
    crate::{
        lexer::TokenKind,
        parser,
        son::{hbvm::HbvmBackend, Codegen, CodegenCtx},
    },
    alloc::string::String,
    core::{fmt::Write, hash::BuildHasher, ops::Range},
};

#[derive(Default)]
struct Rand(pub u64);

impl Rand {
    pub fn next(&mut self) -> u64 {
        self.0 = crate::FnvBuildHasher::default().hash_one(self.0);
        self.0
    }

    pub fn range(&mut self, min: u64, max: u64) -> u64 {
        self.next() % (max - min) + min
    }

    fn bool(&mut self) -> bool {
        self.next() % 2 == 0
    }
}

#[derive(Default)]
struct FuncGen {
    rand: Rand,
    buf: String,
    vars: u64,
}

impl FuncGen {
    fn gen(&mut self, seed: u64) -> &str {
        self.rand = Rand(seed);
        self.buf.clear();
        self.buf.push_str("main := fn(): void ");
        self.block().unwrap();
        &self.buf
    }

    fn block(&mut self) -> core::fmt::Result {
        let prev_vars = self.vars;
        self.buf.push('{');
        for _ in 0..self.rand.range(1, 10) {
            self.stmt()?;
        }
        self.buf.push('}');
        self.vars = prev_vars;

        Ok(())
    }

    fn stmt(&mut self) -> core::fmt::Result {
        match self.rand.range(0, 100) {
            0..4 => _ = self.block(),
            4..10 => {
                write!(self.buf, "var{} := ", self.vars)?;
                self.expr()?;
                self.vars += 1;
            }

            10..20 if self.vars != 0 => {
                write!(self.buf, "var{} = ", self.rand.range(0, self.vars))?;
                self.expr()?;
            }
            20..23 => {
                self.buf.push_str("if ");
                self.expr()?;
                self.block()?;
                if self.rand.bool() {
                    self.buf.push_str(" else ");
                    self.block()?;
                }
            }
            _ => {
                self.buf.push_str("return ");
                self.expr()?;
            }
        }

        self.buf.push(';');
        Ok(())
    }

    fn expr(&mut self) -> core::fmt::Result {
        match self.rand.range(0, 100) {
            0..80 => {
                write!(self.buf, "{}", self.rand.next())
            }
            80..90 if self.vars != 0 => {
                write!(self.buf, "var{}", self.rand.range(0, self.vars))
            }
            80..100 => {
                self.expr()?;
                let ops = [
                    TokenKind::Add,
                    TokenKind::Sub,
                    TokenKind::Mul,
                    TokenKind::Div,
                    TokenKind::Shl,
                    TokenKind::Eq,
                    TokenKind::Ne,
                    TokenKind::Lt,
                    TokenKind::Gt,
                    TokenKind::Le,
                    TokenKind::Ge,
                    TokenKind::Band,
                    TokenKind::Bor,
                    TokenKind::Xor,
                    TokenKind::Mod,
                    TokenKind::Shr,
                ];
                let op = ops[self.rand.range(0, ops.len() as u64) as usize];
                write!(self.buf, " {op} ")?;
                self.expr()
            }
            _ => unreachable!(),
        }
    }
}

pub fn fuzz(seed_range: Range<u64>) {
    let mut gen = FuncGen::default();
    let mut ctx = CodegenCtx::default();
    for i in seed_range {
        ctx.clear();
        let src = gen.gen(i);
        let parsed = parser::Ast::new("fuzz", src, &mut ctx.parser, &mut parser::no_loader);

        assert!(ctx.parser.errors.get_mut().is_empty());

        let mut backend = HbvmBackend::default();
        let mut cdg = Codegen::new(&mut backend, core::slice::from_ref(&parsed), &mut ctx);
        cdg.generate(0);
    }
}
