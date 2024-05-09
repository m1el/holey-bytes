use {crate::parser, std::fmt::Write};

const STACK_PTR: &str = "r254";
const ZERO: &str = "r0";
const RET_ADDR: &str = "r31";

pub struct Codegen<'a> {
    path: &'a std::path::Path,
    code: String,
    data: String,
}

impl<'a> Codegen<'a> {
    pub fn new(path: &'a std::path::Path) -> Self {
        Self {
            path,
            code: String::new(),
            data: String::new(),
        }
    }

    pub fn file(&mut self, exprs: &[parser::Expr]) -> std::fmt::Result {
        for expr in exprs {
            self.expr(expr)?;
        }
        Ok(())
    }

    fn expr(&mut self, expr: &parser::Expr) -> std::fmt::Result {
        use parser::Expr as E;
        match expr {
            E::Decl {
                name,
                val:
                    E::Closure {
                        ret: E::Ident { name: "void" },
                        body,
                    },
            } => {
                writeln!(self.code, "{name}:")?;
                self.expr(body)
            }
            E::Return { val: None } => self.ret(),
            E::Block { stmts } => {
                for stmt in stmts {
                    self.expr(stmt)?;
                }
                Ok(())
            }
            ast => unimplemented!("{:?}", ast),
        }
    }

    fn stack_push(&mut self, value: impl std::fmt::Display, size: usize) -> std::fmt::Result {
        writeln!(self.code, "    st {value}, {STACK_PTR}, {ZERO}, {size}")?;
        writeln!(
            self.code,
            "    addi{} {STACK_PTR}, {STACK_PTR}, {size}",
            size * 8
        )
    }

    fn stack_pop(&mut self, value: impl std::fmt::Display, size: usize) -> std::fmt::Result {
        writeln!(
            self.code,
            "    subi{} {STACK_PTR}, {STACK_PTR}, {size}",
            size * 8
        )?;
        writeln!(self.code, "    ld {value}, {STACK_PTR}, {ZERO}, {size}")
    }

    fn call(&mut self, func: impl std::fmt::Display) -> std::fmt::Result {
        self.stack_push(&func, 8)?;
        self.global_jump(func)
    }

    fn ret(&mut self) -> std::fmt::Result {
        self.stack_pop(RET_ADDR, 8)?;
        self.global_jump(RET_ADDR)
    }

    fn global_jump(&mut self, label: impl std::fmt::Display) -> std::fmt::Result {
        writeln!(self.code, "    jala {ZERO}, {label}, 0")
    }

    pub fn dump(&mut self, mut out: impl std::fmt::Write) -> std::fmt::Result {
        writeln!(out, "start:")?;
        writeln!(out, "    jala {ZERO}, main, 0")?;
        writeln!(out, "    tx")?;
        writeln!(out, "{}", self.code)?;
        writeln!(out, "{}", self.data)
    }
}

#[cfg(test)]
mod tests {
    fn generate(input: &'static str, output: &mut String) {
        let mut parser = super::parser::Parser::new(input, std::path::Path::new("test"));
        let exprs = parser.file();
        let mut codegen = super::Codegen::new(std::path::Path::new("test"));
        codegen.file(&exprs).unwrap();
        codegen.dump(output).unwrap();
    }

    crate::run_tests! { generate:
        example => include_str!("../examples/main_fn.hb");
    }
}
