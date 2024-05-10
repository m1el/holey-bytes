use {crate::parser, std::fmt::Write};

type Reg = u8;
type MaskElem = u64;

const STACK_PTR: Reg = 254;
const ZERO: Reg = 0;
const RET_ADDR: Reg = 31;
const ELEM_WIDTH: usize = std::mem::size_of::<MaskElem>() * 8;

#[derive(Default)]
pub struct RegAlloc {
    free: Vec<Reg>,
    // TODO:use 256 bit mask instead
    used: Vec<std::cmp::Reverse<Reg>>,
}

impl RegAlloc {
    fn callee_general_purpose() -> Self {
        Self {
            free: (32..=253).collect(),
            used: Vec::new(),
        }
    }

    fn allocate(&mut self) -> Reg {
        let reg = self.free.pop().expect("TODO: we need to spill");
        if self.used.binary_search(&std::cmp::Reverse(reg)).is_err() {
            self.used.push(std::cmp::Reverse(reg));
        }
        reg
    }

    fn free(&mut self, reg: Reg) {
        self.free.push(reg);
    }
}

pub struct Codegen<'a> {
    path:        &'a std::path::Path,
    gpa:         RegAlloc,
    code:        String,
    data:        String,
    prelude_buf: String,
}

impl<'a> Codegen<'a> {
    pub fn new() -> Self {
        Self {
            path:        std::path::Path::new(""),
            gpa:         RegAlloc::callee_general_purpose(),
            code:        String::new(),
            data:        String::new(),
            prelude_buf: String::new(),
        }
    }

    pub fn file(&mut self, path: &'a std::path::Path, exprs: &[parser::Expr]) -> std::fmt::Result {
        self.path = path;
        for expr in exprs {
            self.expr(expr)?;
        }
        Ok(())
    }

    fn expr(&mut self, expr: &parser::Expr) -> std::fmt::Result {
        use parser::Expr as E;
        match *expr {
            E::Decl {
                name,
                val:
                    E::Closure {
                        ret: E::Ident { name: "void" },
                        body,
                    },
            } => {
                writeln!(self.code, "{name}:")?;
                let fn_start = self.code.len();
                self.expr(body)?;
                self.write_fn_prelude(fn_start)
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

    fn write_fn_prelude(&mut self, fn_start: usize) -> std::fmt::Result {
        self.prelude_buf.clear();
        // TODO: avoid clone here
        for reg in self.gpa.used.clone().iter() {
            stack_push(&mut self.prelude_buf, reg.0, 8)?;
        }

        self.code.insert_str(fn_start, &self.prelude_buf);
        self.gpa = RegAlloc::callee_general_purpose();

        Ok(())
    }

    fn ret(&mut self) -> std::fmt::Result {
        for reg in self.gpa.used.clone().iter().rev() {
            stack_pop(&mut self.code, reg.0, 8)?;
        }
        ret(&mut self.code)
    }

    pub fn dump(self, mut out: impl std::fmt::Write) -> std::fmt::Result {
        prelude(&mut out)?;
        writeln!(out, "{}", self.code)?;
        writeln!(out, "{}", self.data)
    }
}

fn stack_push(out: &mut impl std::fmt::Write, value: Reg, size: usize) -> std::fmt::Result {
    writeln!(out, "    st r{value}, r{STACK_PTR}, r{ZERO}, {size}")?;
    writeln!(
        out,
        "    addi{} r{STACK_PTR}, r{STACK_PTR}, {size}",
        size * 8
    )
}

fn stack_pop(out: &mut impl std::fmt::Write, value: Reg, size: usize) -> std::fmt::Result {
    writeln!(
        out,
        "    subi{} r{STACK_PTR}, r{STACK_PTR}, {size}",
        size * 8
    )?;
    writeln!(out, "    ld r{value}, r{STACK_PTR}, r{ZERO}, {size}")
}

fn call(out: &mut impl std::fmt::Write, func: &str) -> std::fmt::Result {
    stack_push(out, RET_ADDR, 8)?;
    jump_label(out, func)?;
    stack_pop(out, RET_ADDR, 8)
}

fn ret(out: &mut impl std::fmt::Write) -> std::fmt::Result {
    writeln!(out, "    jala r{ZERO}, r{RET_ADDR}, 0")
}

fn jump_label(out: &mut impl std::fmt::Write, label: &str) -> std::fmt::Result {
    writeln!(out, "    jal r{RET_ADDR}, r{ZERO}, {label}")
}

fn prelude(out: &mut impl std::fmt::Write) -> std::fmt::Result {
    writeln!(out, "start:")?;
    writeln!(out, "    jal r{RET_ADDR}, r{ZERO}, main")?;
    writeln!(out, "    tx")
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    struct TestMem;

    impl hbvm::mem::Memory for TestMem {
        #[inline]
        unsafe fn load(
            &mut self,
            addr: hbvm::mem::Address,
            target: *mut u8,
            count: usize,
        ) -> Result<(), hbvm::mem::LoadError> {
            unsafe { core::ptr::copy(addr.get() as *const u8, target, count) }
            Ok(())
        }

        #[inline]
        unsafe fn store(
            &mut self,
            addr: hbvm::mem::Address,
            source: *const u8,
            count: usize,
        ) -> Result<(), hbvm::mem::StoreError> {
            unsafe { core::ptr::copy(source, addr.get() as *mut u8, count) }
            Ok(())
        }

        #[inline]
        unsafe fn prog_read<T: Copy>(&mut self, addr: hbvm::mem::Address) -> T {
            unsafe { core::ptr::read(addr.get() as *const T) }
        }
    }

    fn generate(input: &'static str, output: &mut String) {
        let path = std::path::Path::new("test");
        let arena = crate::parser::Arena::default();
        let mut buffer = Vec::new();
        let mut parser = super::parser::Parser::new(input, path, &arena, &mut buffer);
        let exprs = parser.file();
        let mut codegen = super::Codegen::new();
        codegen.file(path, &exprs).unwrap();
        codegen.dump(&mut *output).unwrap();

        let mut proc = std::process::Command::new("/usr/bin/hbas")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        proc.stdin
            .as_mut()
            .unwrap()
            .write_all(output.as_bytes())
            .unwrap();
        let out = proc.wait_with_output().unwrap();

        if !out.status.success() {
            panic!(
                "hbas failed with status: {}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        } else {
            use std::fmt::Write;

            let mut stack = [0_u64; 1024];

            for b in &out.stdout {
                writeln!(output, "{:02x}", b).unwrap();
            }

            let mut vm = unsafe {
                hbvm::Vm::<TestMem, 0>::new(
                    TestMem,
                    hbvm::mem::Address::new(out.stdout.as_ptr() as u64),
                )
            };

            vm.write_reg(super::STACK_PTR, stack.as_mut_ptr() as u64);

            let stat = loop {
                match vm.run() {
                    Ok(hbvm::VmRunOk::End) => break Ok(()),
                    Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
                    Err(e) => break Err(e),
                }
            };

            writeln!(output, "ret: {:?}", vm.read_reg(0)).unwrap();
            writeln!(output, "status: {:?}", stat).unwrap();
        }
    }

    crate::run_tests! { generate:
        example => include_str!("../examples/main_fn.hb");
    }
}
