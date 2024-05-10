use {
    crate::parser::{self, Expr},
    std::rc::Rc,
};

type LabelId = u32;
type Reg = u8;
type MaskElem = u64;

const STACK_PTR: Reg = 254;
const ZERO: Reg = 0;
const RET_ADDR: Reg = 31;
const ELEM_WIDTH: usize = std::mem::size_of::<MaskElem>() * 8;

struct Frame {
    label:       LabelId,
    prev_relocs: usize,
    offset:      u32,
}

struct Reloc {
    id:     LabelId,
    offset: u32,
    size:   u16,
}

#[derive(Default)]
pub struct Func {
    code:   Vec<u8>,
    relocs: Vec<Reloc>,
}

impl Func {
    pub fn extend(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub fn offset(&mut self, id: LabelId, offset: u32, size: u16) {
        self.relocs.push(Reloc {
            id,
            offset: self.code.len() as u32 + offset,
            size,
        });
    }

    fn push(&mut self, value: Reg, size: usize) {
        self.st(value, STACK_PTR, 0, size as _);
        self.addi64(STACK_PTR, STACK_PTR, size as _);
    }

    fn pop(&mut self, value: Reg, size: usize) {
        self.addi64(STACK_PTR, STACK_PTR, (size as u64).wrapping_neg());
        self.ld(value, STACK_PTR, 0, size as _);
    }

    fn call(&mut self, func: LabelId) {
        self.jal(RET_ADDR, ZERO, func);
    }

    fn ret(&mut self) {
        self.jala(ZERO, RET_ADDR, 0);
    }

    fn prelude(&mut self, entry: LabelId) {
        self.call(entry);
        self.tx();
    }

    fn relocate(&mut self, labels: &[Label], shift: i64) {
        for reloc in self.relocs.drain(..) {
            let label = &labels[reloc.id as usize];
            let offset = if reloc.size == 8 {
                reloc.offset as i64
            } else {
                label.offset as i64 - reloc.offset as i64
            } + shift;

            let dest = &mut self.code[reloc.offset as usize..][..reloc.size as usize];
            match reloc.size {
                2 => dest.copy_from_slice(&(offset as i16).to_le_bytes()),
                4 => dest.copy_from_slice(&(offset as i32).to_le_bytes()),
                8 => dest.copy_from_slice(&(offset as i64).to_le_bytes()),
                _ => unreachable!(),
            };
        }
    }
}

#[derive(Default)]
pub struct RegAlloc {
    free: Vec<Reg>,
    // TODO:use 256 bit mask instead
    used: Vec<Reg>,
}

impl RegAlloc {
    fn init_caller(&mut self) {
        self.clear();
        self.free.extend(1..=31);
    }

    fn clear(&mut self) {
        self.free.clear();
        self.used.clear();
    }

    fn allocate(&mut self) -> Reg {
        let reg = self.free.pop().expect("TODO: we need to spill");
        if self.used.binary_search_by_key(&!reg, |&r| !r).is_err() {
            self.used.push(reg);
        }
        reg
    }

    fn free(&mut self, reg: Reg) {
        self.free.push(reg);
    }
}

struct Label {
    offset: u32,
    // TODO: use different stile of identifier that does not allocate, eg. index + length into a
    // file
    name:   Rc<str>,
}

pub struct Codegen<'a> {
    path:   &'a std::path::Path,
    ret:    Expr<'a>,
    gpa:    RegAlloc,
    code:   Func,
    temp:   Func,
    labels: Vec<Label>,
}

impl<'a> Codegen<'a> {
    pub fn new() -> Self {
        Self {
            path:   std::path::Path::new(""),
            ret:    Expr::Return { val: None },
            gpa:    Default::default(),
            code:   Default::default(),
            temp:   Default::default(),
            labels: Default::default(),
        }
    }

    pub fn file(
        &mut self,
        path: &'a std::path::Path,
        exprs: &'a [parser::Expr<'a>],
    ) -> std::fmt::Result {
        self.path = path;
        for expr in exprs {
            self.expr(expr, None);
        }
        Ok(())
    }

    fn expr(&mut self, expr: &'a parser::Expr<'a>, expeted: Option<Expr<'a>>) -> Option<Value<'a>> {
        use parser::Expr as E;
        match *expr {
            E::Decl {
                name,
                val: E::Closure { ret, body },
            } => {
                let frame = self.add_label(name);
                self.ret = **ret;
                self.expr(body, None);
                self.write_fn_prelude(frame);
                None
            }
            E::Return { val } => {
                if let Some(val) = val {
                    let val = self.expr(val, Some(self.ret)).unwrap();
                    if val.ty != self.ret {
                        panic!("expected {:?}, got {:?}", self.ret, val.ty);
                    }
                    match val.loc {
                        Loc::Reg(reg) => self.code.cp(1, reg),
                        Loc::Imm(imm) => self.code.li64(1, imm),
                    }
                }
                self.ret();
                None
            }
            E::Block { stmts } => {
                for stmt in stmts {
                    self.expr(stmt, None);
                }
                None
            }
            E::Number { value } => Some(Value {
                ty:  expeted.unwrap_or(Expr::Ident { name: "int" }),
                loc: Loc::Imm(value),
            }),
            ast => unimplemented!("{:?}", ast),
        }
    }

    fn get_or_reserve_label(&mut self, name: &str) -> LabelId {
        if let Some(label) = self.labels.iter().position(|l| l.name.as_ref() == name) {
            label as u32
        } else {
            self.labels.push(Label {
                offset: 0,
                name:   name.into(),
            });
            self.labels.len() as u32 - 1
        }
    }

    fn add_label(&mut self, name: &str) -> Frame {
        let offset = self.code.code.len() as u32;
        let label = if let Some(label) = self.labels.iter().position(|l| l.name.as_ref() == name) {
            self.labels[label].offset = offset;
            label as u32
        } else {
            self.labels.push(Label {
                offset,
                name: name.into(),
            });
            self.labels.len() as u32 - 1
        };

        Frame {
            label,
            prev_relocs: self.code.relocs.len(),
            offset,
        }
    }

    fn get_label(&self, name: &str) -> LabelId {
        self.labels
            .iter()
            .position(|l| l.name.as_ref() == name)
            .unwrap() as _
    }

    fn write_fn_prelude(&mut self, frame: Frame) {
        for &reg in self.gpa.used.clone().iter() {
            self.temp.push(reg, 8);
        }

        for reloc in &mut self.code.relocs[frame.prev_relocs..] {
            reloc.offset += self.temp.code.len() as u32;
        }

        self.code.code.splice(
            frame.offset as usize..frame.offset as usize,
            self.temp.code.drain(..),
        );
    }

    fn ret(&mut self) {
        for reg in self.gpa.used.clone().iter().rev() {
            self.code.pop(*reg, 8);
        }
        self.code.ret();
    }

    pub fn dump(mut self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        self.temp.prelude(self.get_label("main"));
        self.temp
            .relocate(&self.labels, self.temp.code.len() as i64);

        self.code.relocate(&self.labels, 0);
        out.write_all(&self.temp.code)?;
        out.write_all(&self.code.code)
    }
}

pub struct Value<'a> {
    ty:  Expr<'a>,
    loc: Loc,
}

pub enum Loc {
    Reg(Reg),
    Imm(u64),
}

#[cfg(test)]
mod tests {
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
        let mut out = Vec::new();
        codegen.dump(&mut out).unwrap();

        use std::fmt::Write;

        let mut stack = [0_u64; 1024];

        for (i, b) in out.iter().enumerate() {
            write!(output, "{:02x}", b).unwrap();
            if (i + 1) % 4 == 0 {
                writeln!(output).unwrap();
            }
        }
        writeln!(output).unwrap();

        let mut vm = unsafe {
            hbvm::Vm::<TestMem, 0>::new(TestMem, hbvm::mem::Address::new(out.as_ptr() as u64))
        };

        vm.write_reg(super::STACK_PTR, stack.as_mut_ptr() as u64);

        let stat = loop {
            match vm.run() {
                Ok(hbvm::VmRunOk::End) => break Ok(()),
                Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
                Err(e) => break Err(e),
            }
        };

        writeln!(output, "ret: {:?}", vm.read_reg(1)).unwrap();
        writeln!(output, "status: {:?}", stat).unwrap();
    }

    crate::run_tests! { generate:
        example => include_str!("../examples/main_fn.hb");
    }
}
