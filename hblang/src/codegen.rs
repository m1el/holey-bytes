use {
    crate::{
        instrs, lexer,
        parser::{self, Expr},
    },
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

struct StackReloc {
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

    fn encode(&mut self, (len, instr): (usize, [u8; instrs::MAX_SIZE])) {
        let name = instrs::NAMES[instr[0] as usize];
        println!(
            "{}: {}",
            name,
            instr
                .iter()
                .take(len)
                .skip(1)
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        );
        self.code.extend_from_slice(&instr[..len]);
    }

    fn push(&mut self, value: Reg, size: usize) {
        self.subi64(STACK_PTR, STACK_PTR, size as _);
        self.encode(instrs::st(value, STACK_PTR, 0, size as _));
    }

    fn pop(&mut self, value: Reg, size: usize) {
        self.encode(instrs::ld(value, STACK_PTR, 0, size as _));
        self.encode(instrs::addi64(STACK_PTR, STACK_PTR, size as _));
    }

    fn subi64(&mut self, dest: Reg, src: Reg, imm: u64) {
        self.encode(instrs::addi64(dest, src, imm.wrapping_neg()));
    }

    fn call(&mut self, func: LabelId) {
        self.offset(func, 3, 4);
        self.encode(instrs::jal(RET_ADDR, ZERO, 0));
    }

    fn ret(&mut self) {
        self.pop(RET_ADDR, 8);
        self.encode(instrs::jala(ZERO, RET_ADDR, 0));
    }

    fn prelude(&mut self, entry: LabelId) {
        self.call(entry);
        self.encode(instrs::tx());
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

struct Variable<'a> {
    name:   Rc<str>,
    offset: u64,
    ty:     Expr<'a>,
}

pub struct Codegen<'a> {
    path:       &'a std::path::Path,
    ret:        Expr<'a>,
    gpa:        RegAlloc,
    code:       Func,
    temp:       Func,
    labels:     Vec<Label>,
    stack_size: u64,
    vars:       Vec<Variable<'a>>,

    stack_relocs: Vec<StackReloc>,
}

impl<'a> Codegen<'a> {
    pub fn new() -> Self {
        Self {
            path:       std::path::Path::new(""),
            ret:        Expr::Return { val: None },
            gpa:        Default::default(),
            code:       Default::default(),
            temp:       Default::default(),
            labels:     Default::default(),
            stack_size: 0,
            vars:       Default::default(),

            stack_relocs: Default::default(),
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

    fn loc_to_reg(&mut self, loc: Loc) -> Reg {
        match loc {
            Loc::Reg(reg) => reg,
            Loc::Imm(imm) => {
                let reg = self.gpa.allocate();
                self.code.encode(instrs::li64(reg, imm));
                reg
            }
            Loc::Stack(offset) => {
                let reg = self.gpa.allocate();
                self.load_stack(reg, offset, 8);
                reg
            }
        }
    }

    fn alloc_stack(&mut self, size: u32) -> u64 {
        let offset = self.stack_size;
        self.stack_size += size as u64;
        offset
    }

    fn store_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.stack_relocs.push(StackReloc {
            offset: self.code.code.len() as u32 + 3,
            size,
        });
        self.code.encode(instrs::st(reg, STACK_PTR, offset, size));
    }

    fn load_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.stack_relocs.push(StackReloc {
            offset: self.code.code.len() as u32 + 3,
            size,
        });
        self.code.encode(instrs::ld(reg, STACK_PTR, offset, size));
    }

    fn reloc_stack(&mut self, stack_height: u64) {
        for reloc in self.stack_relocs.drain(..) {
            let dest = &mut self.code.code[reloc.offset as usize..][..reloc.size as usize];
            let value = u64::from_ne_bytes(dest.try_into().unwrap());
            let offset = stack_height - value;
            dest.copy_from_slice(&offset.to_ne_bytes());
        }
    }

    fn expr(&mut self, expr: &'a parser::Expr<'a>, expeted: Option<Expr<'a>>) -> Option<Value<'a>> {
        use {lexer::TokenKind as T, parser::Expr as E};
        match *expr {
            E::Decl {
                name,
                val: E::Closure { ret, body },
            } => {
                let frame = self.add_label(name);
                self.gpa.init_caller();
                self.ret = **ret;
                self.expr(body, None);
                let stack = std::mem::take(&mut self.stack_size);
                self.reloc_stack(stack);
                self.write_fn_prelude(frame);
                None
            }
            E::Decl { name, val } => {
                let val = self.expr(val, None).unwrap();
                let reg = self.loc_to_reg(val.loc);
                let offset = self.alloc_stack(8);
                self.decl_var(name, offset, val.ty);
                self.store_stack(reg, offset, 8);
                None
            }
            E::Ident { name } => {
                let var = self.vars.iter().find(|v| v.name.as_ref() == name).unwrap();
                Some(Value {
                    ty:  var.ty,
                    loc: Loc::Stack(var.offset),
                })
            }
            E::Return { val } => {
                if let Some(val) = val {
                    let val = self.expr(val, Some(self.ret)).unwrap();
                    if val.ty != self.ret {
                        panic!("expected {:?}, got {:?}", self.ret, val.ty);
                    }
                    self.assign(
                        Value {
                            ty:  self.ret,
                            loc: Loc::Reg(1),
                        },
                        val,
                    );
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
            E::BinOp { left, op, right } => {
                let left = self.expr(left, expeted).unwrap();
                let right = self.expr(right, Some(left.ty)).unwrap();

                let op = match op {
                    T::Plus => instrs::add64,
                    T::Minus => instrs::sub64,
                    T::Star => instrs::mul64,
                    T::FSlash => |reg0, reg1, reg2| instrs::diru64(reg0, ZERO, reg1, reg2),
                    T::Assign => return self.assign(left, right),
                    _ => unimplemented!("{:#?}", op),
                };

                let lhs = self.loc_to_reg(left.loc);
                let rhs = self.loc_to_reg(right.loc);

                self.code.encode(op(lhs, lhs, rhs));
                self.gpa.free(rhs);

                Some(Value {
                    ty:  left.ty,
                    loc: Loc::Reg(lhs),
                })
            }
            ast => unimplemented!("{:#?}", ast),
        }
    }

    fn assign(&mut self, left: Value<'a>, right: Value<'a>) -> Option<Value<'a>> {
        let rhs = self.loc_to_reg(right.loc);
        match left.loc {
            Loc::Reg(reg) => self.code.encode(instrs::cp(reg, rhs)),
            Loc::Stack(offset) => self.store_stack(rhs, offset, 8),
            _ => unimplemented!(),
        }
        self.gpa.free(rhs);
        Some(left)
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
        self.temp.push(RET_ADDR, 8);
        for &reg in self.gpa.used.clone().iter() {
            self.temp.push(reg, 8);
        }
        self.temp.subi64(STACK_PTR, STACK_PTR, self.stack_size as _);

        for reloc in &mut self.code.relocs[frame.prev_relocs..] {
            reloc.offset += self.temp.code.len() as u32;
        }

        self.code.code.splice(
            frame.offset as usize..frame.offset as usize,
            self.temp.code.drain(..),
        );
    }

    fn ret(&mut self) {
        self.stack_relocs.push(StackReloc {
            offset: self.code.code.len() as u32 + 3,
            size:   8,
        });
        self.code.encode(instrs::addi64(STACK_PTR, STACK_PTR, 0));
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

    fn decl_var(&mut self, name: &str, offset: u64, ty: Expr<'a>) {
        self.vars.push(Variable {
            name: name.into(),
            offset,
            ty,
        });
    }
}

pub struct Value<'a> {
    ty:  Expr<'a>,
    loc: Loc,
}

pub enum Loc {
    Reg(Reg),
    Imm(u64),
    Stack(u64),
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

        let mut vm = unsafe {
            hbvm::Vm::<TestMem, 0>::new(TestMem, hbvm::mem::Address::new(out.as_ptr() as u64))
        };

        vm.write_reg(
            super::STACK_PTR,
            unsafe { stack.as_mut_ptr().add(stack.len()) } as u64,
        );

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
        arithmetic => include_str!("../examples/arithmetic.hb");
        variables => include_str!("../examples/variables.hb");
    }
}
