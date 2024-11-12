use {
    super::{AssemblySpec, Backend, Nid, Node, Nodes},
    crate::{
        lexer::TokenKind,
        parser, reg,
        son::{debug_assert_matches, write_reloc, Kind, MEM},
        ty::{self, Loc, Module},
        utils::{BitSet, Ent, EntVec, Vc},
        Offset, Reloc, Size, TypedReloc, Types,
    },
    alloc::{boxed::Box, collections::BTreeMap, string::String, vec::Vec},
    core::mem,
    hbbytecode::{self as instrs, *},
};

mod my_regalloc;
mod their_regalloc;

struct FuncDt {
    offset: Offset,
    // TODO: change to indices into common vec
    relocs: Vec<TypedReloc>,
    code: Vec<u8>,
}

impl Default for FuncDt {
    fn default() -> Self {
        Self { offset: u32::MAX, relocs: Default::default(), code: Default::default() }
    }
}

struct GlobalDt {
    offset: Offset,
}

impl Default for GlobalDt {
    fn default() -> Self {
        Self { offset: u32::MAX }
    }
}

#[derive(Default)]
struct Assembler {
    frontier: Vec<ty::Id>,
    globals: Vec<ty::Global>,
    funcs: Vec<ty::Func>,
}

#[derive(Default)]
pub struct HbvmBackend {
    funcs: EntVec<ty::Func, FuncDt>,
    globals: EntVec<ty::Global, GlobalDt>,
    asm: Assembler,
    ralloc: their_regalloc::Regalloc,
    ralloc_my: my_regalloc::Res,

    ret_relocs: Vec<Reloc>,
    relocs: Vec<TypedReloc>,
    jump_relocs: Vec<(Nid, Reloc)>,
    code: Vec<u8>,
    offsets: Vec<Offset>,
}

impl HbvmBackend {
    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        emit(&mut self.code, instr);
    }
}

impl Backend for HbvmBackend {
    fn assemble_bin(&mut self, entry: ty::Func, types: &Types, to: &mut Vec<u8>) {
        to.extend([0u8; HEADER_SIZE]);

        binary_prelude(to);
        let AssemblySpec { code_length, data_length, entry } =
            self.assemble_reachable(entry, types, to);

        let exe = AbleOsExecutableHeader {
            magic_number: [0x15, 0x91, 0xD2],
            executable_version: 0,
            code_length,
            data_length,
            debug_length: 0,
            config_length: 0,
            metadata_length: 0,
        };
        Reloc::new(HEADER_SIZE, 3, 4).apply_jump(to, entry, 0);

        unsafe { *to.as_mut_ptr().cast::<AbleOsExecutableHeader>() = exe }
    }

    fn assemble_reachable(
        &mut self,
        from: ty::Func,
        types: &Types,
        to: &mut Vec<u8>,
    ) -> AssemblySpec {
        debug_assert!(self.asm.frontier.is_empty());
        debug_assert!(self.asm.funcs.is_empty());
        debug_assert!(self.asm.globals.is_empty());

        self.globals.shadow(types.ins.globals.len());

        self.asm.frontier.push(ty::Kind::Func(from).compress());
        while let Some(itm) = self.asm.frontier.pop() {
            match itm.expand() {
                ty::Kind::Func(func) => {
                    let fuc = &mut self.funcs[func];
                    debug_assert!(!fuc.code.is_empty());
                    if fuc.offset != u32::MAX {
                        continue;
                    }
                    fuc.offset = 0;
                    self.asm.funcs.push(func);
                    self.asm.frontier.extend(fuc.relocs.iter().map(|r| r.target));
                }
                ty::Kind::Global(glob) => {
                    let glb = &mut self.globals[glob];
                    if glb.offset != u32::MAX {
                        continue;
                    }
                    glb.offset = 0;
                    self.asm.globals.push(glob);
                }
                _ => unreachable!(),
            }
        }

        let init_len = to.len();

        for &func in &self.asm.funcs {
            let fuc = &mut self.funcs[func];
            fuc.offset = to.len() as _;
            debug_assert!(!fuc.code.is_empty());
            to.extend(&fuc.code);
        }

        let code_length = to.len() - init_len;

        for global in self.asm.globals.drain(..) {
            self.globals[global].offset = to.len() as _;
            to.extend(&types.ins.globals[global].data);
        }

        let data_length = to.len() - code_length - init_len;

        for func in self.asm.funcs.drain(..) {
            let fuc = &self.funcs[func];
            for rel in &fuc.relocs {
                let offset = match rel.target.expand() {
                    ty::Kind::Func(fun) => self.funcs[fun].offset,
                    ty::Kind::Global(glo) => self.globals[glo].offset,
                    _ => unreachable!(),
                };
                rel.reloc.apply_jump(to, offset, fuc.offset);
            }
        }

        AssemblySpec {
            code_length: code_length as _,
            data_length: data_length as _,
            entry: self.funcs[from].offset,
        }
    }

    fn disasm<'a>(
        &'a self,
        mut sluce: &[u8],
        eca_handler: &mut dyn FnMut(&mut &[u8]),
        types: &'a Types,
        files: &'a [parser::Ast],
        output: &mut String,
    ) -> Result<(), hbbytecode::DisasmError<'a>> {
        use hbbytecode::DisasmItem;
        let functions = types
            .ins
            .funcs
            .iter()
            .zip(self.funcs.iter())
            .filter(|(_, f)| f.offset != u32::MAX)
            .map(|(f, fd)| {
                let name = if f.file != Module::default() {
                    let file = &files[f.file.index()];
                    file.ident_str(f.name)
                } else {
                    "target_fn"
                };
                (fd.offset, (name, fd.code.len() as u32, DisasmItem::Func))
            })
            .chain(
                types
                    .ins
                    .globals
                    .iter()
                    .zip(self.globals.iter())
                    .filter(|(_, g)| g.offset != u32::MAX)
                    .map(|(g, gd)| {
                        let name = if g.file == Module::default() {
                            core::str::from_utf8(&g.data).unwrap_or("invalid utf-8")
                        } else {
                            let file = &files[g.file.index()];
                            file.ident_str(g.name)
                        };
                        (gd.offset, (name, g.data.len() as Size, DisasmItem::Global))
                    }),
            )
            .collect::<BTreeMap<_, _>>();
        hbbytecode::disasm(&mut sluce, &functions, output, eca_handler)
    }

    fn emit_ct_body(
        &mut self,
        id: ty::Func,
        nodes: &mut Nodes,
        tys: &Types,
        files: &[parser::Ast],
    ) {
        self.emit_body(id, nodes, tys, files);
        let fd = &mut self.funcs[id];
        fd.code.truncate(fd.code.len() - instrs::jala(0, 0, 0).0);
        emit(&mut fd.code, instrs::tx());
    }

    fn emit_body(&mut self, id: ty::Func, nodes: &mut Nodes, tys: &Types, files: &[parser::Ast]) {
        let sig = tys.ins.funcs[id].sig.unwrap();

        debug_assert!(self.code.is_empty());

        self.offsets.clear();
        self.offsets.resize(nodes.values.len(), Offset::MAX);

        let mut stack_size = 0;
        '_compute_stack: {
            let mems = mem::take(&mut nodes[MEM].outputs);
            for &stck in mems.iter() {
                if !matches!(nodes[stck].kind, Kind::Stck | Kind::Arg) {
                    debug_assert_matches!(
                        nodes[stck].kind,
                        Kind::Phi | Kind::Return | Kind::Load | Kind::Call { .. } | Kind::Stre
                    );
                    continue;
                }
                stack_size += tys.size_of(nodes[stck].ty);
                self.offsets[stck as usize] = stack_size;
            }
            for &stck in mems.iter() {
                if !matches!(nodes[stck].kind, Kind::Stck | Kind::Arg) {
                    continue;
                }
                self.offsets[stck as usize] = stack_size - self.offsets[stck as usize];
            }
            nodes[MEM].outputs = mems;
        }

        let (saved, tail) = self.emit_body_code(nodes, sig, tys, files);
        //let (saved, tail) = self.emit_body_code_my(nodes, sig, tys, files);

        if let Some(last_ret) = self.ret_relocs.last()
            && last_ret.offset as usize == self.code.len() - 5
            && self
                .jump_relocs
                .last()
                .map_or(true, |&(r, _)| self.offsets[r as usize] as usize != self.code.len())
        {
            self.code.truncate(self.code.len() - 5);
            self.ret_relocs.pop();
        }

        // FIXME: maybe do this incrementally
        for (nd, rel) in self.jump_relocs.drain(..) {
            let offset = self.offsets[nd as usize];
            //debug_assert!(offset < self.code.len() as u32 - 1);
            rel.apply_jump(&mut self.code, offset, 0);
        }

        let end = self.code.len();
        for ret_rel in self.ret_relocs.drain(..) {
            ret_rel.apply_jump(&mut self.code, end as _, 0);
        }

        let mut stripped_prelude_size = 0;
        '_close_function: {
            let pushed = (saved as i64 + !tail as i64) * 8;
            let stack = stack_size as i64;

            let add_len = instrs::addi64(0, 0, 0).0;
            let st_len = instrs::st(0, 0, 0, 0).0;

            match (pushed, stack) {
                (0, 0) => {
                    stripped_prelude_size = add_len + st_len;
                    self.code.drain(0..stripped_prelude_size);
                    break '_close_function;
                }
                (0, stack) => {
                    write_reloc(&mut self.code, 3, -stack, 8);
                    stripped_prelude_size = st_len;
                    let end = add_len + st_len;
                    self.code.drain(add_len..end);
                    self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, stack as _));
                    break '_close_function;
                }
                _ => {}
            }

            write_reloc(&mut self.code, 3, -(pushed + stack), 8);
            write_reloc(&mut self.code, 3 + 8 + 3, stack, 8);
            write_reloc(&mut self.code, 3 + 8 + 3 + 8, pushed, 2);

            self.emit(instrs::ld(
                reg::RET_ADDR + tail as u8,
                reg::STACK_PTR,
                stack as _,
                pushed as _,
            ));
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, (pushed + stack) as _));
        }
        self.relocs.iter_mut().for_each(|r| r.reloc.offset -= stripped_prelude_size as u32);
        if sig.ret != ty::Id::NEVER {
            self.emit(instrs::jala(reg::ZERO, reg::RET_ADDR, 0));
        }

        self.funcs.shadow(tys.ins.funcs.len());
        self.funcs[id].code = mem::take(&mut self.code);
        self.funcs[id].relocs = mem::take(&mut self.relocs);

        debug_assert_eq!(self.ret_relocs.len(), 0);
        debug_assert_eq!(self.relocs.len(), 0);
        debug_assert_eq!(self.jump_relocs.len(), 0);
        debug_assert_eq!(self.code.len(), 0);
    }
}

impl Nodes {
    fn reschedule_block(&mut self, from: Nid, outputs: &mut Vc) {
        // NOTE: this code is horible
        let from = Some(&from);
        let mut buf = Vec::with_capacity(outputs.len());
        let mut seen = BitSet::default();
        seen.clear(self.values.len());

        for &o in outputs.iter() {
            if !self.is_cfg(o) {
                continue;
            }

            seen.set(o);

            let mut cursor = buf.len();
            buf.push(o);
            while let Some(&n) = buf.get(cursor) {
                for &i in &self[n].inputs[1..] {
                    if from == self[i].inputs.first()
                        && self[i]
                            .outputs
                            .iter()
                            .all(|&o| self[o].inputs.first() != from || seen.get(o))
                        && seen.set(i)
                    {
                        for &o in outputs.iter().filter(|&&n| n == i) {
                            buf.push(o);
                        }
                    }
                }
                cursor += 1;
            }
        }

        for &o in outputs.iter() {
            if !seen.set(o) {
                continue;
            }
            let mut cursor = buf.len();
            for &o in outputs.iter().filter(|&&n| n == o) {
                buf.push(o);
            }
            while let Some(&n) = buf.get(cursor) {
                for &i in &self[n].inputs[1..] {
                    if from == self[i].inputs.first()
                        && self[i]
                            .outputs
                            .iter()
                            .all(|&o| self[o].inputs.first() != from || seen.get(o))
                        && seen.set(i)
                    {
                        for &o in outputs.iter().filter(|&&n| n == i) {
                            buf.push(o);
                        }
                    }
                }
                cursor += 1;
            }
        }

        buf.sort_by_key(|&n| !self.is_cfg(n));

        if outputs.len() != buf.len() {
            panic!("{:?} {:?}", outputs, buf);
        }
        outputs.copy_from_slice(&buf);
    }

    fn is_never_used(&self, nid: Nid, tys: &Types) -> bool {
        let node = &self[nid];
        match node.kind {
            Kind::CInt { .. } => node.outputs.iter().all(|&o| {
                matches!(self[o].kind, Kind::BinOp { op }
                        if op.imm_binop(self[o].ty).is_some()
                            && self.is_const(self[o].inputs[2])
                            && op.cond_op(self[o].ty).is_none())
            }),
            Kind::BinOp { op: TokenKind::Add | TokenKind::Sub } => {
                self[node.inputs[1]].lock_rc != 0
                    || (self.is_const(node.inputs[2])
                        && node.outputs.iter().all(|&n| self[n].uses_direct_offset_of(nid, tys)))
            }
            Kind::BinOp { op } => {
                op.cond_op(node.ty).is_some()
                    && node.outputs.iter().all(|&n| self[n].kind == Kind::If)
            }
            Kind::Stck if tys.size_of(node.ty) == 0 => true,
            Kind::Stck | Kind::Arg => node.outputs.iter().all(|&n| {
                self[n].uses_direct_offset_of(nid, tys)
                    || (matches!(self[n].kind, Kind::BinOp { op: TokenKind::Add })
                        && self.is_never_used(n, tys))
            }),
            Kind::Load { .. } => node.ty.loc(tys) == Loc::Stack,
            _ => false,
        }
    }
}

impl Node {
    fn uses_direct_offset_of(&self, nid: Nid, tys: &Types) -> bool {
        ((self.kind == Kind::Stre && self.inputs[2] == nid)
            || (self.kind == Kind::Load && self.inputs[1] == nid))
            && self.ty.loc(tys) == Loc::Reg
    }
}

impl TokenKind {
    fn cmp_against(self) -> Option<u64> {
        Some(match self {
            TokenKind::Le | TokenKind::Gt => 1,
            TokenKind::Ne | TokenKind::Eq => 0,
            TokenKind::Ge | TokenKind::Lt => (-1i64) as _,
            _ => return None,
        })
    }

    fn float_cmp(self, ty: ty::Id) -> Option<fn(u8, u8, u8) -> EncodedInstr> {
        if !ty.is_float() {
            return None;
        }
        let size = ty.simple_size().unwrap();

        let ops = match self {
            TokenKind::Gt => [instrs::fcmpgt32, instrs::fcmpgt64],
            TokenKind::Lt => [instrs::fcmplt32, instrs::fcmplt64],
            _ => return None,
        };

        Some(ops[size.ilog2() as usize - 2])
    }

    #[expect(clippy::type_complexity)]
    fn cond_op(self, ty: ty::Id) -> Option<(fn(u8, u8, i16) -> EncodedInstr, bool)> {
        if ty.is_float() {
            return None;
        }
        let signed = ty.is_signed();
        Some((
            match self {
                Self::Le if signed => instrs::jgts,
                Self::Le => instrs::jgtu,
                Self::Lt if signed => instrs::jlts,
                Self::Lt => instrs::jltu,
                Self::Ge if signed => instrs::jlts,
                Self::Ge => instrs::jltu,
                Self::Gt if signed => instrs::jgts,
                Self::Gt => instrs::jgtu,
                Self::Eq => instrs::jne,
                Self::Ne => instrs::jeq,
                _ => return None,
            },
            matches!(self, Self::Lt | TokenKind::Gt),
        ))
    }

    fn binop(self, ty: ty::Id) -> Option<fn(u8, u8, u8) -> EncodedInstr> {
        let size = ty.simple_size().unwrap();
        if ty.is_integer() || ty == ty::Id::BOOL || ty.is_pointer() {
            macro_rules! div { ($($op:ident),*) => {[$(|a, b, c| $op(a, 0, b, c)),*]}; }
            macro_rules! rem { ($($op:ident),*) => {[$(|a, b, c| $op(0, a, b, c)),*]}; }
            let signed = ty.is_signed();

            let ops = match self {
                Self::Add => [add8, add16, add32, add64],
                Self::Sub => [sub8, sub16, sub32, sub64],
                Self::Mul => [mul8, mul16, mul32, mul64],
                Self::Div if signed => div!(dirs8, dirs16, dirs32, dirs64),
                Self::Div => div!(diru8, diru16, diru32, diru64),
                Self::Mod if signed => rem!(dirs8, dirs16, dirs32, dirs64),
                Self::Mod => rem!(diru8, diru16, diru32, diru64),
                Self::Band => return Some(and),
                Self::Bor => return Some(or),
                Self::Xor => return Some(xor),
                Self::Shl => [slu8, slu16, slu32, slu64],
                Self::Shr if signed => [srs8, srs16, srs32, srs64],
                Self::Shr => [sru8, sru16, sru32, sru64],
                _ => return None,
            };

            Some(ops[size.ilog2() as usize])
        } else {
            debug_assert!(ty.is_float(), "{self} {ty:?}");
            let ops = match self {
                Self::Add => [fadd32, fadd64],
                Self::Sub => [fsub32, fsub64],
                Self::Mul => [fmul32, fmul64],
                Self::Div => [fdiv32, fdiv64],
                _ => return None,
            };
            Some(ops[size.ilog2() as usize - 2])
        }
    }

    fn imm_binop(self, ty: ty::Id) -> Option<fn(u8, u8, u64) -> EncodedInstr> {
        macro_rules! def_op {
            ($name:ident |$a:ident, $b:ident, $c:ident| $($tt:tt)*) => {
                macro_rules! $name {
                    ($$($$op:ident),*) => {
                        [$$(
                            |$a, $b, $c: u64| $$op($($tt)*),
                        )*]
                    }
                }
            };
        }

        if ty.is_float() {
            return None;
        }

        def_op!(basic_op | a, b, c | a, b, c as _);
        def_op!(sub_op | a, b, c | a, b, c.wrapping_neg() as _);

        let signed = ty.is_signed();
        let ops = match self {
            Self::Add => basic_op!(addi8, addi16, addi32, addi64),
            Self::Sub => sub_op!(addi8, addi16, addi32, addi64),
            Self::Mul => basic_op!(muli8, muli16, muli32, muli64),
            Self::Band => return Some(andi),
            Self::Bor => return Some(ori),
            Self::Xor => return Some(xori),
            Self::Shr if signed => basic_op!(srsi8, srsi16, srsi32, srsi64),
            Self::Shr => basic_op!(srui8, srui16, srui32, srui64),
            Self::Shl => basic_op!(slui8, slui16, slui32, slui64),
            _ => return None,
        };

        let size = ty.simple_size().unwrap();
        Some(ops[size.ilog2() as usize])
    }

    fn unop(&self, dst: ty::Id, src: ty::Id) -> Option<fn(u8, u8) -> EncodedInstr> {
        let src_idx =
            src.simple_size().unwrap_or_else(|| panic!("{:?}", src.expand())).ilog2() as usize;
        Some(match self {
            Self::Sub => [
                |a, b| sub8(a, reg::ZERO, b),
                |a, b| sub16(a, reg::ZERO, b),
                |a, b| sub32(a, reg::ZERO, b),
                |a, b| sub64(a, reg::ZERO, b),
            ][src_idx],
            Self::Not => instrs::not,
            Self::Float if dst.is_float() && src.is_integer() => {
                debug_assert_matches!(
                    (dst.simple_size(), src.simple_size()),
                    (Some(4 | 8), Some(8))
                );
                [instrs::itf32, instrs::itf64][src_idx - 2]
            }
            Self::Number if src.is_float() && dst.is_integer() => {
                [|a, b| instrs::fti32(a, b, 1), |a, b| instrs::fti64(a, b, 1)][src_idx - 2]
            }
            Self::Number if src.is_signed() && dst.is_integer() => {
                [instrs::sxt8, instrs::sxt16, instrs::sxt32][src_idx]
            }
            Self::Number if (src.is_unsigned() || src == ty::Id::BOOL) && dst.is_integer() => [
                |a, b| instrs::andi(a, b, 0xff),
                |a, b| instrs::andi(a, b, 0xffff),
                |a, b| instrs::andi(a, b, 0xffffffff),
            ][src_idx],
            Self::Float if dst.is_float() && src.is_float() => {
                [instrs::fc32t64, |a, b| instrs::fc64t32(a, b, 1)][src_idx - 2]
            }
            _ => return None,
        })
    }
}

type EncodedInstr = (usize, [u8; instrs::MAX_SIZE]);
fn emit(out: &mut Vec<u8>, (len, instr): EncodedInstr) {
    out.extend_from_slice(&instr[..len]);
}

fn binary_prelude(to: &mut Vec<u8>) {
    emit(to, instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
    emit(to, instrs::tx());
}

#[derive(Default)]
pub struct LoggedMem {
    pub mem: hbvm::mem::HostMemory,
    op_buf: Vec<hbbytecode::Oper>,
    disp_buf: String,
    prev_instr: Option<hbbytecode::Instr>,
}

impl LoggedMem {
    unsafe fn display_instr<T>(&mut self, instr: hbbytecode::Instr, addr: hbvm::mem::Address) {
        let novm: *const hbvm::Vm<Self, 0> = core::ptr::null();
        let offset = core::ptr::addr_of!((*novm).memory) as usize;
        let regs = unsafe {
            &*core::ptr::addr_of!(
                (*(((self as *mut _ as *mut u8).sub(offset)) as *const hbvm::Vm<Self, 0>))
                    .registers
            )
        };

        let mut bytes = core::slice::from_raw_parts(
            (addr.get() - 1) as *const u8,
            core::mem::size_of::<T>() + 1,
        );
        use core::fmt::Write;
        hbbytecode::parse_args(&mut bytes, instr, &mut self.op_buf).unwrap();
        debug_assert!(bytes.is_empty());
        self.disp_buf.clear();
        write!(self.disp_buf, "{:<10}", format!("{instr:?}")).unwrap();
        for (i, op) in self.op_buf.drain(..).enumerate() {
            if i != 0 {
                write!(self.disp_buf, ", ").unwrap();
            }
            write!(self.disp_buf, "{op:?}").unwrap();
            if let hbbytecode::Oper::R(r) = op {
                write!(self.disp_buf, "({})", regs[r as usize].0).unwrap()
            }
        }
        log::trace!("read-typed: {:x}: {}", addr.get(), self.disp_buf);
    }
}

impl hbvm::mem::Memory for LoggedMem {
    unsafe fn load(
        &mut self,
        addr: hbvm::mem::Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), hbvm::mem::LoadError> {
        log::trace!(
            "load: {:x} {}",
            addr.get(),
            AsHex(core::slice::from_raw_parts(addr.get() as *const u8, count))
        );
        self.mem.load(addr, target, count)
    }

    unsafe fn store(
        &mut self,
        addr: hbvm::mem::Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), hbvm::mem::StoreError> {
        log::trace!(
            "store: {:x} {}",
            addr.get(),
            AsHex(core::slice::from_raw_parts(source, count))
        );
        self.mem.store(addr, source, count)
    }

    unsafe fn prog_read<T: Copy + 'static>(&mut self, addr: hbvm::mem::Address) -> T {
        if log::log_enabled!(log::Level::Trace) {
            if core::any::TypeId::of::<u8>() == core::any::TypeId::of::<T>() {
                if let Some(instr) = self.prev_instr {
                    self.display_instr::<()>(instr, addr);
                }
                self.prev_instr = hbbytecode::Instr::try_from(*(addr.get() as *const u8)).ok();
            } else {
                let instr = self.prev_instr.take().unwrap();
                self.display_instr::<T>(instr, addr);
            }
        }

        self.mem.prog_read(addr)
    }
}

struct AsHex<'a>(&'a [u8]);

impl core::fmt::Display for AsHex<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for &b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

const VM_STACK_SIZE: usize = 1024 * 64;

pub struct Comptime {
    pub vm: hbvm::Vm<LoggedMem, { 1024 * 10 }>,
    stack: Box<[u8; VM_STACK_SIZE]>,
    pub code: Vec<u8>,
    depth: usize,
}

impl Comptime {
    pub fn run(&mut self, ret_loc: &mut [u8], offset: u32) -> u64 {
        self.vm.write_reg(reg::RET, ret_loc.as_mut_ptr() as u64);
        let prev_pc = self.push_pc(offset);
        loop {
            match self.vm.run().expect("TODO") {
                hbvm::VmRunOk::End => break,
                hbvm::VmRunOk::Timer => todo!(),
                hbvm::VmRunOk::Ecall => todo!(),
                hbvm::VmRunOk::Breakpoint => todo!(),
            }
        }
        self.pop_pc(prev_pc);

        if let len @ 1..=8 = ret_loc.len() {
            ret_loc.copy_from_slice(&self.vm.read_reg(reg::RET).0.to_ne_bytes()[..len])
        }

        self.vm.read_reg(reg::RET).0
    }

    pub fn reset(&mut self) {
        let ptr = unsafe { self.stack.as_mut_ptr().cast::<u8>().add(VM_STACK_SIZE) as u64 };
        self.vm.registers.fill(hbvm::value::Value(0));
        self.vm.write_reg(reg::STACK_PTR, ptr);
        self.vm.pc = hbvm::mem::Address::new(self.code.as_ptr() as u64 + HEADER_SIZE as u64);
    }

    fn push_pc(&mut self, offset: Offset) -> hbvm::mem::Address {
        let entry = &mut self.code[offset as usize] as *mut _ as _;
        core::mem::replace(&mut self.vm.pc, hbvm::mem::Address::new(entry))
            - self.code.as_ptr() as usize
    }

    fn pop_pc(&mut self, prev_pc: hbvm::mem::Address) {
        self.vm.pc = prev_pc + self.code.as_ptr() as usize;
    }

    pub fn clear(&mut self) {
        self.code.clear();
    }

    #[must_use]
    pub fn active(&self) -> bool {
        self.depth != 0
    }

    pub fn activate(&mut self) {
        self.depth += 1;
    }

    pub fn deactivate(&mut self) {
        self.depth -= 1;
    }
}

impl Default for Comptime {
    fn default() -> Self {
        let mut stack = Box::<[u8; VM_STACK_SIZE]>::new_uninit();
        let mut vm = hbvm::Vm::default();
        let ptr = unsafe { stack.as_mut_ptr().cast::<u8>().add(VM_STACK_SIZE) as u64 };
        vm.write_reg(reg::STACK_PTR, ptr);
        Self { vm, stack: unsafe { stack.assume_init() }, code: Default::default(), depth: 0 }
    }
}

const HEADER_SIZE: usize = core::mem::size_of::<AbleOsExecutableHeader>();

#[repr(packed)]
#[expect(dead_code)]
pub struct AbleOsExecutableHeader {
    magic_number: [u8; 3],
    executable_version: u32,

    code_length: u64,
    data_length: u64,
    debug_length: u64,
    config_length: u64,
    metadata_length: u64,
}

#[cfg(test)]
pub fn test_run_vm(out: &[u8], output: &mut String) {
    use core::fmt::Write;

    let mut stack = [0_u64; 1024 * 20];

    let mut vm = unsafe {
        hbvm::Vm::<_, { 1024 * 100 }>::new(
            LoggedMem::default(),
            hbvm::mem::Address::new(out.as_ptr() as u64).wrapping_add(HEADER_SIZE),
        )
    };

    vm.write_reg(reg::STACK_PTR, unsafe { stack.as_mut_ptr().add(stack.len()) } as u64);

    let stat = loop {
        match vm.run() {
            Ok(hbvm::VmRunOk::End) => break Ok(()),
            Ok(hbvm::VmRunOk::Ecall) => match vm.read_reg(2).0 {
                1 => writeln!(output, "ev: Ecall").unwrap(), // compatibility with a test
                69 => {
                    let [size, align] = [vm.read_reg(3).0 as usize, vm.read_reg(4).0 as usize];
                    let layout = core::alloc::Layout::from_size_align(size, align).unwrap();
                    let ptr = unsafe { alloc::alloc::alloc(layout) };
                    vm.write_reg(1, ptr as u64);
                }
                96 => {
                    let [ptr, size, align] = [
                        vm.read_reg(3).0 as usize,
                        vm.read_reg(4).0 as usize,
                        vm.read_reg(5).0 as usize,
                    ];

                    let layout = core::alloc::Layout::from_size_align(size, align).unwrap();
                    unsafe { alloc::alloc::dealloc(ptr as *mut u8, layout) };
                }
                3 => vm.write_reg(1, 42),
                8 => {}
                unknown => writeln!(output, "unknown ecall: {unknown:?}").unwrap(),
            },
            Ok(hbvm::VmRunOk::Timer) => {
                writeln!(output, "timed out").unwrap();
                break Ok(());
            }
            Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
            Err(e) => break Err(e),
        }
    };

    writeln!(output, "code size: {}", out.len() - HEADER_SIZE).unwrap();
    writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
    writeln!(output, "status: {:?}", stat).unwrap();
}
