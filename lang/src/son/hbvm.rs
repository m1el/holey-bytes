use {
    super::{ItemCtx, Nid, Nodes, Pool, RallocBRef, Regalloc, ARG_START, NEVER, VOID},
    crate::{
        lexer::TokenKind,
        parser, reg,
        son::{write_reloc, Kind, MEM},
        task,
        ty::{self, Arg, Loc},
        utils::{BitSet, Vc},
        HashMap, Offset, PLoc, Reloc, Sig, Size, TypedReloc, Types,
    },
    alloc::{borrow::ToOwned, string::String, vec::Vec},
    core::{assert_matches::debug_assert_matches, mem},
    hbbytecode::{self as instrs, *},
    std::{boxed::Box, collections::BTreeMap},
};

impl Types {
    pub fn assemble(&mut self, to: &mut Vec<u8>) {
        to.extend([0u8; HEADER_SIZE]);

        binary_prelude(to);
        let exe = self.dump_reachable(0, to);
        Reloc::new(HEADER_SIZE, 3, 4).apply_jump(to, self.ins.funcs[0].offset, 0);

        unsafe { *to.as_mut_ptr().cast::<AbleOsExecutableHeader>() = exe }
    }

    pub fn dump_reachable(&mut self, from: ty::Func, to: &mut Vec<u8>) -> AbleOsExecutableHeader {
        debug_assert!(self.tmp.frontier.is_empty());
        debug_assert!(self.tmp.funcs.is_empty());
        debug_assert!(self.tmp.globals.is_empty());

        self.tmp.frontier.push(ty::Kind::Func(from).compress());
        while let Some(itm) = self.tmp.frontier.pop() {
            match itm.expand() {
                ty::Kind::Func(func) => {
                    let fuc = &mut self.ins.funcs[func as usize];
                    if task::is_done(fuc.offset) {
                        continue;
                    }
                    fuc.offset = 0;
                    self.tmp.funcs.push(func);
                    self.tmp.frontier.extend(fuc.relocs.iter().map(|r| r.target));
                }
                ty::Kind::Global(glob) => {
                    let glb = &mut self.ins.globals[glob as usize];
                    if task::is_done(glb.offset) {
                        continue;
                    }
                    glb.offset = 0;
                    self.tmp.globals.push(glob);
                }
                _ => unreachable!(),
            }
        }

        for &func in &self.tmp.funcs {
            let fuc = &mut self.ins.funcs[func as usize];
            fuc.offset = to.len() as _;
            debug_assert!(!fuc.code.is_empty());
            to.extend(&fuc.code);
        }

        let code_length = to.len();

        for global in self.tmp.globals.drain(..) {
            let global = &mut self.ins.globals[global as usize];
            global.offset = to.len() as _;
            to.extend(&global.data);
        }

        let data_length = to.len() - code_length;

        for func in self.tmp.funcs.drain(..) {
            let fuc = &self.ins.funcs[func as usize];
            for rel in &fuc.relocs {
                let offset = match rel.target.expand() {
                    ty::Kind::Func(fun) => self.ins.funcs[fun as usize].offset,
                    ty::Kind::Global(glo) => self.ins.globals[glo as usize].offset,
                    _ => unreachable!(),
                };
                rel.reloc.apply_jump(to, offset, fuc.offset);
            }
        }

        AbleOsExecutableHeader {
            magic_number: [0x15, 0x91, 0xD2],
            executable_version: 0,
            code_length: code_length.saturating_sub(HEADER_SIZE) as _,
            data_length: data_length as _,
            debug_length: 0,
            config_length: 0,
            metadata_length: 0,
        }
    }

    pub fn disasm<'a>(
        &'a self,
        mut sluce: &[u8],
        files: &'a [parser::Ast],
        output: &mut String,
        eca_handler: impl FnMut(&mut &[u8]),
    ) -> Result<(), hbbytecode::DisasmError<'a>> {
        use hbbytecode::DisasmItem;
        let functions = self
            .ins
            .funcs
            .iter()
            .filter(|f| task::is_done(f.offset))
            .map(|f| {
                let name = if f.file != u32::MAX {
                    let file = &files[f.file as usize];
                    file.ident_str(f.name)
                } else {
                    "target_fn"
                };
                (f.offset, (name, f.code.len() as u32, DisasmItem::Func))
            })
            .chain(self.ins.globals.iter().filter(|g| task::is_done(g.offset)).map(|g| {
                let name = if g.file == u32::MAX {
                    core::str::from_utf8(&g.data).unwrap_or("invalid utf-8")
                } else {
                    let file = &files[g.file as usize];
                    file.ident_str(g.name)
                };
                (g.offset, (name, g.data.len() as Size, DisasmItem::Global))
            }))
            .collect::<BTreeMap<_, _>>();
        hbbytecode::disasm(&mut sluce, &functions, output, eca_handler)
    }
}

impl ItemCtx {
    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        emit(&mut self.code, instr);
    }

    fn emit_body_code(
        &mut self,
        sig: Sig,
        tys: &Types,
        files: &[parser::Ast],
        ralloc: &mut Regalloc,
    ) -> usize {
        let mut nodes = mem::take(&mut self.nodes);

        let fuc = Function::new(&mut nodes, tys, sig);
        log::info!("{:?}", fuc);
        if self.call_count != 0 {
            mem::swap(
                &mut ralloc.env.preferred_regs_by_class,
                &mut ralloc.env.non_preferred_regs_by_class,
            );
        };

        let options = regalloc2::RegallocOptions {
            verbose_log: false,
            validate_ssa: cfg!(debug_assertions),
            algorithm: regalloc2::Algorithm::Ion,
        };
        regalloc2::run_with_ctx(&fuc, &ralloc.env, &options, &mut ralloc.ctx).unwrap_or_else(
            |err| {
                if let regalloc2::RegAllocError::SSA(vreg, inst) = err {
                    fuc.nodes[vreg.vreg() as Nid].lock_rc = Nid::MAX;
                    fuc.nodes[fuc.instrs[inst.index()].nid].lock_rc = Nid::MAX - 1;
                }
                fuc.nodes.graphviz_in_browser(tys, files);
                panic!("{err}")
            },
        );

        if self.call_count != 0 {
            mem::swap(
                &mut ralloc.env.preferred_regs_by_class,
                &mut ralloc.env.non_preferred_regs_by_class,
            );
        };

        let mut saved_regs = HashMap::<u8, u8>::default();
        let mut atr = |allc: regalloc2::Allocation| {
            debug_assert!(allc.is_reg());
            let hvenc = regalloc2::PReg::from_index(allc.index()).hw_enc() as u8;
            if hvenc <= 12 {
                return hvenc;
            }
            let would_insert = saved_regs.len() as u8 + reg::RET_ADDR + 1;
            *saved_regs.entry(hvenc).or_insert(would_insert)
        };

        let (retl, mut parama) = tys.parama(sig.ret);
        let mut typs = sig.args.args();
        let mut args = fuc.nodes[VOID].outputs[ARG_START..].iter();
        while let Some(aty) = typs.next(tys) {
            let Arg::Value(ty) = aty else { continue };
            let Some(loc) = parama.next(ty, tys) else { continue };
            let &arg = args.next().unwrap();
            let (rg, size) = match loc {
                PLoc::WideReg(rg, size) => (rg, size),
                PLoc::Reg(rg, size) if ty.loc(tys) == Loc::Stack => (rg, size),
                PLoc::Reg(..) | PLoc::Ref(..) => continue,
            };
            self.emit(instrs::st(rg, reg::STACK_PTR, fuc.nodes[arg].offset as _, size));
            if fuc.nodes[arg].lock_rc == 0 {
                self.emit(instrs::addi64(rg, reg::STACK_PTR, fuc.nodes[arg].offset as _));
            }
        }

        for (i, block) in fuc.blocks.iter().enumerate() {
            let blk = regalloc2::Block(i as _);
            fuc.nodes[block.nid].offset = self.code.len() as _;
            for instr_or_edit in ralloc.ctx.output.block_insts_and_edits(&fuc, blk) {
                let inst = match instr_or_edit {
                    regalloc2::InstOrEdit::Inst(inst) => inst,
                    regalloc2::InstOrEdit::Edit(&regalloc2::Edit::Move { from, to }) => {
                        self.emit(instrs::cp(atr(to), atr(from)));
                        continue;
                    }
                };

                let nid = fuc.instrs[inst.index()].nid;
                if nid == NEVER {
                    continue;
                };
                let allocs = ralloc.ctx.output.inst_allocs(inst);
                let node = &fuc.nodes[nid];

                let mut extend = |base: ty::Id, dest: ty::Id, from: usize, to: usize| {
                    if base.simple_size() == dest.simple_size() {
                        return Default::default();
                    }
                    match (base.is_signed(), dest.is_signed()) {
                        (true, true) => {
                            let op = [instrs::sxt8, instrs::sxt16, instrs::sxt32]
                                [base.simple_size().unwrap().ilog2() as usize];
                            op(atr(allocs[to]), atr(allocs[from]))
                        }
                        _ => {
                            let mask = (1u64 << (base.simple_size().unwrap() * 8)) - 1;
                            instrs::andi(atr(allocs[to]), atr(allocs[from]), mask)
                        }
                    }
                };

                match node.kind {
                    Kind::If => {
                        let &[_, cnd] = node.inputs.as_slice() else { unreachable!() };
                        if let Kind::BinOp { op } = fuc.nodes[cnd].kind
                            && let Some((op, swapped)) =
                                op.cond_op(fuc.nodes[fuc.nodes[cnd].inputs[1]].ty)
                        {
                            let &[lhs, rhs] = allocs else { unreachable!() };
                            let &[_, lh, rh] = fuc.nodes[cnd].inputs.as_slice() else {
                                unreachable!()
                            };

                            self.emit(extend(fuc.nodes[lh].ty, fuc.nodes[lh].ty.extend(), 0, 0));
                            self.emit(extend(fuc.nodes[rh].ty, fuc.nodes[rh].ty.extend(), 1, 1));

                            let rel = Reloc::new(self.code.len(), 3, 2);
                            self.jump_relocs.push((node.outputs[!swapped as usize], rel));
                            self.emit(op(atr(lhs), atr(rhs), 0));
                        } else {
                            self.emit(extend(fuc.nodes[cnd].ty, fuc.nodes[cnd].ty.extend(), 0, 0));
                            let rel = Reloc::new(self.code.len(), 3, 2);
                            self.jump_relocs.push((node.outputs[0], rel));
                            self.emit(instrs::jne(atr(allocs[0]), reg::ZERO, 0));
                        }
                    }
                    Kind::Loop | Kind::Region => {
                        if node.ralloc_backref as usize != i + 1 {
                            let rel = Reloc::new(self.code.len(), 1, 4);
                            self.jump_relocs.push((nid, rel));
                            self.emit(instrs::jmp(0));
                        }
                    }
                    Kind::Return => {
                        match retl {
                            Some(PLoc::Reg(r, size)) if sig.ret.loc(tys) == Loc::Stack => {
                                self.emit(instrs::ld(r, atr(allocs[0]), 0, size))
                            }
                            None | Some(PLoc::Reg(..)) => {}
                            Some(PLoc::WideReg(r, size)) => {
                                self.emit(instrs::ld(r, atr(allocs[0]), 0, size))
                            }
                            Some(PLoc::Ref(_, size)) => {
                                let [src, dst] = [atr(allocs[0]), atr(allocs[1])];
                                if let Ok(size) = u16::try_from(size) {
                                    self.emit(instrs::bmc(src, dst, size));
                                } else {
                                    for _ in 0..size / u16::MAX as u32 {
                                        self.emit(instrs::bmc(src, dst, u16::MAX));
                                        self.emit(instrs::addi64(src, src, u16::MAX as _));
                                        self.emit(instrs::addi64(dst, dst, u16::MAX as _));
                                    }
                                    self.emit(instrs::bmc(src, dst, size as u16));
                                    self.emit(instrs::addi64(src, src, size.wrapping_neg() as _));
                                    self.emit(instrs::addi64(dst, dst, size.wrapping_neg() as _));
                                }
                            }
                        }

                        if i != fuc.blocks.len() - 1 {
                            let rel = Reloc::new(self.code.len(), 1, 4);
                            self.ret_relocs.push(rel);
                            self.emit(instrs::jmp(0));
                        }
                    }
                    Kind::CInt { value } if node.ty.is_float() => {
                        self.emit(match node.ty {
                            ty::Id::F32 => instrs::li32(
                                atr(allocs[0]),
                                (f64::from_bits(value as _) as f32).to_bits(),
                            ),
                            ty::Id::F64 => instrs::li64(atr(allocs[0]), value as _),
                            _ => unreachable!(),
                        });
                    }
                    Kind::CInt { value } => self.emit(match tys.size_of(node.ty) {
                        1 => instrs::li8(atr(allocs[0]), value as _),
                        2 => instrs::li16(atr(allocs[0]), value as _),
                        4 => instrs::li32(atr(allocs[0]), value as _),
                        _ => instrs::li64(atr(allocs[0]), value as _),
                    }),
                    Kind::UnOp { op } => {
                        let op = op
                            .unop(node.ty, fuc.nodes[node.inputs[1]].ty)
                            .expect("TODO: unary operator not supported");
                        let &[dst, oper] = allocs else { unreachable!() };
                        self.emit(op(atr(dst), atr(oper)));
                    }
                    Kind::BinOp { .. } if node.lock_rc != 0 => {}
                    Kind::BinOp { op } => {
                        let &[.., lh, rh] = node.inputs.as_slice() else { unreachable!() };

                        if let Kind::CInt { value } = fuc.nodes[rh].kind
                            && fuc.nodes[rh].lock_rc != 0
                            && let Some(op) = op.imm_binop(node.ty)
                        {
                            let &[dst, lhs] = allocs else { unreachable!() };
                            self.emit(op(atr(dst), atr(lhs), value as _));
                        } else if let Some(op) =
                            op.binop(node.ty).or(op.float_cmp(fuc.nodes[lh].ty))
                        {
                            let &[dst, lhs, rhs] = allocs else { unreachable!() };
                            self.emit(op(atr(dst), atr(lhs), atr(rhs)));
                        } else if let Some(against) = op.cmp_against() {
                            let op_ty = fuc.nodes[lh].ty;

                            self.emit(extend(fuc.nodes[lh].ty, fuc.nodes[lh].ty.extend(), 0, 0));
                            self.emit(extend(fuc.nodes[rh].ty, fuc.nodes[rh].ty.extend(), 1, 1));
                            let &[dst, lhs, rhs] = allocs else { unreachable!() };

                            if op_ty.is_float() && matches!(op, TokenKind::Le | TokenKind::Ge) {
                                let opop = match op {
                                    TokenKind::Le => TokenKind::Gt,
                                    TokenKind::Ge => TokenKind::Lt,
                                    _ => unreachable!(),
                                };
                                let op_fn = opop.float_cmp(op_ty).unwrap();
                                self.emit(op_fn(atr(dst), atr(lhs), atr(rhs)));
                                self.emit(instrs::not(atr(dst), atr(dst)));
                            } else if op_ty.is_integer() {
                                let op_fn =
                                    if op_ty.is_signed() { instrs::cmps } else { instrs::cmpu };
                                self.emit(op_fn(atr(dst), atr(lhs), atr(rhs)));
                                self.emit(instrs::cmpui(atr(dst), atr(dst), against));
                                if matches!(op, TokenKind::Eq | TokenKind::Lt | TokenKind::Gt) {
                                    self.emit(instrs::not(atr(dst), atr(dst)));
                                }
                            } else {
                                todo!("unhandled operator: {op}");
                            }
                        } else {
                            todo!("unhandled operator: {op}");
                        }
                    }
                    Kind::Call { args, func } => {
                        let (ret, mut parama) = tys.parama(node.ty);
                        let has_ret = ret.is_some() as usize;
                        let mut args = args.args();
                        let mut allocs = allocs[has_ret..].iter();
                        while let Some(arg) = args.next(tys) {
                            let Arg::Value(ty) = arg else { continue };
                            let Some(loc) = parama.next(ty, tys) else { continue };

                            let &arg = allocs.next().unwrap();
                            let (rg, size) = match loc {
                                PLoc::Reg(rg, size) if ty.loc(tys) == Loc::Stack => (rg, size),
                                PLoc::WideReg(rg, size) => (rg, size),
                                PLoc::Ref(..) | PLoc::Reg(..) => continue,
                            };
                            self.emit(instrs::ld(rg, atr(arg), 0, size));
                        }

                        debug_assert!(
                            !matches!(ret, Some(PLoc::Ref(..))) || allocs.next().is_some()
                        );

                        if func == ty::ECA {
                            self.emit(instrs::eca());
                        } else {
                            self.relocs.push(TypedReloc {
                                target: ty::Kind::Func(func).compress(),
                                reloc: Reloc::new(self.code.len(), 3, 4),
                            });
                            self.emit(instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
                        }

                        if let Some(PLoc::WideReg(r, size)) = ret {
                            debug_assert_eq!(
                                fuc.nodes[*node.inputs.last().unwrap()].kind,
                                Kind::Stck
                            );
                            let stck = fuc.nodes[*node.inputs.last().unwrap()].offset;
                            self.emit(instrs::st(r, reg::STACK_PTR, stck as _, size));
                        }
                        if let Some(PLoc::Reg(r, size)) = ret
                            && node.ty.loc(tys) == Loc::Stack
                        {
                            debug_assert_eq!(
                                fuc.nodes[*node.inputs.last().unwrap()].kind,
                                Kind::Stck
                            );
                            let stck = fuc.nodes[*node.inputs.last().unwrap()].offset;
                            self.emit(instrs::st(r, reg::STACK_PTR, stck as _, size));
                        }
                    }
                    Kind::Global { global } => {
                        let reloc = Reloc::new(self.code.len(), 3, 4);
                        self.relocs.push(TypedReloc {
                            target: ty::Kind::Global(global).compress(),
                            reloc,
                        });
                        self.emit(instrs::lra(atr(allocs[0]), 0, 0));
                    }
                    Kind::Stck => {
                        let base = reg::STACK_PTR;
                        let offset = fuc.nodes[nid].offset;
                        self.emit(instrs::addi64(atr(allocs[0]), base, offset as _));
                    }
                    Kind::Load => {
                        let mut region = node.inputs[1];
                        let mut offset = 0;
                        if fuc.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                            && let Kind::CInt { value } =
                                fuc.nodes[fuc.nodes[region].inputs[2]].kind
                        {
                            region = fuc.nodes[region].inputs[1];
                            offset = value as Offset;
                        }
                        let size = tys.size_of(node.ty);
                        if node.ty.loc(tys) != Loc::Stack {
                            let (base, offset) = match fuc.nodes[region].kind {
                                Kind::Stck => (reg::STACK_PTR, fuc.nodes[region].offset + offset),
                                _ => (atr(allocs[1]), offset),
                            };
                            self.emit(instrs::ld(atr(allocs[0]), base, offset as _, size as _));
                        }
                    }
                    Kind::Stre if node.inputs[1] == VOID => {}
                    Kind::Stre => {
                        let mut region = node.inputs[2];
                        let mut offset = 0;
                        let size = u16::try_from(tys.size_of(node.ty)).expect("TODO");
                        if fuc.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                            && let Kind::CInt { value } =
                                fuc.nodes[fuc.nodes[region].inputs[2]].kind
                            && node.ty.loc(tys) == Loc::Reg
                        {
                            region = fuc.nodes[region].inputs[1];
                            offset = value as Offset;
                        }
                        let nd = &fuc.nodes[region];
                        let (base, offset, src) = match nd.kind {
                            Kind::Stck if node.ty.loc(tys) == Loc::Reg => {
                                (reg::STACK_PTR, nd.offset + offset, allocs[0])
                            }
                            _ => (atr(allocs[0]), offset, allocs[1]),
                        };

                        match node.ty.loc(tys) {
                            Loc::Reg => self.emit(instrs::st(atr(src), base, offset as _, size)),
                            Loc::Stack => {
                                debug_assert_eq!(offset, 0);
                                self.emit(instrs::bmc(atr(src), base, size))
                            }
                        }
                    }
                    Kind::Start
                    | Kind::Entry
                    | Kind::Mem
                    | Kind::End
                    | Kind::Loops
                    | Kind::Then
                    | Kind::Else
                    | Kind::Phi
                    | Kind::Arg => unreachable!(),
                }
            }
        }

        self.nodes = nodes;

        saved_regs.len()
    }

    pub fn emit_ct_body(
        &mut self,
        tys: &mut Types,
        files: &[parser::Ast],
        sig: Sig,
        pool: &mut Pool,
    ) {
        self.emit_body(tys, files, sig, pool);
        self.code.truncate(self.code.len() - instrs::jala(0, 0, 0).0);
        self.emit(instrs::tx());
    }

    pub fn emit_body(&mut self, tys: &mut Types, files: &[parser::Ast], sig: Sig, pool: &mut Pool) {
        self.nodes.check_final_integrity(tys, files);
        self.nodes.graphviz(tys, files);
        self.nodes.gcm(&mut pool.nid_stack);
        self.nodes.basic_blocks();
        self.nodes.graphviz(tys, files);

        debug_assert!(self.code.is_empty());
        let tail = mem::take(&mut self.call_count) == 0;

        '_open_function: {
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
            self.emit(instrs::st(reg::RET_ADDR + tail as u8, reg::STACK_PTR, 0, 0));
        }

        let mut stack_size = 0;
        '_compute_stack: {
            let mems = mem::take(&mut self.nodes[MEM].outputs);
            for &stck in mems.iter() {
                if !matches!(self.nodes[stck].kind, Kind::Stck | Kind::Arg) {
                    debug_assert_matches!(
                        self.nodes[stck].kind,
                        Kind::Phi | Kind::Return | Kind::Load | Kind::Call { .. } | Kind::Stre
                    );
                    continue;
                }
                stack_size += tys.size_of(self.nodes[stck].ty);
                self.nodes[stck].offset = stack_size;
            }
            for &stck in mems.iter() {
                if !matches!(self.nodes[stck].kind, Kind::Stck | Kind::Arg) {
                    continue;
                }
                self.nodes[stck].offset = stack_size - self.nodes[stck].offset;
            }
            self.nodes[MEM].outputs = mems;
        }

        let saved = self.emit_body_code(sig, tys, files, &mut pool.ralloc);

        if let Some(last_ret) = self.ret_relocs.last()
            && last_ret.offset as usize == self.code.len() - 5
            && self
                .jump_relocs
                .last()
                .map_or(true, |&(r, _)| self.nodes[r].offset as usize != self.code.len())
        {
            self.code.truncate(self.code.len() - 5);
            self.ret_relocs.pop();
        }

        // FIXME: maybe do this incrementally
        for (nd, rel) in self.jump_relocs.drain(..) {
            let offset = self.nodes[nd].offset;
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

            match (pushed, stack) {
                (0, 0) => {
                    stripped_prelude_size = instrs::addi64(0, 0, 0).0 + instrs::st(0, 0, 0, 0).0;
                    self.code.drain(0..stripped_prelude_size);
                    break '_close_function;
                }
                (0, stack) => {
                    write_reloc(&mut self.code, 3, -stack, 8);
                    stripped_prelude_size = instrs::st(0, 0, 0, 0).0;
                    let end = instrs::addi64(0, 0, 0).0 + instrs::st(0, 0, 0, 0).0;
                    self.code.drain(instrs::addi64(0, 0, 0).0..end);
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
        self.emit(instrs::jala(reg::ZERO, reg::RET_ADDR, 0));
    }
}

#[derive(Debug)]
struct Block {
    nid: Nid,
    preds: Vec<regalloc2::Block>,
    succs: Vec<regalloc2::Block>,
    instrs: regalloc2::InstRange,
    params: Vec<regalloc2::VReg>,
    branch_blockparams: Vec<regalloc2::VReg>,
}

#[derive(Debug)]
struct Instr {
    nid: Nid,
    ops: Vec<regalloc2::Operand>,
}

pub struct Function<'a> {
    sig: Sig,
    nodes: &'a mut Nodes,
    tys: &'a Types,
    blocks: Vec<Block>,
    instrs: Vec<Instr>,
}

impl core::fmt::Debug for Function<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, block) in self.blocks.iter().enumerate() {
            writeln!(f, "sb{i}{:?}-{:?}:", block.params, block.preds)?;

            for inst in block.instrs.iter() {
                let instr = &self.instrs[inst.index()];
                writeln!(f, "{}: i{:?}:{:?}", inst.index(), self.nodes[instr.nid].kind, instr.ops)?;
            }

            writeln!(f, "eb{i}{:?}-{:?}:", block.branch_blockparams, block.succs)?;
        }
        Ok(())
    }
}

impl<'a> Function<'a> {
    fn new(nodes: &'a mut Nodes, tys: &'a Types, sig: Sig) -> Self {
        let mut s =
            Self { nodes, tys, sig, blocks: Default::default(), instrs: Default::default() };
        s.nodes.visited.clear(s.nodes.values.len());
        s.emit_node(VOID, VOID);
        s.add_block(0);
        s.blocks.pop();
        s
    }

    fn add_block(&mut self, nid: Nid) -> RallocBRef {
        if let Some(prev) = self.blocks.last_mut() {
            prev.instrs = regalloc2::InstRange::new(
                prev.instrs.first(),
                regalloc2::Inst::new(self.instrs.len()),
            );
        }

        self.blocks.push(Block {
            nid,
            preds: Default::default(),
            succs: Default::default(),
            instrs: regalloc2::InstRange::new(
                regalloc2::Inst::new(self.instrs.len()),
                regalloc2::Inst::new(self.instrs.len() + 1),
            ),
            params: Default::default(),
            branch_blockparams: Default::default(),
        });
        self.blocks.len() as RallocBRef - 1
    }

    fn add_instr(&mut self, nid: Nid, ops: Vec<regalloc2::Operand>) {
        self.instrs.push(Instr { nid, ops });
    }

    fn urg(&mut self, nid: Nid) -> regalloc2::Operand {
        regalloc2::Operand::reg_use(self.rg(nid))
    }

    fn drg(&mut self, nid: Nid) -> regalloc2::Operand {
        regalloc2::Operand::reg_def(self.rg(nid))
    }

    fn rg(&self, nid: Nid) -> regalloc2::VReg {
        debug_assert!(
            !self.nodes.is_cfg(nid) || matches!(self.nodes[nid].kind, Kind::Call { .. }),
            "{:?}",
            self.nodes[nid]
        );
        debug_assert_eq!(self.nodes[nid].lock_rc, 0, "{nid} {:?}", self.nodes[nid]);
        debug_assert!(self.nodes[nid].kind != Kind::Phi || self.nodes[nid].ty != ty::Id::VOID);
        regalloc2::VReg::new(nid as _, regalloc2::RegClass::Int)
    }

    fn emit_node(&mut self, nid: Nid, prev: Nid) {
        if matches!(self.nodes[nid].kind, Kind::Region | Kind::Loop) {
            let prev_bref = self.nodes[prev].ralloc_backref;
            let node = self.nodes[nid].clone();

            let idx = 1 + node.inputs.iter().position(|&i| i == prev).unwrap();

            for ph in node.outputs {
                if self.nodes[ph].kind != Kind::Phi || self.nodes[ph].ty == ty::Id::VOID {
                    continue;
                }

                let rg = self.rg(self.nodes[ph].inputs[idx]);
                self.blocks[prev_bref as usize].branch_blockparams.push(rg);
            }

            self.add_instr(nid, vec![]);

            match (self.nodes[nid].kind, self.nodes.visited.set(nid)) {
                (Kind::Loop, false) => {
                    for i in node.inputs {
                        self.bridge(i, nid);
                    }
                    return;
                }
                (Kind::Region, true) => return,
                _ => {}
            }
        } else if !self.nodes.visited.set(nid) {
            return;
        }

        let mut node = self.nodes[nid].clone();
        match node.kind {
            Kind::Start => {
                debug_assert_matches!(self.nodes[node.outputs[0]].kind, Kind::Entry);
                self.emit_node(node.outputs[0], VOID)
            }
            Kind::If => {
                self.nodes[nid].ralloc_backref = self.nodes[prev].ralloc_backref;

                let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Kind::BinOp { op } = self.nodes[cond].kind
                    && let Some((_, swapped)) = op.cond_op(node.ty)
                {
                    if swapped {
                        mem::swap(&mut then, &mut else_);
                    }
                    let &[_, lhs, rhs] = self.nodes[cond].inputs.as_slice() else { unreachable!() };
                    let ops = vec![self.urg(lhs), self.urg(rhs)];
                    self.add_instr(nid, ops);
                } else {
                    mem::swap(&mut then, &mut else_);
                    let ops = vec![self.urg(cond)];
                    self.add_instr(nid, ops);
                }

                self.emit_node(then, nid);
                self.emit_node(else_, nid);
            }
            Kind::Region | Kind::Loop => {
                self.nodes[nid].ralloc_backref = self.add_block(nid);
                if node.kind == Kind::Region {
                    for i in node.inputs {
                        self.bridge(i, nid);
                    }
                }
                let mut block = vec![];
                for ph in node.outputs.clone() {
                    if self.nodes[ph].kind != Kind::Phi || self.nodes[ph].ty == ty::Id::VOID {
                        continue;
                    }
                    block.push(self.rg(ph));
                }
                self.blocks[self.nodes[nid].ralloc_backref as usize].params = block;
                self.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::Return => {
                let ops = match self.tys.parama(self.sig.ret).0 {
                    None => vec![],
                    Some(PLoc::Reg(..)) if self.sig.ret.loc(self.tys) == Loc::Stack => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1])]
                    }
                    Some(PLoc::Reg(r, ..)) => {
                        vec![regalloc2::Operand::reg_fixed_use(
                            self.rg(node.inputs[1]),
                            regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                        )]
                    }
                    Some(PLoc::WideReg(..))  => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1])]
                    }
                    Some(PLoc::Ref(..)) => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1]), self.urg(MEM)]
                    }
                };

                self.add_instr(nid, ops);
                self.emit_node(node.outputs[0], nid);
            }
            Kind::CInt { .. }
                if node.outputs.iter().all(|&o| {
                    let ond = &self.nodes[o];
                    matches!(ond.kind, Kind::BinOp { op }
                        if op.imm_binop(ond.ty).is_some()
                            && self.nodes.is_const(ond.inputs[2])
                            && op.cond_op(ond.ty).is_none())
                }) =>
            {
                self.nodes.lock(nid)
            }
            Kind::CInt { .. } => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Entry => {
                self.nodes[nid].ralloc_backref = self.add_block(nid);

                let (ret, mut parama) = self.tys.parama(self.sig.ret);
                let mut typs = self.sig.args.args();
                #[expect(clippy::unnecessary_to_owned)]
                let mut args = self.nodes[VOID].outputs[ARG_START..].to_owned().into_iter();
                while let Some(ty) = typs.next_value(self.tys) {
                    let arg = args.next().unwrap();
                    debug_assert_eq!(self.nodes[arg].kind, Kind::Arg);
                    match parama.next(ty, self.tys) {
                        None => {}
                        Some(PLoc::Reg(r, _) | PLoc::WideReg(r, _) | PLoc::Ref(r, _)) => {
                            self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                                self.rg(arg),
                                regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                            )]);
                        }
                    }
                }

                if let Some(PLoc::Ref(r, ..)) = ret {
                    self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                        self.rg(MEM),
                        regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                    )]);
                }

                self.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::Then | Kind::Else => {
                self.nodes[nid].ralloc_backref = self.add_block(nid);
                self.bridge(prev, nid);
                self.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::BinOp { op: TokenKind::Add } if self.nodes[node.inputs[1]].lock_rc != 0 => self.nodes.lock(nid),
            Kind::BinOp { op: TokenKind::Add }
                if self.nodes.is_const(node.inputs[2])
                    && node.outputs.iter().all(|&n| {
                        (matches!(self.nodes[n].kind, Kind::Stre if self.nodes[n].inputs[2] == nid)
                            || matches!(self.nodes[n].kind, Kind::Load if self.nodes[n].inputs[1] == nid))
                            && self.nodes[n].ty.loc(self.tys) == Loc::Reg
                    }) =>
            {
                self.nodes.lock(nid)
            }
            Kind::BinOp { op }
                if op.cond_op(node.ty).is_some()
                    && node.outputs.iter().all(|&n| self.nodes[n].kind == Kind::If) =>
            {
                self.nodes.lock(nid)
            }
            Kind::BinOp { .. } => {
                let &[_, lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                let ops = if let Kind::CInt { .. } = self.nodes[rhs].kind
                    && self.nodes[rhs].lock_rc != 0
                {
                    vec![self.drg(nid), self.urg(lhs)]
                } else {
                    vec![self.drg(nid), self.urg(lhs), self.urg(rhs)]
                };
                self.add_instr(nid, ops);
            }
            Kind::UnOp { .. } => {
                let ops = vec![self.drg(nid), self.urg(node.inputs[1])];
                self.add_instr(nid, ops);
            }
            Kind::Call { args, .. } => {
                self.nodes[nid].ralloc_backref = self.nodes[prev].ralloc_backref;
                let mut ops = vec![];

                let (ret, mut parama) = self.tys.parama(node.ty);
                if ret.is_some() {
                    ops.push(regalloc2::Operand::reg_fixed_def(
                        self.rg(nid),
                        regalloc2::PReg::new(1, regalloc2::RegClass::Int),
                    ));
                }

                let mut tys = args.args();
                let mut args = node.inputs[1..].iter();
                while let Some(ty) = tys.next_value(self.tys) {
                    let mut i = *args.next().unwrap();
                    let Some(loc) = parama.next(ty, self.tys) else { continue };

                    match loc {
                        PLoc::Reg(r, _) if ty.loc(self.tys) == Loc::Reg => {
                            ops.push(regalloc2::Operand::reg_fixed_use(
                                self.rg(i),
                                regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                            ));
                        }
                        PLoc::WideReg(..) | PLoc::Reg(..) => {
                            loop {
                                match self.nodes[i].kind {
                                    Kind::Stre { .. } => i = self.nodes[i].inputs[2],
                                    Kind::Load { .. } => i = self.nodes[i].inputs[1],
                                    _ => break,
                                }
                                debug_assert_ne!(i, 0);
                            }
                            debug_assert!(i != 0);
                            ops.push(self.urg(i));
                        }
                        PLoc::Ref(r, _) => {
                            loop {
                                match self.nodes[i].kind {
                                    Kind::Stre { .. } => i = self.nodes[i].inputs[2],
                                    Kind::Load { .. } => i = self.nodes[i].inputs[1],
                                    _ => break,
                                }
                                debug_assert_ne!(i, 0);
                            }
                            debug_assert!(i != 0);
                            ops.push(regalloc2::Operand::reg_fixed_use(
                                self.rg(i),
                                regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                            ));
                        }
                    }
                }

                if let Some(PLoc::Ref(r, _)) = ret {
                    ops.push(regalloc2::Operand::reg_fixed_use(
                        self.rg(*node.inputs.last().unwrap()),
                        regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                    ));
                }

                self.add_instr(nid, ops);

                self.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    if self.nodes[o].inputs[0] == nid
                        || (matches!(self.nodes[o].kind, Kind::Loop | Kind::Region)
                            && self.nodes[o].inputs[1] == nid)
                    {
                        self.emit_node(o, nid);
                    }
                }
            }
            Kind::Global { .. } => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Stck | Kind::Arg
                if node.outputs.iter().all(|&n| {
                    matches!(self.nodes[n].kind,  Kind::Load
                        if self.nodes[n].ty.loc(self.tys) == Loc::Reg)
                    || matches!(self.nodes[n].kind, Kind::Stre
                        if self.nodes[n].ty.loc(self.tys) == Loc::Reg
                        && self.nodes[n].inputs[1] != nid)
                    || matches!(self.nodes[n].kind, Kind::BinOp { op: TokenKind::Add }
                        if self.nodes.is_const(self.nodes[n].inputs[2])
                            && self.nodes[n]
                                .outputs
                                .iter()
                                .all(|&n| matches!(self.nodes[n].kind, Kind::Load
                                    if self.nodes[n].ty.loc(self.tys) == Loc::Reg)
                                || matches!(self.nodes[n].kind, Kind::Stre
                                    if self.nodes[n].ty.loc(self.tys) == Loc::Reg
                                && self.nodes[n].inputs[1] != nid)))
                }) => self.nodes.lock(nid),
            Kind::Stck if self.tys.size_of(node.ty) == 0 => self.nodes.lock(nid),
            Kind::Stck => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::End |
            Kind::Phi | Kind::Arg | Kind::Mem | Kind::Loops  => {}
            Kind::Load { .. } if node.ty.loc(self.tys) == Loc::Stack => {
                self.nodes.lock(nid)
            }
            Kind::Load { .. } => {
                let mut region = node.inputs[1];
                if self.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                    && self.nodes.is_const(self.nodes[region].inputs[2])
                    && node.ty.loc(self.tys) == Loc::Reg
                {
                    region = self.nodes[region].inputs[1]
                }
                let ops = match self.nodes[region].kind {
                    Kind::Stck => vec![self.drg(nid)],
                    _ => vec![self.drg(nid), self.urg(region)],
                };
                self.add_instr(nid, ops);
            }
            Kind::Stre if node.inputs[1] == VOID => self.nodes.lock(nid),
            Kind::Stre => {
                let mut region = node.inputs[2];
                if self.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                    && self.nodes.is_const(self.nodes[region].inputs[2])
                    && node.ty.loc(self.tys) == Loc::Reg
                {
                    region = self.nodes[region].inputs[1]
                }
                let ops = match self.nodes[region].kind {
                    _ if node.ty.loc(self.tys) == Loc::Stack => {
                        if self.nodes[node.inputs[1]].kind == Kind::Arg {
                            vec![self.urg(region), self.urg(node.inputs[1])]
                        } else {
                            vec![self.urg(region), self.urg(self.nodes[node.inputs[1]].inputs[1])]
                        }
                    }
                    Kind::Stck => vec![self.urg(node.inputs[1])],
                    _ => vec![self.urg(region), self.urg(node.inputs[1])],
                };
                self.add_instr(nid, ops);
            }
        }
    }

    fn bridge(&mut self, pred: u16, succ: u16) {
        if self.nodes[pred].ralloc_backref == u16::MAX
            || self.nodes[succ].ralloc_backref == u16::MAX
        {
            return;
        }
        self.blocks[self.nodes[pred].ralloc_backref as usize]
            .succs
            .push(regalloc2::Block::new(self.nodes[succ].ralloc_backref as usize));
        self.blocks[self.nodes[succ].ralloc_backref as usize]
            .preds
            .push(regalloc2::Block::new(self.nodes[pred].ralloc_backref as usize));
    }

    fn reschedule_block(&mut self, from: Nid, outputs: &mut Vc) {
        let from = Some(&from);
        let mut buf = Vec::with_capacity(outputs.len());
        let mut seen = BitSet::default();
        seen.clear(self.nodes.values.len());

        for &o in outputs.iter() {
            if !self.nodes.is_cfg(o) {
                continue;
            }

            seen.set(o);

            let mut cursor = buf.len();
            buf.push(o);
            while let Some(&n) = buf.get(cursor) {
                for &i in &self.nodes[n].inputs[1..] {
                    if from == self.nodes[i].inputs.first()
                        && self.nodes[i]
                            .outputs
                            .iter()
                            .all(|&o| self.nodes[o].inputs.first() != from || seen.get(o))
                        && seen.set(i)
                    {
                        buf.push(i);
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
            buf.push(o);
            while let Some(&n) = buf.get(cursor) {
                for &i in &self.nodes[n].inputs[1..] {
                    if from == self.nodes[i].inputs.first()
                        && self.nodes[i]
                            .outputs
                            .iter()
                            .all(|&o| self.nodes[o].inputs.first() != from || seen.get(o))
                        && seen.set(i)
                    {
                        buf.push(i);
                    }
                }
                cursor += 1;
            }
        }

        debug_assert!(
            outputs.len() == buf.len() || outputs.len() == buf.len() + 1,
            "{:?} {:?}",
            outputs,
            buf
        );

        if buf.len() + 1 == outputs.len() {
            outputs.remove(outputs.len() - 1);
        }
        outputs.copy_from_slice(&buf);
    }
}

impl regalloc2::Function for Function<'_> {
    fn num_insts(&self) -> usize {
        self.instrs.len()
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        regalloc2::Block(0)
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        self.blocks[block.index()].instrs
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        &self.blocks[block.index()].succs
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        &self.blocks[block.index()].preds
    }

    fn block_params(&self, block: regalloc2::Block) -> &[regalloc2::VReg] {
        &self.blocks[block.index()].params
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        self.nodes[self.instrs[insn.index()].nid].kind == Kind::Return
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        matches!(
            self.nodes[self.instrs[insn.index()].nid].kind,
            Kind::If | Kind::Then | Kind::Else | Kind::Entry | Kind::Loop | Kind::Region
        )
    }

    fn branch_blockparams(
        &self,
        block: regalloc2::Block,
        _insn: regalloc2::Inst,
        _succ_idx: usize,
    ) -> &[regalloc2::VReg] {
        debug_assert!(
            self.blocks[block.index()].succs.len() == 1
                || self.blocks[block.index()].branch_blockparams.is_empty()
        );

        &self.blocks[block.index()].branch_blockparams
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        &self.instrs[insn.index()].ops
    }

    fn inst_clobbers(&self, insn: regalloc2::Inst) -> regalloc2::PRegSet {
        let node = &self.nodes[self.instrs[insn.index()].nid];
        if matches!(node.kind, Kind::Call { .. }) {
            let mut set = regalloc2::PRegSet::default();
            let returns = self.tys.parama(node.ty).0.is_some();
            for i in 1 + returns as usize..13 {
                set.add(regalloc2::PReg::new(i, regalloc2::RegClass::Int));
            }
            set
        } else {
            regalloc2::PRegSet::default()
        }
    }

    fn num_vregs(&self) -> usize {
        self.nodes.values.len()
    }

    fn spillslot_size(&self, regclass: regalloc2::RegClass) -> usize {
        match regclass {
            regalloc2::RegClass::Int => 1,
            regalloc2::RegClass::Float => unreachable!(),
            regalloc2::RegClass::Vector => unreachable!(),
        }
    }
}

impl TokenKind {
    pub fn cmp_against(self) -> Option<u64> {
        Some(match self {
            TokenKind::Le | TokenKind::Gt => 1,
            TokenKind::Ne | TokenKind::Eq => 0,
            TokenKind::Ge | TokenKind::Lt => (-1i64) as _,
            _ => return None,
        })
    }

    pub fn float_cmp(self, ty: ty::Id) -> Option<fn(u8, u8, u8) -> EncodedInstr> {
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
            Self::Shr if signed => basic_op!(srui8, srui16, srui32, srui64),
            Self::Shr => basic_op!(srui8, srui16, srui32, srui64),
            Self::Shl => basic_op!(slui8, slui16, slui32, slui64),
            _ => return None,
        };

        let size = ty.simple_size().unwrap();
        Some(ops[size.ilog2() as usize])
    }

    pub fn unop(&self, dst: ty::Id, src: ty::Id) -> Option<fn(u8, u8) -> EncodedInstr> {
        let src_idx = src.simple_size().unwrap().ilog2() as usize - 2;
        Some(match self {
            Self::Sub => instrs::neg,
            Self::Float if dst.is_float() && src.is_integer() => {
                [instrs::itf32, instrs::itf64][src_idx]
            }
            Self::Number if src.is_float() && dst.is_integer() => {
                [|a, b| instrs::fti32(a, b, 1), |a, b| instrs::fti64(a, b, 1)][src_idx]
            }
            Self::Float if dst.is_float() && src.is_float() => {
                [instrs::fc32t64, |a, b| instrs::fc64t32(a, b, 1)][src_idx]
            }
            _ => return None,
        })
    }
}

type EncodedInstr = (usize, [u8; instrs::MAX_SIZE]);
fn emit(out: &mut Vec<u8>, (len, instr): EncodedInstr) {
    out.extend_from_slice(&instr[..len]);
}

pub fn binary_prelude(to: &mut Vec<u8>) {
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
}

impl Default for Comptime {
    fn default() -> Self {
        let mut stack = Box::<[u8; VM_STACK_SIZE]>::new_uninit();
        let mut vm = hbvm::Vm::default();
        let ptr = unsafe { stack.as_mut_ptr().cast::<u8>().add(VM_STACK_SIZE) as u64 };
        vm.write_reg(reg::STACK_PTR, ptr);
        Self { vm, stack: unsafe { stack.assume_init() }, code: Default::default() }
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
                unknown => unreachable!("unknown ecall: {unknown:?}"),
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
