use {
    super::{HbvmBackend, Nid, Nodes},
    crate::{
        lexer::TokenKind,
        parser,
        reg::{self, Reg},
        son::{debug_assert_matches, Kind, ARG_START, MEM, VOID},
        ty::{self, Arg, Loc},
        utils::BitSet,
        Offset, PLoc, Reloc, Sig, TypedReloc, Types,
    },
    alloc::{borrow::ToOwned, vec::Vec},
    core::{mem, ops::Range},
    hbbytecode::{self as instrs},
};

impl HbvmBackend {
    pub fn emit_body_code_my(
        &mut self,
        nodes: &mut Nodes,
        sig: Sig,
        tys: &Types,
        files: &[parser::Ast],
    ) -> (usize, bool) {
        let fuc = Function::new(nodes, tys, sig);
        log::info!("{fuc:?}");

        let strip_load = |value| match fuc.nodes[value].kind {
            Kind::Load { .. } if fuc.nodes[value].ty.loc(tys) == Loc::Stack => {
                fuc.nodes[value].inputs[1]
            }
            _ => value,
        };

        let mut res = mem::take(&mut self.ralloc_my);

        Env::new(&fuc, &fuc.func, &mut res).run();

        '_open_function: {
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
            self.emit(instrs::st(reg::RET_ADDR + fuc.tail as u8, reg::STACK_PTR, 0, 0));
        }

        res.node_to_reg[MEM as usize] = res.bundles.len() as u8 + 1;

        let reg_offset = if fuc.tail { reg::RET + 12 } else { reg::RET_ADDR + 1 };

        res.node_to_reg.iter_mut().filter(|r| **r != 0).for_each(|r| {
            if *r == u8::MAX {
                *r = 0
            } else {
                *r += reg_offset - 1;
                if fuc.tail && *r >= reg::RET_ADDR {
                    *r += 1;
                }
            }
        });

        let atr = |allc: Nid| {
            let allc = strip_load(allc);
            debug_assert_eq!(
                fuc.nodes[allc].lock_rc,
                0,
                "{:?} {}",
                fuc.nodes[allc],
                ty::Display::new(tys, files, fuc.nodes[allc].ty)
            );
            res.node_to_reg[allc as usize]
        };

        //for (id, node) in fuc.nodes.iter() {
        //    if node.kind == Kind::Phi {
        //        debug_assert_eq!(atr(node.inputs[1]), atr(node.inputs[2]));
        //        debug_assert_eq!(atr(id), atr(node.inputs[2]));
        //    }
        //}

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
                PLoc::Reg(r, ..) | PLoc::Ref(r, ..) => {
                    self.emit(instrs::cp(atr(arg), r));
                    continue;
                }
            };
            self.emit(instrs::st(rg, reg::STACK_PTR, self.offsets[arg as usize] as _, size));
            if fuc.nodes[arg].lock_rc == 0 {
                self.emit(instrs::addi64(rg, reg::STACK_PTR, self.offsets[arg as usize] as _));
            }
            self.emit(instrs::cp(atr(arg), rg));
        }

        for (i, block) in fuc.func.blocks.iter().enumerate() {
            self.offsets[block.entry as usize] = self.code.len() as _;
            for &nid in &fuc.func.instrs[block.range.clone()] {
                if nid == VOID {
                    continue;
                }

                let node = &fuc.nodes[nid];
                let extend = |base: ty::Id, dest: ty::Id, from: Nid, to: Nid| {
                    let (bsize, dsize) = (tys.size_of(base), tys.size_of(dest));
                    debug_assert!(bsize <= 8, "{}", ty::Display::new(tys, files, base));
                    debug_assert!(dsize <= 8, "{}", ty::Display::new(tys, files, dest));
                    if bsize == dsize {
                        return Default::default();
                    }
                    match (base.is_signed(), dest.is_signed()) {
                        (true, true) => {
                            let op = [instrs::sxt8, instrs::sxt16, instrs::sxt32]
                                [bsize.ilog2() as usize];
                            op(atr(to), atr(from))
                        }
                        _ => {
                            let mask = (1u64 << (bsize * 8)) - 1;
                            instrs::andi(atr(to), atr(from), mask)
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
                            let &[_, lhs, rhs] = fuc.nodes[cnd].inputs.as_slice() else {
                                unreachable!()
                            };

                            self.emit(extend(
                                fuc.nodes[lhs].ty,
                                fuc.nodes[lhs].ty.extend(),
                                lhs,
                                lhs,
                            ));
                            self.emit(extend(
                                fuc.nodes[rhs].ty,
                                fuc.nodes[rhs].ty.extend(),
                                rhs,
                                rhs,
                            ));

                            let rel = Reloc::new(self.code.len(), 3, 2);
                            self.jump_relocs.push((node.outputs[!swapped as usize], rel));
                            self.emit(op(atr(lhs), atr(rhs), 0));
                        } else {
                            self.emit(extend(
                                fuc.nodes[cnd].ty,
                                fuc.nodes[cnd].ty.extend(),
                                cnd,
                                cnd,
                            ));
                            let rel = Reloc::new(self.code.len(), 3, 2);
                            debug_assert_eq!(fuc.nodes[node.outputs[0]].kind, Kind::Then);
                            self.jump_relocs.push((node.outputs[0], rel));
                            self.emit(instrs::jne(atr(cnd), reg::ZERO, 0));
                        }
                    }
                    Kind::Loop | Kind::Region => {
                        let index = node
                            .inputs
                            .iter()
                            .position(|&n| block.entry == fuc.idom_of(n))
                            .unwrap()
                            + 1;

                        let mut moves = vec![];
                        for &out in node.outputs.iter() {
                            if fuc.nodes[out].is_data_phi() {
                                let src = fuc.nodes[out].inputs[index];
                                if atr(out) != atr(src) {
                                    moves.push([atr(out), atr(src), 0]);
                                }
                            }
                        }

                        debug_assert_eq!(moves.len(), {
                            moves.sort_unstable();
                            moves.dedup();
                            moves.len()
                        });

                        moves.sort_unstable_by(|[aa, ab, _], [ba, bb, _]| {
                            if aa == bb && ab == ba {
                                core::cmp::Ordering::Equal
                            } else if aa == bb {
                                core::cmp::Ordering::Greater
                            } else {
                                core::cmp::Ordering::Less
                            }
                        });

                        moves.dedup_by(|[aa, ab, _], [ba, bb, kind]| {
                            if aa == bb && ab == ba {
                                *kind = 1;
                                true
                            } else {
                                false
                            }
                        });

                        for [dst, src, kind] in moves {
                            if kind == 0 {
                                self.emit(instrs::cp(dst, src));
                            } else {
                                self.emit(instrs::swa(dst, src));
                            }
                        }

                        if fuc.block_of(nid) as usize != i + 1 {
                            let rel = Reloc::new(self.code.len(), 1, 4);
                            self.jump_relocs.push((nid, rel));
                            self.emit(instrs::jmp(0));
                        }
                    }
                    Kind::Return => {
                        let &[_, ret, ..] = node.inputs.as_slice() else { unreachable!() };
                        match retl {
                            None => {}
                            Some(PLoc::Reg(r, _)) if sig.ret.loc(tys) == Loc::Reg => {
                                self.emit(instrs::cp(r, atr(ret)));
                            }
                            Some(PLoc::Reg(r, size)) | Some(PLoc::WideReg(r, size)) => {
                                self.emit(instrs::ld(r, atr(ret), 0, size))
                            }
                            Some(PLoc::Ref(_, size)) => {
                                let [src, dst] = [atr(ret), atr(MEM)];
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

                        if i != fuc.func.blocks.len() - 1 {
                            let rel = Reloc::new(self.code.len(), 1, 4);
                            self.ret_relocs.push(rel);
                            self.emit(instrs::jmp(0));
                        }
                    }
                    Kind::Die => self.emit(instrs::un()),
                    Kind::CInt { value } if node.ty.is_float() => {
                        self.emit(match node.ty {
                            ty::Id::F32 => instrs::li32(
                                atr(nid),
                                (f64::from_bits(value as _) as f32).to_bits(),
                            ),
                            ty::Id::F64 => instrs::li64(atr(nid), value as _),
                            _ => unreachable!(),
                        });
                    }
                    Kind::CInt { value } => self.emit(match tys.size_of(node.ty) {
                        1 => instrs::li8(atr(nid), value as _),
                        2 => instrs::li16(atr(nid), value as _),
                        4 => instrs::li32(atr(nid), value as _),
                        _ => instrs::li64(atr(nid), value as _),
                    }),
                    Kind::UnOp { op } => {
                        let op = op
                            .unop(node.ty, fuc.nodes[node.inputs[1]].ty)
                            .expect("TODO: unary operator not supported");
                        self.emit(op(atr(nid), atr(node.inputs[1])));
                    }
                    Kind::BinOp { .. } if node.lock_rc != 0 => {}
                    Kind::BinOp { op } => {
                        let &[.., lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                        if let Kind::CInt { value } = fuc.nodes[rhs].kind
                            && fuc.nodes[rhs].lock_rc != 0
                            && let Some(op) = op.imm_binop(node.ty)
                        {
                            self.emit(op(atr(nid), atr(lhs), value as _));
                        } else if let Some(op) =
                            op.binop(node.ty).or(op.float_cmp(fuc.nodes[lhs].ty))
                        {
                            self.emit(op(atr(nid), atr(lhs), atr(rhs)));
                        } else if let Some(against) = op.cmp_against() {
                            let op_ty = fuc.nodes[lhs].ty;

                            self.emit(extend(
                                fuc.nodes[lhs].ty,
                                fuc.nodes[lhs].ty.extend(),
                                lhs,
                                lhs,
                            ));
                            self.emit(extend(
                                fuc.nodes[rhs].ty,
                                fuc.nodes[rhs].ty.extend(),
                                rhs,
                                rhs,
                            ));

                            if op_ty.is_float() && matches!(op, TokenKind::Le | TokenKind::Ge) {
                                let opop = match op {
                                    TokenKind::Le => TokenKind::Gt,
                                    TokenKind::Ge => TokenKind::Lt,
                                    _ => unreachable!(),
                                };
                                let op_fn = opop.float_cmp(op_ty).unwrap();
                                self.emit(op_fn(atr(nid), atr(lhs), atr(rhs)));
                                self.emit(instrs::not(atr(nid), atr(nid)));
                            } else {
                                let op_fn =
                                    if op_ty.is_signed() { instrs::cmps } else { instrs::cmpu };
                                self.emit(op_fn(atr(nid), atr(lhs), atr(rhs)));
                                self.emit(instrs::cmpui(atr(nid), atr(nid), against));
                                if matches!(op, TokenKind::Eq | TokenKind::Lt | TokenKind::Gt) {
                                    self.emit(instrs::not(atr(nid), atr(nid)));
                                }
                            }
                        } else {
                            todo!("unhandled operator: {op}");
                        }
                    }
                    Kind::Call { args, func } => {
                        let (ret, mut parama) = tys.parama(node.ty);
                        let mut args = args.args();
                        let mut allocs = node.inputs[1..].iter();
                        while let Some(arg) = args.next(tys) {
                            let Arg::Value(ty) = arg else { continue };
                            let Some(loc) = parama.next(ty, tys) else { continue };

                            let mut arg = *allocs.next().unwrap();
                            let (rg, size) = match loc {
                                PLoc::Reg(rg, size) if ty.loc(tys) == Loc::Stack => (rg, size),
                                PLoc::WideReg(rg, size) => (rg, size),
                                PLoc::Ref(r, ..) => {
                                    self.emit(instrs::cp(r, atr(arg)));
                                    continue;
                                }
                                PLoc::Reg(r, ..) => {
                                    self.emit(instrs::cp(r, atr(arg)));
                                    continue;
                                }
                            };

                            self.emit(instrs::ld(rg, atr(arg), 0, size));
                        }

                        debug_assert!(
                            !matches!(ret, Some(PLoc::Ref(..))) || allocs.next().is_some()
                        );

                        if let Some(PLoc::Ref(r, ..)) = ret {
                            self.emit(instrs::cp(r, atr(*node.inputs.last().unwrap())))
                        }

                        if func == ty::Func::ECA {
                            self.emit(instrs::eca());
                        } else {
                            self.relocs.push(TypedReloc {
                                target: ty::Kind::Func(func).compress(),
                                reloc: Reloc::new(self.code.len(), 3, 4),
                            });
                            self.emit(instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
                        }

                        match ret {
                            Some(PLoc::WideReg(r, size)) => {
                                debug_assert_eq!(
                                    fuc.nodes[*node.inputs.last().unwrap()].kind,
                                    Kind::Stck
                                );
                                let stck = self.offsets[*node.inputs.last().unwrap() as usize];
                                self.emit(instrs::st(r, reg::STACK_PTR, stck as _, size));
                            }
                            Some(PLoc::Reg(r, size)) if node.ty.loc(tys) == Loc::Stack => {
                                debug_assert_eq!(
                                    fuc.nodes[*node.inputs.last().unwrap()].kind,
                                    Kind::Stck
                                );
                                let stck = self.offsets[*node.inputs.last().unwrap() as usize];
                                self.emit(instrs::st(r, reg::STACK_PTR, stck as _, size));
                            }
                            Some(PLoc::Reg(r, ..)) => self.emit(instrs::cp(atr(nid), r)),
                            None | Some(PLoc::Ref(..)) => {}
                        }
                    }
                    Kind::Global { global } => {
                        let reloc = Reloc::new(self.code.len(), 3, 4);
                        self.relocs.push(TypedReloc {
                            target: ty::Kind::Global(global).compress(),
                            reloc,
                        });
                        self.emit(instrs::lra(atr(nid), 0, 0));
                    }
                    Kind::Stck => {
                        let base = reg::STACK_PTR;
                        let offset = self.offsets[nid as usize];
                        self.emit(instrs::addi64(atr(nid), base, offset as _));
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
                                Kind::Stck => {
                                    (reg::STACK_PTR, self.offsets[region as usize] + offset)
                                }
                                _ => (atr(region), offset),
                            };
                            self.emit(instrs::ld(atr(nid), base, offset as _, size as _));
                        }
                    }
                    Kind::Stre if node.inputs[1] == VOID => {}
                    Kind::Stre => {
                        let mut region = node.inputs[2];
                        let mut offset = 0;
                        let size = u16::try_from(tys.size_of(node.ty)).expect("TODO");
                        if matches!(fuc.nodes[region].kind, Kind::BinOp {
                            op: TokenKind::Add | TokenKind::Sub
                        }) && let Kind::CInt { value } =
                            fuc.nodes[fuc.nodes[region].inputs[2]].kind
                            && node.ty.loc(tys) == Loc::Reg
                        {
                            region = fuc.nodes[region].inputs[1];
                            offset = value as Offset;
                        }
                        let nd = &fuc.nodes[region];
                        let value = node.inputs[1];
                        let (base, offset, src) = match nd.kind {
                            Kind::Stck if node.ty.loc(tys) == Loc::Reg => {
                                (reg::STACK_PTR, self.offsets[region as usize] + offset, value)
                            }
                            _ => (atr(region), offset, value),
                        };

                        match node.ty.loc(tys) {
                            Loc::Reg => self.emit(instrs::st(atr(src), base, offset as _, size)),
                            Loc::Stack => {
                                debug_assert_eq!(offset, 0);
                                self.emit(instrs::bmc(atr(src), base, size))
                            }
                        }
                    }

                    Kind::Mem => self.emit(instrs::cp(atr(MEM), reg::RET)),
                    Kind::Arg => {}
                    e @ (Kind::Start
                    | Kind::Entry
                    | Kind::End
                    | Kind::Loops
                    | Kind::Then
                    | Kind::Else
                    | Kind::Phi
                    | Kind::Assert { .. }) => unreachable!("{e:?}"),
                }
            }
        }

        self.ralloc_my = res;

        let bundle_count = self.ralloc_my.bundles.len() + (reg_offset as usize);
        (
            if fuc.tail {
                bundle_count.saturating_sub(reg::RET_ADDR as _)
            } else {
                assert!(bundle_count < reg::STACK_PTR as usize, "TODO: spill memory");
                self.ralloc_my.bundles.len()
            },
            fuc.tail,
        )
    }
}

pub struct Function<'a> {
    sig: Sig,
    tail: bool,
    backrefs: Vec<u16>,
    nodes: &'a mut Nodes,
    tys: &'a Types,
    visited: BitSet,
    func: Func,
}

impl Function<'_> {
    fn vreg_count(&self) -> usize {
        self.nodes.values.len()
    }

    fn uses_of(&self, nid: Nid, buf: &mut Vec<(Nid, Nid)>) {
        if self.nodes[nid].kind.is_cfg() && !matches!(self.nodes[nid].kind, Kind::Call { .. }) {
            return;
        }

        self.nodes[nid]
            .outputs
            .iter()
            .filter(|&&n| self.nodes.is_data_dep(nid, n))
            .map(|n| self.nodes.this_or_delegates(nid, n))
            .flat_map(|(p, ls)| ls.iter().map(move |l| (p, l)))
            .filter(|&(o, &n)| self.nodes.is_data_dep(o, n))
            .map(|(p, &n)| (self.use_block(p, n), n))
            .inspect(|&(_, n)| debug_assert_eq!(self.nodes[n].lock_rc, 0))
            .collect_into(buf);
    }

    fn use_block(&self, inst: Nid, uinst: Nid) -> Nid {
        let mut block = self.nodes.use_block(inst, uinst);
        while !self.nodes[block].kind.starts_basic_block() {
            block = self.nodes.idom(block);
        }
        block
    }

    fn phi_inputs_of(&self, nid: Nid, buf: &mut Vec<Nid>) {
        match self.nodes[nid].kind {
            Kind::Region => {
                for &inp in self.nodes[nid].outputs.as_slice() {
                    if self.nodes[inp].is_data_phi() {
                        buf.extend(&self.nodes[inp].inputs[1..]);
                        buf.push(inp);
                    }
                }
            }
            Kind::Loop => {
                for &inp in self.nodes[nid].outputs.as_slice() {
                    if self.nodes[inp].is_data_phi() {
                        buf.push(self.nodes[inp].inputs[1]);
                        buf.push(inp);
                        buf.push(self.nodes[inp].inputs[2]);
                    }
                }
            }
            _ => {}
        }
    }

    fn instr_of(&self, nid: Nid) -> Option<Nid> {
        if self.nodes[nid].kind == Kind::Phi || self.nodes[nid].lock_rc != 0 {
            return None;
        }
        debug_assert_ne!(self.backrefs[nid as usize], Nid::MAX, "{:?}", self.nodes[nid]);
        Some(self.backrefs[nid as usize])
    }

    fn block_of(&self, nid: Nid) -> Nid {
        debug_assert!(self.nodes[nid].kind.starts_basic_block());
        self.backrefs[nid as usize]
    }

    fn idom_of(&self, mut nid: Nid) -> Nid {
        while !self.nodes[nid].kind.starts_basic_block() {
            nid = self.nodes.idom(nid);
        }
        nid
    }
}

impl core::fmt::Debug for Function<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for block in &self.func.blocks {
            writeln!(f, "{:?}", self.nodes[block.entry].kind)?;
            for &instr in &self.func.instrs[block.range.clone()] {
                writeln!(f, "{:?}", self.nodes[instr].kind)?;
            }
        }

        Ok(())
    }
}

impl<'a> Function<'a> {
    fn new(nodes: &'a mut Nodes, tys: &'a Types, sig: Sig) -> Self {
        let mut s = Self {
            backrefs: vec![u16::MAX; nodes.values.len()],
            tail: true,
            nodes,
            tys,
            sig,
            visited: Default::default(),
            func: Default::default(),
        };
        s.visited.clear(s.nodes.values.len());
        s.emit_node(VOID);
        s
    }

    fn add_block(&mut self, entry: Nid) {
        self.func
            .blocks
            .push(Block { range: self.func.instrs.len()..self.func.instrs.len(), entry });
        self.backrefs[entry as usize] = self.func.blocks.len() as u16 - 1;
    }

    fn close_block(&mut self, exit: Nid) {
        if !matches!(self.nodes[exit].kind, Kind::Loop | Kind::Region) {
            self.add_instr(exit);
        } else {
            self.func.instrs.push(exit);
        }
        let prev = self.func.blocks.last_mut().unwrap();
        prev.range.end = self.func.instrs.len();
    }

    fn add_instr(&mut self, nid: Nid) {
        debug_assert_ne!(self.nodes[nid].kind, Kind::Loop);
        self.backrefs[nid as usize] = self.func.instrs.len() as u16;
        self.func.instrs.push(nid);
    }

    fn emit_node(&mut self, nid: Nid) {
        if matches!(self.nodes[nid].kind, Kind::Region | Kind::Loop) {
            match (self.nodes[nid].kind, self.visited.set(nid)) {
                (Kind::Loop, false) | (Kind::Region, true) => {
                    self.close_block(nid);
                    return;
                }
                _ => {}
            }
        } else if !self.visited.set(nid) {
            return;
        }

        if self.nodes.is_never_used(nid, self.tys) {
            self.nodes.lock(nid);
            return;
        }

        let mut node = self.nodes[nid].clone();
        match node.kind {
            Kind::Start => {
                debug_assert_matches!(self.nodes[node.outputs[0]].kind, Kind::Entry);
                self.add_block(VOID);
                self.emit_node(node.outputs[0])
            }
            Kind::If => {
                let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Kind::BinOp { op } = self.nodes[cond].kind
                    && let Some((_, swapped)) = op.cond_op(node.ty)
                {
                    if swapped {
                        mem::swap(&mut then, &mut else_);
                    }
                } else {
                    mem::swap(&mut then, &mut else_);
                }

                self.close_block(nid);
                self.emit_node(then);
                self.emit_node(else_);
            }
            Kind::Region | Kind::Loop => {
                self.close_block(nid);
                self.add_block(nid);
                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o);
                }
            }
            Kind::Return | Kind::Die => {
                self.close_block(nid);
                self.emit_node(node.outputs[0]);
            }
            Kind::Entry => {
                let (ret, mut parama) = self.tys.parama(self.sig.ret);
                let mut typs = self.sig.args.args();
                #[expect(clippy::unnecessary_to_owned)]
                let mut args = self.nodes[VOID].outputs[ARG_START..].to_owned().into_iter();
                while let Some(ty) = typs.next_value(self.tys) {
                    let arg = args.next().unwrap();
                    debug_assert_eq!(self.nodes[arg].kind, Kind::Arg);
                    match parama.next(ty, self.tys) {
                        None => {}
                        Some(_) => self.add_instr(arg),
                    }
                }

                if let Some(PLoc::Ref(..)) = ret {
                    self.add_instr(MEM);
                }

                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o);
                }
            }
            Kind::Then | Kind::Else => {
                self.add_block(nid);
                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o);
                }
            }
            Kind::Call { func, .. } => {
                self.tail &= func == ty::Func::ECA;

                self.add_instr(nid);

                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    if self.nodes[o].inputs[0] == nid
                        || (matches!(self.nodes[o].kind, Kind::Loop | Kind::Region)
                            && self.nodes[o].inputs[1] == nid)
                    {
                        self.emit_node(o);
                    }
                }
            }
            Kind::CInt { .. }
            | Kind::BinOp { .. }
            | Kind::UnOp { .. }
            | Kind::Global { .. }
            | Kind::Load { .. }
            | Kind::Stre
            | Kind::Stck => self.add_instr(nid),
            Kind::End | Kind::Phi | Kind::Arg | Kind::Mem | Kind::Loops => {}
            Kind::Assert { .. } => unreachable!(),
        }
    }
}

pub struct Env<'a> {
    ctx: &'a Function<'a>,
    func: &'a Func,
    res: &'a mut Res,
}

impl<'a> Env<'a> {
    pub fn new(ctx: &'a Function<'a>, func: &'a Func, res: &'a mut Res) -> Self {
        Self { ctx, func, res }
    }

    pub fn run(&mut self) {
        self.res.bundles.clear();
        self.res.node_to_reg.clear();
        self.res.node_to_reg.resize(self.ctx.vreg_count(), 0);

        debug_assert!(self.res.dfs_buf.is_empty());
        debug_assert!(self.res.use_buf.is_empty());
        debug_assert!(self.res.phi_input_buf.is_empty());

        let mut bundle = Bundle::new(self.func.instrs.len());
        let mut visited = BitSet::with_capacity(self.ctx.nodes.values.len());
        let mut use_buf = mem::take(&mut self.res.use_buf);

        let mut phi_input_buf = mem::take(&mut self.res.phi_input_buf);
        for block in &self.func.blocks {
            self.ctx.phi_inputs_of(block.entry, &mut phi_input_buf);
            for param in phi_input_buf.drain(..) {
                if !visited.set(param) {
                    continue;
                }
                self.append_bundle(param, &mut bundle, &mut use_buf);
            }
        }
        self.res.phi_input_buf = phi_input_buf;

        for &inst in &self.func.instrs {
            if visited.get(inst) || inst == 0 {
                continue;
            }
            self.append_bundle(inst, &mut bundle, &mut use_buf);
        }

        self.res.use_buf = use_buf;
    }

    fn append_bundle(&mut self, inst: Nid, bundle: &mut Bundle, use_buf: &mut Vec<(Nid, Nid)>) {
        let dom = self.ctx.idom_of(inst);
        self.ctx.uses_of(inst, use_buf);
        for (cursor, uinst) in use_buf.drain(..) {
            self.reverse_cfg_dfs(cursor, dom, |_, n, b| {
                let mut range = b.range.clone();
                debug_assert!(range.start < range.end);
                range.start =
                    range.start.max(self.ctx.instr_of(inst).map_or(0, |n| n + 1) as usize);
                debug_assert!(range.start < range.end, "{:?}", range);
                range.end = range.end.min(
                    self.ctx
                        .instr_of(uinst)
                        .filter(|_| {
                            n == cursor
                                && self.ctx.nodes.loop_depth(dom)
                                    == self.ctx.nodes.loop_depth(cursor)
                        })
                        .map_or(Nid::MAX, |n| n + 1) as usize,
                );
                debug_assert!(range.start < range.end);

                bundle.add(range);
            });
        }

        if !bundle.taken.contains(&true) {
            self.res.node_to_reg[inst as usize] = u8::MAX;
            return;
        }

        match self.res.bundles.iter_mut().enumerate().find(|(_, b)| !b.overlaps(bundle)) {
            Some((i, other)) => {
                other.merge(bundle);
                bundle.clear();
                self.res.node_to_reg[inst as usize] = i as Reg + 1;
            }
            None => {
                self.res.bundles.push(mem::replace(bundle, Bundle::new(self.func.instrs.len())));
                self.res.node_to_reg[inst as usize] = self.res.bundles.len() as Reg;
            }
        }
    }

    fn reverse_cfg_dfs(
        &mut self,
        from: Nid,
        until: Nid,
        mut each: impl FnMut(&mut Self, Nid, &Block),
    ) {
        debug_assert!(self.res.dfs_buf.is_empty());
        self.res.dfs_buf.push(from);
        self.res.dfs_seem.clear(self.ctx.nodes.values.len());

        debug_assert!(self.ctx.nodes.dominates(until, from));

        while let Some(nid) = self.res.dfs_buf.pop() {
            debug_assert!(
                self.ctx.nodes.dominates(until, nid),
                "{until} {:?}",
                self.ctx.nodes[until]
            );
            each(self, nid, &self.func.blocks[self.ctx.block_of(nid) as usize]);
            if nid == until {
                continue;
            }
            match self.ctx.nodes[nid].kind {
                Kind::Then | Kind::Else | Kind::Region | Kind::Loop => {
                    for &n in self.ctx.nodes[nid].inputs.iter() {
                        if self.ctx.nodes[n].kind == Kind::Loops {
                            continue;
                        }
                        let d = self.ctx.idom_of(n);
                        if self.res.dfs_seem.set(d) {
                            self.res.dfs_buf.push(d);
                        }
                    }
                }
                Kind::Start => {}
                _ => unreachable!(),
            }
        }
    }
}

#[derive(Default)]
pub struct Res {
    pub bundles: Vec<Bundle>,
    pub node_to_reg: Vec<Reg>,
    use_buf: Vec<(Nid, Nid)>,
    phi_input_buf: Vec<Nid>,
    dfs_buf: Vec<Nid>,
    dfs_seem: BitSet,
}

pub struct Bundle {
    taken: Vec<bool>,
}

impl Bundle {
    fn new(size: usize) -> Self {
        Self { taken: vec![false; size] }
    }

    fn add(&mut self, range: Range<usize>) {
        self.taken[range].fill(true);
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.taken.iter().zip(other.taken.iter()).any(|(a, b)| a & b)
    }

    fn merge(&mut self, other: &Self) {
        debug_assert!(!self.overlaps(other));
        self.taken.iter_mut().zip(other.taken.iter()).for_each(|(a, b)| *a |= *b);
    }

    fn clear(&mut self) {
        self.taken.fill(false);
    }
}

#[derive(Default)]
pub struct Func {
    pub blocks: Vec<Block>,
    pub instrs: Vec<Nid>,
}

pub struct Block {
    pub range: Range<usize>,
    pub entry: Nid,
}
