use {
    super::{HbvmBackend, Nid, Nodes},
    crate::{
        parser,
        reg::{self, Reg},
        son::{debug_assert_matches, Kind, ARG_START, MEM, VOID},
        ty::{self, Arg, Loc},
        utils::BitSet,
        PLoc, Sig, Types,
    },
    alloc::{borrow::ToOwned, vec::Vec},
    core::{cell::RefCell, mem, ops::Range},
    hashbrown::HashSet,
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

        let mut alloc_buf = vec![];
        for (i, block) in fuc.func.blocks.iter().enumerate() {
            self.offsets[block.entry as usize] = self.code.len() as _;
            for &nid in &fuc.func.instrs[block.range.clone()] {
                if nid == VOID {
                    continue;
                }

                let node = &fuc.nodes[nid];
                alloc_buf.clear();

                let atr = |allc: Nid| {
                    let allc = strip_load(allc);
                    debug_assert_eq!(
                        fuc.nodes[allc].lock_rc,
                        0,
                        "{:?} {}",
                        fuc.nodes[allc],
                        ty::Display::new(tys, files, fuc.nodes[allc].ty)
                    );
                    debug_assert!(
                        fuc.marked.borrow().contains(&(allc, nid))
                            || nid == allc
                            || fuc.nodes.is_hard_zero(allc)
                            || allc == MEM
                            || matches!(node.kind, Kind::Loop | Kind::Region),
                        "{nid} {:?}\n{allc} {:?}",
                        fuc.nodes[nid],
                        fuc.nodes[allc]
                    );
                    res.node_to_reg[allc as usize]
                };

                let mut is_next_block = false;
                match node.kind {
                    Kind::If => {
                        let &[_, cnd] = node.inputs.as_slice() else { unreachable!() };
                        if let Kind::BinOp { op } = fuc.nodes[cnd].kind
                            && op.cond_op(fuc.nodes[fuc.nodes[cnd].inputs[1]].ty).is_some()
                        {
                            let &[_, lh, rh] = fuc.nodes[cnd].inputs.as_slice() else {
                                unreachable!()
                            };
                            alloc_buf.extend([atr(lh), atr(rh)]);
                        } else {
                            alloc_buf.push(atr(cnd));
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
                        is_next_block = fuc.block_of(nid) as usize == i + 1;
                    }
                    Kind::Return => {
                        let &[_, ret, ..] = node.inputs.as_slice() else { unreachable!() };
                        match retl {
                            Some(PLoc::Reg(r, _)) if sig.ret.loc(tys) == Loc::Reg => {
                                alloc_buf.push(atr(ret));
                                self.emit(instrs::cp(r, atr(ret)));
                            }
                            Some(PLoc::Ref(..)) => alloc_buf.extend([atr(ret), atr(MEM)]),
                            Some(_) => alloc_buf.push(atr(ret)),
                            None => {}
                        }
                    }
                    Kind::Die => self.emit(instrs::un()),
                    Kind::CInt { .. } => alloc_buf.push(atr(nid)),
                    Kind::UnOp { .. } => alloc_buf.extend([atr(nid), atr(node.inputs[1])]),
                    Kind::BinOp { .. } if node.lock_rc != 0 => {}
                    Kind::BinOp { op } => {
                        let &[.., lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                        if let Kind::CInt { .. } = fuc.nodes[rhs].kind
                            && fuc.nodes[rhs].lock_rc != 0
                            && op.imm_binop(node.ty).is_some()
                        {
                            alloc_buf.extend([atr(nid), atr(lhs)]);
                        } else {
                            alloc_buf.extend([atr(nid), atr(lhs), atr(rhs)]);
                        }
                    }
                    Kind::Call { args, .. } => {
                        let (ret, mut parama) = tys.parama(node.ty);
                        if ret.is_some() {
                            alloc_buf.push(atr(nid));
                        }
                        let mut args = args.args();
                        let mut allocs = node.inputs[1..].iter();
                        while let Some(arg) = args.next(tys) {
                            let Arg::Value(ty) = arg else { continue };
                            let Some(loc) = parama.next(ty, tys) else { continue };

                            let arg = *allocs.next().unwrap();
                            alloc_buf.push(atr(arg));
                            match loc {
                                PLoc::Reg(..) if ty.loc(tys) == Loc::Stack => {}
                                PLoc::WideReg(..) => alloc_buf.push(0),
                                PLoc::Reg(r, ..) | PLoc::Ref(r, ..) => {
                                    self.emit(instrs::cp(r, atr(arg)))
                                }
                            };
                        }

                        if node.ty.loc(tys) == Loc::Stack
                            && let Some(PLoc::Reg(r, ..) | PLoc::WideReg(r, ..) | PLoc::Ref(r, ..)) =
                                ret
                        {
                            alloc_buf.push(atr(*node.inputs.last().unwrap()));
                            self.emit(instrs::cp(r, *alloc_buf.last().unwrap()))
                        }
                    }
                    Kind::Stck | Kind::Global { .. } => alloc_buf.push(atr(nid)),
                    Kind::Load => {
                        let (region, _) = fuc.nodes.strip_offset(node.inputs[1], node.ty, tys);
                        if node.ty.loc(tys) != Loc::Stack {
                            alloc_buf.push(atr(nid));
                            match fuc.nodes[region].kind {
                                Kind::Stck => {}
                                _ => alloc_buf.push(atr(region)),
                            }
                        }
                    }
                    Kind::Stre if node.inputs[1] == VOID => {}
                    Kind::Stre => {
                        let (region, _) = fuc.nodes.strip_offset(node.inputs[2], node.ty, tys);
                        match fuc.nodes[region].kind {
                            Kind::Stck if node.ty.loc(tys) == Loc::Reg => {
                                alloc_buf.push(atr(node.inputs[1]))
                            }
                            _ => alloc_buf.extend([atr(region), atr(node.inputs[1])]),
                        }
                    }
                    Kind::Mem => {
                        self.emit(instrs::cp(atr(MEM), reg::RET));
                        continue;
                    }
                    Kind::Arg => {
                        continue;
                    }
                    _ => {}
                }

                self.emit_instr(super::InstrCtx {
                    nid,
                    sig,
                    is_next_block,
                    is_last_block: i == fuc.func.blocks.len() - 1,
                    retl,
                    allocs: &alloc_buf,
                    nodes: fuc.nodes,
                    tys,
                    files,
                });

                if let Kind::Call { .. } = node.kind {
                    let (ret, ..) = tys.parama(node.ty);

                    match ret {
                        Some(PLoc::WideReg(..)) => {}
                        Some(PLoc::Reg(..)) if node.ty.loc(tys) == Loc::Stack => {}
                        Some(PLoc::Reg(r, ..)) => self.emit(instrs::cp(atr(nid), r)),
                        None | Some(PLoc::Ref(..)) => {}
                    }
                }
            }
        }

        self.ralloc_my = res;

        let bundle_count = self.ralloc_my.bundles.len() + (reg_offset as usize);
        (
            if fuc.tail {
                assert!(bundle_count < reg::STACK_PTR as usize, "TODO: spill memory");
                self.ralloc_my.bundles.len()
            } else {
                bundle_count.saturating_sub(reg::RET_ADDR as _)
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
    marked: RefCell<HashSet<(Nid, Nid), crate::FnvBuildHasher>>,
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
            .inspect(|&(_, n)| _ = self.marked.borrow_mut().insert((nid, n)))
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
            Kind::Region | Kind::Loop => {
                for &inp in self.nodes[nid].outputs.as_slice() {
                    if self.nodes[inp].is_data_phi() {
                        buf.push(inp);
                        buf.extend(&self.nodes[inp].inputs[1..]);
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
            marked: Default::default(),
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
            Kind::CInt { value: 0 } if self.nodes.is_hard_zero(nid) => {}
            Kind::CInt { .. }
            | Kind::BinOp { .. }
            | Kind::UnOp { .. }
            | Kind::Global { .. }
            | Kind::Load { .. }
            | Kind::Stre
            | Kind::Stck => self.add_instr(nid),
            Kind::End | Kind::Phi | Kind::Arg | Kind::Mem | Kind::Loops | Kind::Join => {}
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
        for block in self.func.blocks.iter().rev() {
            self.ctx.phi_inputs_of(block.entry, &mut phi_input_buf);
            for [a, rest @ ..] in phi_input_buf.drain(..).array_chunks::<3>() {
                if visited.set(a) {
                    self.append_bundle(a, &mut bundle, &mut use_buf, None);
                }

                for r in rest {
                    if !visited.set(r) {
                        continue;
                    }

                    self.append_bundle(
                        r,
                        &mut bundle,
                        &mut use_buf,
                        Some(self.res.node_to_reg[a as usize] as usize - 1),
                    );
                }
            }
        }
        self.res.phi_input_buf = phi_input_buf;

        for &inst in &self.func.instrs {
            if visited.get(inst) || inst == 0 {
                continue;
            }
            self.append_bundle(inst, &mut bundle, &mut use_buf, None);
        }

        self.res.use_buf = use_buf;
    }

    fn append_bundle(
        &mut self,
        inst: Nid,
        bundle: &mut Bundle,
        use_buf: &mut Vec<(Nid, Nid)>,
        prefered: Option<usize>,
    ) {
        let dom = self.ctx.idom_of(inst);
        self.ctx.uses_of(inst, use_buf);
        for (cursor, uinst) in use_buf.drain(..) {
            self.reverse_cfg_dfs(cursor, dom, |_, n, b| {
                let mut range = b.range.clone();
                debug_assert!(range.start < range.end);
                range.start =
                    range.start.max(self.ctx.instr_of(inst).map_or(0, |n| n + 1) as usize);
                debug_assert!(range.start < range.end, "{:?}", range);
                let new = range.end.min(
                    self.ctx
                        .instr_of(uinst)
                        .filter(|_| {
                            n == cursor
                                && self.ctx.nodes.loop_depth(dom)
                                    == self.ctx.nodes.loop_depth(cursor)
                        })
                        .map_or(Nid::MAX, |n| n + 1) as usize,
                );

                range.end = new;
                debug_assert!(range.start < range.end);

                bundle.add(range);
            });
        }

        if !bundle.taken.contains(&true) {
            self.res.node_to_reg[inst as usize] = u8::MAX;
            return;
        }

        if let Some(prefered) = prefered
            && !self.res.bundles[prefered].overlaps(bundle)
        {
            self.res.bundles[prefered].merge(bundle);
            bundle.clear();
            self.res.node_to_reg[inst as usize] = prefered as Reg + 1;
        } else {
            match self.res.bundles.iter_mut().enumerate().find(|(_, b)| !b.overlaps(bundle)) {
                Some((i, other)) => {
                    other.merge(bundle);
                    bundle.clear();
                    self.res.node_to_reg[inst as usize] = i as Reg + 1;
                }
                None => {
                    self.res
                        .bundles
                        .push(mem::replace(bundle, Bundle::new(self.func.instrs.len())));
                    self.res.node_to_reg[inst as usize] = self.res.bundles.len() as Reg;
                }
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
