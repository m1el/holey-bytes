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
    core::{mem, ops::Range},
    hbbytecode::{self as instrs},
};

impl HbvmBackend {
    pub(super) fn emit_body_code(
        &mut self,
        nodes: &Nodes,
        sig: Sig,
        tys: &Types,
        files: &[parser::Ast],
    ) -> (usize, bool) {
        let tail = Function::build(nodes, tys, &mut self.ralloc, sig);

        let strip_load = |value| match nodes[value].kind {
            Kind::Load { .. } if nodes[value].ty.loc(tys) == Loc::Stack => nodes[value].inputs[1],
            _ => value,
        };

        let mut res = mem::take(&mut self.ralloc);

        Regalloc::run(nodes, &mut res);

        '_open_function: {
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
            self.emit(instrs::st(reg::RET_ADDR + tail as u8, reg::STACK_PTR, 0, 0));
        }

        res.node_to_reg[MEM as usize] = res.bundles.len() as u8 + 1;

        let reg_offset = if tail { reg::RET + 12 } else { reg::RET_ADDR + 1 };

        res.node_to_reg.iter_mut().filter(|r| **r != 0).for_each(|r| {
            if *r == u8::MAX {
                *r = 0
            } else {
                *r += reg_offset - 1;
                if tail && *r >= reg::RET_ADDR {
                    *r += 1;
                }
            }
        });

        let atr = |allc: Nid| {
            let allc = strip_load(allc);
            debug_assert_eq!(
                nodes[allc].lock_rc.get(),
                0,
                "{:?} {}",
                nodes[allc],
                ty::Display::new(tys, files, nodes[allc].ty)
            );
            res.node_to_reg[allc as usize]
        };

        let (retl, mut parama) = tys.parama(sig.ret);
        let mut typs = sig.args.args();
        let mut args = nodes[VOID].outputs[ARG_START..].iter();
        while let Some(aty) = typs.next(tys) {
            let Arg::Value(ty) = aty else { continue };
            let Some(loc) = parama.next(ty, tys) else { continue };
            let &arg = args.next().unwrap();
            let (rg, size) = match loc {
                PLoc::WideReg(rg, size) => (rg, size),
                PLoc::Reg(rg, size) if ty.loc(tys) == Loc::Stack => (rg, size),
                PLoc::Reg(r, ..) | PLoc::Ref(r, ..) => {
                    self.emit_cp(atr(arg), r);
                    continue;
                }
            };
            self.emit(instrs::st(rg, reg::STACK_PTR, self.offsets[arg as usize] as _, size));
            if nodes.is_unlocked(arg) {
                self.emit(instrs::addi64(rg, reg::STACK_PTR, self.offsets[arg as usize] as _));
            }
            self.emit_cp(atr(arg), rg);
        }

        let mut alloc_buf = vec![];
        for (i, block) in res.blocks.iter().enumerate() {
            self.offsets[block.entry as usize] = self.code.len() as _;
            for &nid in &res.instrs[block.range()] {
                if nid == VOID {
                    continue;
                }

                let node = &nodes[nid];
                alloc_buf.clear();

                let atr = |allc: Nid| {
                    let allc = strip_load(allc);
                    debug_assert_eq!(
                        nodes[allc].lock_rc.get(),
                        0,
                        "{:?} {}",
                        nodes[allc],
                        ty::Display::new(tys, files, nodes[allc].ty)
                    );
                    #[cfg(debug_assertions)]
                    debug_assert!(
                        res.marked.contains(&(allc, nid))
                            || nid == allc
                            || nodes.is_hard_zero(allc)
                            || allc == MEM
                            || matches!(node.kind, Kind::Loop | Kind::Region),
                        "{nid} {:?}\n{allc} {:?}",
                        nodes[nid],
                        nodes[allc]
                    );
                    res.node_to_reg[allc as usize]
                };

                let mut is_next_block = false;
                match node.kind {
                    Kind::If => {
                        let &[_, cnd] = node.inputs.as_slice() else { unreachable!() };
                        if nodes.cond_op(cnd).is_some() {
                            let &[_, lh, rh] = nodes[cnd].inputs.as_slice() else { unreachable!() };
                            alloc_buf.extend([atr(lh), atr(rh)]);
                        } else {
                            alloc_buf.push(atr(cnd));
                        }
                    }
                    Kind::Loop | Kind::Region => {
                        let index = node
                            .inputs
                            .iter()
                            .position(|&n| block.entry == nodes.idom_of(n))
                            .unwrap()
                            + 1;

                        let mut moves = vec![];
                        for &out in node.outputs.iter() {
                            if nodes[out].is_data_phi() {
                                let src = nodes[out].inputs[index];
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
                        is_next_block = res.backrefs[nid as usize] as usize == i + 1;
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
                    Kind::Die => {}
                    Kind::CInt { .. } => alloc_buf.push(atr(nid)),
                    Kind::UnOp { .. } => alloc_buf.extend([atr(nid), atr(node.inputs[1])]),
                    Kind::BinOp { op } => {
                        let &[.., lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                        if let Kind::CInt { .. } = nodes[rhs].kind
                            && nodes.is_locked(rhs)
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

                        if node.ty.loc(tys) == Loc::Stack {
                            alloc_buf.push(atr(*node.inputs.last().unwrap()));
                        }

                        if let Some(PLoc::Ref(r, ..)) = ret {
                            self.emit(instrs::cp(r, *alloc_buf.last().unwrap()))
                        }
                    }
                    Kind::Stck | Kind::Global { .. } => alloc_buf.push(atr(nid)),
                    Kind::Load => {
                        let (region, _) = nodes.strip_offset(node.inputs[1], node.ty, tys);
                        if node.ty.loc(tys) != Loc::Stack {
                            alloc_buf.push(atr(nid));
                            match nodes[region].kind {
                                Kind::Stck => {}
                                _ => alloc_buf.push(atr(region)),
                            }
                        }
                    }
                    Kind::Stre if node.inputs[1] == VOID => {}
                    Kind::Stre => {
                        let (region, _) = nodes.strip_offset(node.inputs[2], node.ty, tys);
                        match nodes[region].kind {
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
                    is_last_block: i == res.blocks.len() - 1,
                    retl,
                    allocs: &alloc_buf,
                    nodes,
                    tys,
                    files,
                });

                if let Kind::Call { .. } = node.kind {
                    let (ret, ..) = tys.parama(node.ty);

                    match ret {
                        Some(PLoc::WideReg(..)) => {}
                        Some(PLoc::Reg(..)) if node.ty.loc(tys) == Loc::Stack => {}
                        Some(PLoc::Reg(r, ..)) => self.emit_cp(atr(nid), r),
                        None | Some(PLoc::Ref(..)) => {}
                    }
                }
            }
        }

        self.ralloc = res;

        let bundle_count = self.ralloc.bundles.len() + (reg_offset as usize);
        (
            if tail {
                assert!(bundle_count < reg::STACK_PTR as usize, "TODO: spill memory");
                self.ralloc.bundles.len()
            } else {
                bundle_count.saturating_sub(reg::RET_ADDR as _)
            },
            tail,
        )
    }

    fn emit_cp(&mut self, dst: Reg, src: Reg) {
        if dst != 0 {
            self.emit(instrs::cp(dst, src));
        }
    }
}

struct Function<'a> {
    sig: Sig,
    tail: bool,
    nodes: &'a Nodes,
    tys: &'a Types,
    func: &'a mut Res,
}

impl core::fmt::Debug for Function<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for block in &self.func.blocks {
            writeln!(f, "{:?}", self.nodes[block.entry].kind)?;
            for &instr in &self.func.instrs[block.range()] {
                writeln!(f, "{:?}", self.nodes[instr].kind)?;
            }
        }

        Ok(())
    }
}

impl<'a> Function<'a> {
    fn build(nodes: &'a Nodes, tys: &'a Types, func: &'a mut Res, sig: Sig) -> bool {
        func.blocks.clear();
        func.instrs.clear();
        func.backrefs.resize(nodes.values.len(), u16::MAX);
        func.visited.clear(nodes.values.len());
        let mut s = Self { tail: true, nodes, tys, sig, func };
        s.emit_node(VOID);
        s.tail
    }

    fn add_block(&mut self, entry: Nid) {
        self.func.blocks.push(Block {
            start: self.func.instrs.len() as _,
            end: self.func.instrs.len() as _,
            entry,
        });
        self.func.backrefs[entry as usize] = self.func.blocks.len() as u16 - 1;
    }

    fn close_block(&mut self, exit: Nid) {
        if !matches!(self.nodes[exit].kind, Kind::Loop | Kind::Region) {
            self.add_instr(exit);
        } else {
            self.func.instrs.push(exit);
        }
        let prev = self.func.blocks.last_mut().unwrap();
        prev.end = self.func.instrs.len() as _;
    }

    fn add_instr(&mut self, nid: Nid) {
        debug_assert_ne!(self.nodes[nid].kind, Kind::Loop);
        self.func.backrefs[nid as usize] = self.func.instrs.len() as u16;
        self.func.instrs.push(nid);
    }

    fn emit_node(&mut self, nid: Nid) {
        if matches!(self.nodes[nid].kind, Kind::Region | Kind::Loop) {
            match (self.nodes[nid].kind, self.func.visited.set(nid)) {
                (Kind::Loop, false) | (Kind::Region, true) => {
                    self.close_block(nid);
                    return;
                }
                _ => {}
            }
        } else if !self.func.visited.set(nid) {
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
                let &[_, cnd] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Some((_, swapped)) = self.nodes.cond_op(cnd) {
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

                if let Some(PLoc::Ref(..)) = ret {
                    self.add_instr(MEM);
                }

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

impl Nodes {
    fn vreg_count(&self) -> usize {
        self.values.len()
    }

    fn use_block_of(&self, inst: Nid, uinst: Nid) -> Nid {
        let mut block = self.use_block(inst, uinst);
        while !self[block].kind.starts_basic_block() {
            block = self.idom(block);
        }
        block
    }

    fn phi_inputs_of(&self, nid: Nid) -> impl Iterator<Item = [Nid; 3]> + use<'_> {
        match self[nid].kind {
            Kind::Region | Kind::Loop => Some({
                self[nid]
                    .outputs
                    .as_slice()
                    .iter()
                    .filter(|&&n| self[n].is_data_phi())
                    .map(|&n| [n, self[n].inputs[1], self[n].inputs[2]])
            })
            .into_iter()
            .flatten(),
            _ => None.into_iter().flatten(),
        }
    }

    fn idom_of(&self, mut nid: Nid) -> Nid {
        while !self[nid].kind.starts_basic_block() {
            nid = self.idom(nid);
        }
        nid
    }

    fn uses_of(&self, nid: Nid) -> impl Iterator<Item = (Nid, Nid)> + use<'_> {
        if self[nid].kind.is_cfg() && !matches!(self[nid].kind, Kind::Call { .. }) {
            return None.into_iter().flatten();
        }

        Some(
            self[nid]
                .outputs
                .iter()
                .filter(move |&&n| self.is_data_dep(nid, n))
                .map(move |n| self.this_or_delegates(nid, n))
                .flat_map(|(p, ls)| ls.iter().map(move |l| (p, l)))
                .filter(|&(o, &n)| self.is_data_dep(o, n))
                .map(|(p, &n)| (self.use_block_of(p, n), n))
                .inspect(|&(_, n)| debug_assert_eq!(self[n].lock_rc.get(), 0)),
        )
        .into_iter()
        .flatten()
    }
}

struct Regalloc<'a> {
    nodes: &'a Nodes,
    res: &'a mut Res,
}

impl<'a> Regalloc<'a> {
    fn instr_of(&self, nid: Nid) -> Option<Nid> {
        if self.nodes[nid].kind == Kind::Phi || self.nodes.is_locked(nid) {
            return None;
        }
        debug_assert_ne!(self.res.backrefs[nid as usize], Nid::MAX, "{:?}", self.nodes[nid]);
        Some(self.res.backrefs[nid as usize])
    }

    fn block_of(&self, nid: Nid) -> Nid {
        debug_assert!(self.nodes[nid].kind.starts_basic_block());
        self.res.backrefs[nid as usize]
    }

    fn run(ctx: &'a Nodes, res: &'a mut Res) {
        Self { nodes: ctx, res }.run_low();
    }

    fn run_low(&mut self) {
        self.res.bundles.clear();
        self.res.node_to_reg.clear();
        #[cfg(debug_assertions)]
        self.res.marked.clear();
        self.res.node_to_reg.resize(self.nodes.vreg_count(), 0);

        debug_assert!(self.res.dfs_buf.is_empty());

        let mut bundle = Bundle::new(self.res.instrs.len());
        self.res.visited.clear(self.nodes.values.len());

        for i in (0..self.res.blocks.len()).rev() {
            for [a, rest @ ..] in self.nodes.phi_inputs_of(self.res.blocks[i].entry) {
                if self.res.visited.set(a) {
                    self.append_bundle(a, &mut bundle, None);
                }

                for r in rest {
                    if !self.res.visited.set(r) {
                        continue;
                    }

                    self.append_bundle(
                        r,
                        &mut bundle,
                        Some(self.res.node_to_reg[a as usize] as usize - 1),
                    );
                }
            }
        }

        let instrs = mem::take(&mut self.res.instrs);
        for &inst in &instrs {
            if self.nodes[inst].has_no_value() || self.res.visited.get(inst) || inst == 0 {
                continue;
            }
            self.append_bundle(inst, &mut bundle, None);
        }
        self.res.instrs = instrs;
    }

    fn collect_bundle(&mut self, inst: Nid, into: &mut Bundle) {
        let dom = self.nodes.idom_of(inst);
        self.res.dfs_seem.clear(self.nodes.values.len());
        for (cursor, uinst) in self.nodes.uses_of(inst) {
            if !self.res.dfs_seem.set(uinst) {
                continue;
            }
            #[cfg(debug_assertions)]
            debug_assert!(self.res.marked.insert((inst, uinst)));

            self.reverse_cfg_dfs(cursor, dom, |s, n, b| {
                let mut range = b.range();
                debug_assert!(range.start < range.end);
                range.start = range.start.max(s.instr_of(inst).map_or(0, |n| n + 1) as usize);
                debug_assert!(range.start < range.end, "{:?}", range);
                let new = range.end.min(
                    s.instr_of(uinst)
                        .filter(|_| {
                            n == cursor
                                && self.nodes.loop_depth(dom) == self.nodes.loop_depth(cursor)
                        })
                        .map_or(Nid::MAX, |n| n + 1) as usize,
                );

                range.end = new;
                debug_assert!(range.start < range.end, "{:?} {inst} {uinst}", range);

                into.add(range);
            });
        }
    }

    fn append_bundle(&mut self, inst: Nid, tmp: &mut Bundle, prefered: Option<usize>) {
        self.collect_bundle(inst, tmp);

        if tmp.is_empty() {
            self.res.node_to_reg[inst as usize] = u8::MAX;
            return;
        }

        if let Some(prefered) = prefered
            && !self.res.bundles[prefered].overlaps(tmp)
        {
            self.res.bundles[prefered].merge(tmp);
            tmp.clear();
            self.res.node_to_reg[inst as usize] = prefered as Reg + 1;
            return;
        }

        match self.res.bundles.iter_mut().enumerate().find(|(_, b)| !b.overlaps(tmp)) {
            Some((i, other)) => {
                other.merge(tmp);
                tmp.clear();
                self.res.node_to_reg[inst as usize] = i as Reg + 1;
            }
            None => {
                self.res.bundles.push(tmp.take());
                self.res.node_to_reg[inst as usize] = self.res.bundles.len() as Reg;
            }
        }
    }

    fn reverse_cfg_dfs(
        &mut self,
        from: Nid,
        until: Nid,
        mut each: impl FnMut(&mut Self, Nid, Block),
    ) {
        debug_assert!(self.res.dfs_buf.is_empty());
        self.res.dfs_buf.push(from);

        debug_assert!(self.nodes.dominates(until, from));

        while let Some(nid) = self.res.dfs_buf.pop() {
            debug_assert!(self.nodes.dominates(until, nid), "{until} {:?}", self.nodes[until]);
            each(self, nid, self.res.blocks[self.block_of(nid) as usize]);
            if nid == until {
                continue;
            }
            match self.nodes[nid].kind {
                Kind::Then | Kind::Else | Kind::Region | Kind::Loop => {
                    for &n in self.nodes[nid].inputs.iter() {
                        if self.nodes[n].kind == Kind::Loops {
                            continue;
                        }
                        let d = self.nodes.idom_of(n);
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
pub(super) struct Res {
    blocks: Vec<Block>,
    instrs: Vec<Nid>,
    backrefs: Vec<u16>,

    bundles: Vec<Bundle>,
    node_to_reg: Vec<Reg>,

    visited: BitSet,
    dfs_buf: Vec<Nid>,
    dfs_seem: BitSet,
    #[cfg(debug_assertions)]
    marked: hashbrown::HashSet<(Nid, Nid), crate::FnvBuildHasher>,
}

struct Bundle {
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

    fn is_empty(&self) -> bool {
        !self.taken.contains(&true)
    }

    fn take(&mut self) -> Self {
        mem::replace(self, Self::new(self.taken.len()))
    }
}

#[derive(Clone, Copy)]
struct Block {
    start: u16,
    end: u16,
    entry: Nid,
}

impl Block {
    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}
