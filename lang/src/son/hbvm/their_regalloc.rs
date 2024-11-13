use {
    super::{HbvmBackend, Nid, Nodes},
    crate::{
        lexer::TokenKind,
        parser, reg,
        son::{debug_assert_matches, Kind, ARG_START, MEM, NEVER, VOID},
        ty::{self, Arg, Loc},
        utils::BitSet,
        HashMap, PLoc, Sig, Types,
    },
    alloc::{borrow::ToOwned, vec::Vec},
    core::mem,
    hbbytecode::{self as instrs},
};

pub struct Regalloc {
    env: regalloc2::MachineEnv,
    ctx: regalloc2::Ctx,
}

impl Default for Regalloc {
    fn default() -> Self {
        Self {
            env: regalloc2::MachineEnv {
                preferred_regs_by_class: [
                    (1..13).map(|i| regalloc2::PReg::new(i, regalloc2::RegClass::Int)).collect(),
                    vec![],
                    vec![],
                ],
                non_preferred_regs_by_class: [
                    (13..64).map(|i| regalloc2::PReg::new(i, regalloc2::RegClass::Int)).collect(),
                    vec![],
                    vec![],
                ],
                scratch_by_class: Default::default(),
                fixed_stack_slots: Default::default(),
            },
            ctx: Default::default(),
        }
    }
}

impl HbvmBackend {
    pub fn emit_body_code(
        &mut self,
        nodes: &mut Nodes,
        sig: Sig,
        tys: &Types,
        files: &[parser::Ast],
    ) -> (usize, bool) {
        let mut ralloc = mem::take(&mut self.ralloc);

        let fuc = Function::new(nodes, tys, files, sig);
        log::info!("{:?}", fuc);
        if !fuc.tail {
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
                fuc.nodes.graphviz_in_browser(ty::Display::new(tys, files, ty::Id::VOID));
                panic!("{err}")
            },
        );

        if !fuc.tail {
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

        '_open_function: {
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
            self.emit(instrs::st(reg::RET_ADDR + fuc.tail as u8, reg::STACK_PTR, 0, 0));
        }

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
            self.emit(instrs::st(rg, reg::STACK_PTR, self.offsets[arg as usize] as _, size));
            if fuc.nodes[arg].lock_rc == 0 {
                self.emit(instrs::addi64(rg, reg::STACK_PTR, self.offsets[arg as usize] as _));
            }
        }

        let mut alloc_buf = vec![];
        for (i, block) in fuc.blocks.iter().enumerate() {
            let blk = regalloc2::Block(i as _);
            self.offsets[block.nid as usize] = self.code.len() as _;
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

                self.emit_instr(super::InstrCtx {
                    nid,
                    sig,
                    is_next_block: fuc.backrefs[nid as usize] as usize == i + 1,
                    is_last_block: i == fuc.blocks.len() - 1,
                    retl,
                    allocs: {
                        alloc_buf.clear();
                        alloc_buf.extend(
                            ralloc.ctx.output.inst_allocs(inst).iter().copied().map(&mut atr),
                        );
                        alloc_buf.as_slice()
                    },
                    nodes: fuc.nodes,
                    tys,
                    files,
                });
            }
        }

        self.ralloc = ralloc;

        (saved_regs.len(), fuc.tail)
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
    files: &'a [parser::Ast],
    tail: bool,
    visited: BitSet,
    backrefs: Vec<u16>,
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
    fn new(nodes: &'a mut Nodes, tys: &'a Types, files: &'a [parser::Ast], sig: Sig) -> Self {
        let mut s = Self {
            tys,
            sig,
            files,
            tail: true,
            visited: Default::default(),
            backrefs: vec![u16::MAX; nodes.values.len()],
            blocks: Default::default(),
            instrs: Default::default(),
            nodes,
        };
        s.visited.clear(s.nodes.values.len());
        s.emit_node(VOID, VOID);
        s.add_block(0);
        s.blocks.pop();
        s
    }

    fn add_block(&mut self, nid: Nid) -> u16 {
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
        self.blocks.len() as u16 - 1
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

    fn rg(&mut self, nid: Nid) -> regalloc2::VReg {
        debug_assert!(
            !self.nodes.is_cfg(nid) || matches!(self.nodes[nid].kind, Kind::Call { .. }),
            "{:?}",
            self.nodes[nid]
        );
        debug_assert_eq!(
            { self.nodes[nid].lock_rc },
            0,
            "{nid} {:?} {:?} {:?}",
            self.nodes[nid].clone(),
            nid,
            {
                self.nodes[nid].lock_rc = u16::MAX - 1;
                self.nodes.graphviz_in_browser(ty::Display::new(
                    self.tys,
                    self.files,
                    ty::Id::VOID,
                ));
            }
        );
        debug_assert!(self.nodes[nid].kind != Kind::Phi || self.nodes[nid].ty != ty::Id::VOID);
        if self.nodes.is_hard_zero(nid) {
            regalloc2::VReg::new(NEVER as _, regalloc2::RegClass::Int)
        } else {
            regalloc2::VReg::new(nid as _, regalloc2::RegClass::Int)
        }
    }

    fn emit_node(&mut self, nid: Nid, prev: Nid) {
        if matches!(self.nodes[nid].kind, Kind::Region | Kind::Loop) {
            let prev_bref = self.backrefs[prev as usize];
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

            match (self.nodes[nid].kind, self.visited.set(nid)) {
                (Kind::Loop, false) => {
                    for i in node.inputs {
                        self.bridge(i, nid);
                    }
                    return;
                }
                (Kind::Region, true) => return,
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
                self.emit_node(node.outputs[0], VOID)
            }
            Kind::If => {
                self.backrefs[nid as usize] = self.backrefs[prev as usize];

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
                self.backrefs[nid as usize] = self.add_block(nid);
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
                self.blocks[self.backrefs[nid as usize] as usize].params = block;
                self.nodes.reschedule_block(nid, &mut node.outputs);
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
                    Some(PLoc::WideReg(..)) => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1])]
                    }
                    Some(PLoc::Ref(..)) => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1]), self.urg(MEM)]
                    }
                };

                self.add_instr(nid, ops);
                self.emit_node(node.outputs[0], nid);
            }
            Kind::Die => {
                self.add_instr(nid, vec![]);
                self.emit_node(node.outputs[0], nid);
            }
            Kind::CInt { value: 0 } if self.nodes.is_hard_zero(nid) => {}
            Kind::CInt { .. } => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Entry => {
                self.backrefs[nid as usize] = self.add_block(nid);

                self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                    regalloc2::VReg::new(NEVER as _, regalloc2::RegClass::Int),
                    regalloc2::PReg::new(0, regalloc2::RegClass::Int),
                )]);

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
                            let a = self.rg(arg);
                            self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                                a,
                                regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                            )]);
                        }
                    }
                }

                if let Some(PLoc::Ref(r, ..)) = ret {
                    let m = self.rg(MEM);
                    self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                        m,
                        regalloc2::PReg::new(r as _, regalloc2::RegClass::Int),
                    )]);
                }

                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::Then | Kind::Else => {
                self.backrefs[nid as usize] = self.add_block(nid);
                self.bridge(prev, nid);
                self.nodes.reschedule_block(nid, &mut node.outputs);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
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
            Kind::Call { args, func } => {
                self.tail &= func == ty::Func::ECA;
                self.backrefs[nid as usize] = self.backrefs[prev as usize];
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
                        PLoc::WideReg(r, size) | PLoc::Reg(r, size) => {
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
                            if size > 8 {
                                ops.push(regalloc2::Operand::reg_fixed_use(
                                    self.rg(i),
                                    regalloc2::PReg::new((r + 1) as _, regalloc2::RegClass::Int),
                                ));
                            }
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

                self.nodes.reschedule_block(nid, &mut node.outputs);
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
            Kind::Stck => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Assert { .. } => unreachable!(),
            Kind::End | Kind::Phi | Kind::Arg | Kind::Mem | Kind::Loops | Kind::Join => {}
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
            Kind::Stre => {
                debug_assert_ne!(self.tys.size_of(node.ty), 0);
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
        if self.backrefs[pred as usize] == u16::MAX || self.backrefs[succ as usize] == u16::MAX {
            return;
        }
        self.blocks[self.backrefs[pred as usize] as usize]
            .succs
            .push(regalloc2::Block::new(self.backrefs[succ as usize] as usize));
        self.blocks[self.backrefs[succ as usize] as usize]
            .preds
            .push(regalloc2::Block::new(self.backrefs[pred as usize] as usize));
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
        matches!(self.nodes[self.instrs[insn.index()].nid].kind, Kind::Return | Kind::Die)
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
