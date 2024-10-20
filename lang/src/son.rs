use {
    crate::{
        ctx_map::CtxEntry,
        ident::Ident,
        instrs,
        lexer::{self, TokenKind},
        parser::{
            self,
            idfl::{self},
            CtorField, Expr, ExprRef, FileId, Pos,
        },
        reg, task,
        ty::{self, ArrayLen, Tuple},
        vc::{BitSet, Vc},
        Comptime, Func, Global, HashMap, Offset, OffsetIter, Reloc, Sig, TypeParser, TypedReloc,
        Types,
    },
    alloc::{string::String, vec::Vec},
    core::{
        assert_matches::debug_assert_matches,
        cell::RefCell,
        fmt::{self, Debug, Display, Write},
        format_args as fa, mem,
        ops::{self},
    },
    hashbrown::hash_map,
    regalloc2::VReg,
};

const VOID: Nid = 0;
const NEVER: Nid = 1;
const ENTRY: Nid = 2;
const MEM: Nid = 3;

type Nid = u16;

type Lookup = crate::ctx_map::CtxMap<Nid>;

trait StoreId: Sized {
    fn to_store(self) -> Option<Self>;
}

impl StoreId for Nid {
    fn to_store(self) -> Option<Self> {
        (self != ENTRY).then_some(self)
    }
}

impl crate::ctx_map::CtxEntry for Nid {
    type Ctx = [Result<Node, Nid>];
    type Key<'a> = (Kind, &'a [Nid], ty::Id);

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        ctx[*self as usize].as_ref().unwrap().key()
    }
}

struct Nodes {
    values: Vec<Result<Node, Nid>>,
    visited: BitSet,
    free: Nid,
    lookup: Lookup,
}

impl Default for Nodes {
    fn default() -> Self {
        Self {
            values: Default::default(),
            free: Nid::MAX,
            lookup: Default::default(),
            visited: Default::default(),
        }
    }
}

impl Nodes {
    fn graphviz_low(
        &self,
        tys: &Types,
        files: &[parser::Ast],
        out: &mut String,
    ) -> core::fmt::Result {
        use core::fmt::Write;

        for (i, node) in self.iter() {
            let color = if self.is_cfg(i) { "yellow" } else { "white" };
            writeln!(
                out,
                "node{i}[label=\"{} {}\" color={color}]",
                node.kind,
                ty::Display::new(tys, files, node.ty)
            )?;
            for (j, &o) in node.outputs.iter().enumerate() {
                let color = if self.is_cfg(i) && self.is_cfg(o) { "red" } else { "lightgray" };
                let index = self[o].inputs.iter().position(|&inp| i == inp).unwrap();
                let style = if index == 0 && !self.is_cfg(o) { "style=dotted" } else { "" };
                writeln!(
                    out,
                    "node{o} -> node{i}[color={color} taillabel={index} headlabel={j} {style}]",
                )?;
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn graphviz(&self, tys: &Types, files: &[parser::Ast]) {
        let out = &mut String::new();
        _ = self.graphviz_low(tys, files, out);
        log::info!("{out}");
    }

    fn gcm(&mut self) {
        self.visited.clear(self.values.len());
        push_up(self);
        // TODO: handle infinte loops
        self.visited.clear(self.values.len());
        push_down(self, VOID);
    }

    fn remove_low(&mut self, id: Nid) -> Node {
        if cfg!(debug_assertions) {
            let value = mem::replace(&mut self.values[id as usize], Err(self.free)).unwrap();
            self.free = id;
            value
        } else {
            mem::replace(&mut self.values[id as usize], Err(Nid::MAX)).unwrap()
        }
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = Nid::MAX;
    }

    fn new_node_nop(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Nid {
        let node =
            Node { ralloc_backref: u16::MAX, inputs: inps.into(), kind, ty, ..Default::default() };

        let mut lookup_meta = None;
        if !node.is_not_gvnd() {
            let (raw_entry, hash) = self.lookup.entry(node.key(), &self.values);

            let entry = match raw_entry {
                hash_map::RawEntryMut::Occupied(o) => return o.get_key_value().0.value,
                hash_map::RawEntryMut::Vacant(v) => v,
            };

            lookup_meta = Some((entry, hash));
        }

        if self.free == Nid::MAX {
            self.free = self.values.len() as _;
            self.values.push(Err(Nid::MAX));
        }

        let free = self.free;
        for &d in node.inputs.as_slice() {
            debug_assert_ne!(d, free);
            self.values[d as usize].as_mut().unwrap_or_else(|_| panic!("{d}")).outputs.push(free);
        }
        self.free = mem::replace(&mut self.values[free as usize], Ok(node)).unwrap_err();

        if let Some((entry, hash)) = lookup_meta {
            entry.insert(crate::ctx_map::Key { value: free, hash }, ());
        }
        free
    }

    fn remove_node_lookup(&mut self, target: Nid) {
        if !self[target].is_not_gvnd() {
            self.lookup.remove(&target, &self.values).unwrap();
        }
    }

    fn new_node_low(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> (Nid, bool) {
        let id = self.new_node_nop(ty, kind, inps);
        if let Some(opt) = self.peephole(id) {
            debug_assert_ne!(opt, id);
            self.lock(opt);
            self.remove(id);
            self.unlock(opt);
            (opt, true)
        } else {
            (id, false)
        }
    }

    fn new_node(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Nid {
        self.new_node_low(ty, kind, inps).0
    }

    fn new_node_lit(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Value {
        Value::new(self.new_node_low(ty, kind, inps).0).ty(ty)
    }

    fn lock(&mut self, target: Nid) {
        self[target].lock_rc += 1;
    }

    #[track_caller]
    fn unlock(&mut self, target: Nid) {
        self[target].lock_rc -= 1;
    }

    fn remove(&mut self, target: Nid) -> bool {
        if !self[target].is_dangling() {
            return false;
        }

        for i in 0..self[target].inputs.len() {
            let inp = self[target].inputs[i];
            let index = self[inp].outputs.iter().position(|&p| p == target).unwrap();
            self[inp].outputs.swap_remove(index);
            self.remove(inp);
        }

        self.remove_node_lookup(target);
        self.remove_low(target);

        true
    }

    fn peephole(&mut self, target: Nid) -> Option<Nid> {
        use {Kind as K, TokenKind as T};
        match self[target].kind {
            K::BinOp { op } => {
                let &[ctrl, mut lhs, mut rhs] = self[target].inputs.as_slice() else {
                    unreachable!()
                };
                let ty = self[target].ty;

                if let (&K::CInt { value: a }, &K::CInt { value: b }) =
                    (&self[lhs].kind, &self[rhs].kind)
                {
                    return Some(
                        self.new_node(ty, K::CInt { value: op.apply_binop(a, b) }, [ctrl]),
                    );
                }

                if lhs == rhs {
                    match op {
                        T::Sub => return Some(self.new_node(ty, K::CInt { value: 0 }, [ctrl])),
                        T::Add => {
                            let rhs = self.new_node_nop(ty, K::CInt { value: 2 }, [ctrl]);
                            return Some(
                                self.new_node(ty, K::BinOp { op: T::Mul }, [ctrl, lhs, rhs]),
                            );
                        }
                        _ => {}
                    }
                }

                // this is more general the pushing constants to left to help deduplicate expressions more
                let mut changed = false;
                if op.is_comutative() && self[lhs].key() < self[rhs].key() {
                    core::mem::swap(&mut lhs, &mut rhs);
                    changed = true;
                }

                if let K::CInt { value } = self[lhs].kind
                    && op == T::Sub
                {
                    let lhs = self.new_node_nop(ty, K::CInt { value: -value }, [ctrl]);
                    return Some(self.new_node(ty, K::BinOp { op: T::Add }, [ctrl, rhs, lhs]));
                }

                if let K::CInt { value } = self[rhs].kind {
                    match (op, value) {
                        (T::Add | T::Sub | T::Shl, 0) | (T::Mul | T::Div, 1) => return Some(lhs),
                        (T::Mul, 0) => return Some(rhs),
                        _ => {}
                    }
                }

                if op.is_comutative() && self[lhs].kind == (K::BinOp { op }) {
                    let &[_, a, b] = self[lhs].inputs.as_slice() else { unreachable!() };
                    if let K::CInt { value: av } = self[b].kind
                        && let K::CInt { value: bv } = self[rhs].kind
                    {
                        // (a op #b) op #c => a op (#b op #c)
                        let new_rhs =
                            self.new_node_nop(ty, K::CInt { value: op.apply_binop(av, bv) }, [
                                ctrl,
                            ]);
                        return Some(self.new_node(ty, K::BinOp { op }, [ctrl, a, new_rhs]));
                    }

                    if self.is_const(b) {
                        // (a op #b) op c => (a op c) op #b
                        let new_lhs = self.new_node(ty, K::BinOp { op }, [ctrl, a, rhs]);
                        return Some(self.new_node(ty, K::BinOp { op }, [ctrl, new_lhs, b]));
                    }
                }

                if op == T::Add
                    && self[lhs].kind == (K::BinOp { op: T::Mul })
                    && self[lhs].inputs[1] == rhs
                    && let K::CInt { value } = self[self[lhs].inputs[2]].kind
                {
                    // a * #n + a => a * (#n + 1)
                    let new_rhs = self.new_node_nop(ty, K::CInt { value: value + 1 }, [ctrl]);
                    return Some(self.new_node(ty, K::BinOp { op: T::Mul }, [ctrl, rhs, new_rhs]));
                }

                if op == T::Sub && self[lhs].kind == (K::BinOp { op }) {
                    // (a - b) - c => a - (b + c)
                    let &[_, a, b] = self[lhs].inputs.as_slice() else { unreachable!() };
                    let c = rhs;
                    let new_rhs = self.new_node(ty, K::BinOp { op: T::Add }, [ctrl, b, c]);
                    return Some(self.new_node(ty, K::BinOp { op }, [ctrl, a, new_rhs]));
                }

                if changed {
                    return Some(self.new_node(ty, self[target].kind, [ctrl, lhs, rhs]));
                }
            }
            K::UnOp { op } => {
                let &[ctrl, oper] = self[target].inputs.as_slice() else { unreachable!() };
                let ty = self[target].ty;

                if let K::CInt { value } = self[oper].kind {
                    return Some(
                        self.new_node(ty, K::CInt { value: op.apply_unop(value) }, [ctrl]),
                    );
                }
            }
            K::If => {
                let cond = self[target].inputs[1];
                if let K::CInt { value } = self[cond].kind {
                    let ty = if value == 0 {
                        ty::Id::LEFT_UNREACHABLE
                    } else {
                        ty::Id::RIGHT_UNREACHABLE
                    };
                    return Some(self.new_node_nop(ty, K::If, [self[target].inputs[0], cond]));
                }
            }
            K::Phi => {
                let &[ctrl, lhs, rhs] = self[target].inputs.as_slice() else { unreachable!() };

                if lhs == rhs {
                    return Some(lhs);
                }

                if self[lhs].kind == Kind::Stre
                    && self[rhs].kind == Kind::Stre
                    && self[lhs].ty == self[rhs].ty
                    && self[lhs].inputs[2] == self[rhs].inputs[2]
                    && self[lhs].inputs.get(3) == self[rhs].inputs.get(3)
                {
                    let pick_value = self.new_node(self[lhs].ty, Kind::Phi, [
                        ctrl,
                        self[lhs].inputs[1],
                        self[rhs].inputs[1],
                    ]);
                    let mut vc = Vc::from([VOID, pick_value, self[lhs].inputs[2]]);
                    for &rest in &self[lhs].inputs[3..] {
                        vc.push(rest);
                    }
                    for &rest in &self[rhs].inputs[4..] {
                        vc.push(rest);
                    }
                    return Some(self.new_node(self[lhs].ty, Kind::Stre, vc));
                }
            }
            K::Stre => {
                if self[target].inputs[2] != VOID
                    && self[target].inputs.len() == 4
                    && self[self[target].inputs[3]].kind == Kind::Stre
                    && self[self[target].inputs[3]].lock_rc == 0
                    && self[self[target].inputs[3]].inputs[2] == self[target].inputs[2]
                {
                    return Some(self.modify_input(
                        self[target].inputs[3],
                        1,
                        self[target].inputs[1],
                    ));
                }
            }
            K::Load => {
                if self[target].inputs.len() == 3
                    && self[self[target].inputs[2]].kind == Kind::Stre
                    && self[self[target].inputs[2]].inputs[2] == self[target].inputs[1]
                    && self[self[target].inputs[2]].ty == self[target].ty
                {
                    return Some(self[self[target].inputs[2]].inputs[1]);
                }
            }
            _ => {}
        }

        None
    }

    fn is_const(&self, id: Nid) -> bool {
        matches!(self[id].kind, Kind::CInt { .. })
    }

    fn replace(&mut self, target: Nid, with: Nid) {
        let mut back_press = 0;
        for i in 0..self[target].outputs.len() {
            let out = self[target].outputs[i - back_press];
            let index = self[out].inputs.iter().position(|&p| p == target).unwrap();
            self.lock(target);
            let prev_len = self[target].outputs.len();
            self.modify_input(out, index, with);
            back_press += (self[target].outputs.len() != prev_len) as usize;
            self.unlock(target);
        }

        self.remove(target);
    }

    fn modify_input(&mut self, target: Nid, inp_index: usize, with: Nid) -> Nid {
        self.remove_node_lookup(target);
        debug_assert_ne!(self[target].inputs[inp_index], with);

        let prev = self[target].inputs[inp_index];
        self[target].inputs[inp_index] = with;
        let (entry, hash) = self.lookup.entry(target.key(&self.values), &self.values);
        match entry {
            hash_map::RawEntryMut::Occupied(other) => {
                let rpl = other.get_key_value().0.value;
                self[target].inputs[inp_index] = prev;
                self.replace(target, rpl);
                rpl
            }
            hash_map::RawEntryMut::Vacant(slot) => {
                slot.insert(crate::ctx_map::Key { value: target, hash }, ());
                let index = self[prev].outputs.iter().position(|&o| o == target).unwrap();
                self[prev].outputs.swap_remove(index);
                self[with].outputs.push(target);
                self.remove(prev);

                target
            }
        }
    }

    #[track_caller]
    fn unlock_remove(&mut self, id: Nid) -> bool {
        self[id].lock_rc -= 1;
        self.remove(id)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = (Nid, &Node)> {
        self.values.iter().enumerate().filter_map(|(i, s)| Some((i as _, s.as_ref().ok()?)))
    }

    #[allow(clippy::format_in_format_args)]
    fn basic_blocks_instr(&mut self, out: &mut String, node: Nid) -> core::fmt::Result {
        if self[node].kind != Kind::Loop && self[node].kind != Kind::Region {
            write!(out, "  {node:>2}-c{:>2}: ", self[node].ralloc_backref)?;
        }
        match self[node].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::If => write!(out, "  if:      "),
            Kind::Region | Kind::Loop => writeln!(out, "      goto: {node}"),
            Kind::Return => write!(out, " ret:      "),
            Kind::CInt { value } => write!(out, "cint: #{value:<4}"),
            Kind::Phi => write!(out, " phi:      "),
            Kind::Arg => write!(
                out,
                " arg: {:<5}",
                self[VOID].outputs.iter().position(|&n| n == node).unwrap() - 2
            ),
            Kind::BinOp { op } | Kind::UnOp { op } => {
                write!(out, "{:>4}:      ", op.name())
            }
            Kind::Call { func } => {
                write!(out, "call: {func} {}  ", self[node].depth)
            }
            Kind::Global { global } => write!(out, "glob: {global:<5}"),
            Kind::Entry => write!(out, "ctrl: {:<5}", "entry"),
            Kind::Then => write!(out, "ctrl: {:<5}", "then"),
            Kind::Else => write!(out, "ctrl: {:<5}", "else"),
            Kind::Stck => write!(out, "stck:      "),
            Kind::Load => write!(out, "load:      "),
            Kind::Stre => write!(out, "stre:      "),
            Kind::Mem => write!(out, " mem:      "),
            Kind::Idk => write!(out, " idk:      "),
            Kind::Extend => write!(out, " ext:      "),
        }?;

        if self[node].kind != Kind::Loop && self[node].kind != Kind::Region {
            writeln!(
                out,
                " {:<14} {}",
                format!("{:?}", self[node].inputs),
                format!("{:?}", self[node].outputs)
            )?;
        }

        Ok(())
    }

    fn basic_blocks_low(&mut self, out: &mut String, mut node: Nid) -> core::fmt::Result {
        let iter = |nodes: &Nodes, node| nodes[node].outputs.clone().into_iter().rev();
        while self.visited.set(node) {
            match self[node].kind {
                Kind::Start => {
                    writeln!(out, "start: {}", self[node].depth)?;
                    let mut cfg_index = Nid::MAX;
                    for o in iter(self, node) {
                        self.basic_blocks_instr(out, o)?;
                        if self[o].kind.is_cfg() {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                Kind::End => break,
                Kind::If => {
                    self.basic_blocks_low(out, self[node].outputs[0])?;
                    node = self[node].outputs[1];
                }
                Kind::Region => {
                    writeln!(
                        out,
                        "region{node}: {} {} {:?}",
                        self[node].depth, self[node].loop_depth, self[node].inputs
                    )?;
                    let mut cfg_index = Nid::MAX;
                    for o in iter(self, node) {
                        self.basic_blocks_instr(out, o)?;
                        if self.is_cfg(o) {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                Kind::Loop => {
                    writeln!(
                        out,
                        "loop{node}: {} {} {:?}",
                        self[node].depth, self[node].loop_depth, self[node].outputs
                    )?;
                    let mut cfg_index = Nid::MAX;
                    for o in iter(self, node) {
                        self.basic_blocks_instr(out, o)?;
                        if self.is_cfg(o) {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                Kind::Return => {
                    node = self[node].outputs[0];
                }
                Kind::Then | Kind::Else | Kind::Entry => {
                    writeln!(
                        out,
                        "b{node}: {} {} {:?}",
                        self[node].depth, self[node].loop_depth, self[node].outputs
                    )?;
                    let mut cfg_index = Nid::MAX;
                    for o in iter(self, node) {
                        self.basic_blocks_instr(out, o)?;
                        if self.is_cfg(o) {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                Kind::Call { .. } => {
                    let mut cfg_index = Nid::MAX;
                    let mut print_ret = true;
                    for o in iter(self, node) {
                        if self[o].inputs[0] == node
                            && (self[node].outputs[0] != o || core::mem::take(&mut print_ret))
                        {
                            self.basic_blocks_instr(out, o)?;
                        }
                        if self.is_cfg(o) {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn basic_blocks(&mut self) {
        let mut out = String::new();
        self.visited.clear(self.values.len());
        self.basic_blocks_low(&mut out, VOID).unwrap();
        log::info!("{out}");
    }

    fn is_cfg(&self, o: Nid) -> bool {
        self[o].kind.is_cfg()
    }

    fn check_final_integrity(&self) {
        if !cfg!(debug_assertions) {
            return;
        }

        //let mut failed = false;
        for (_, node) in self.iter() {
            debug_assert_eq!(node.lock_rc, 0, "{:?}", node.kind);
            // if !matches!(node.kind, Kind::Return | Kind::End) && node.outputs.is_empty() {
            //     log::err!("outputs are empry {i} {:?}", node.kind);
            //     failed = true;
            // }

            // let mut allowed_cfgs = 1 + (node.kind == Kind::If) as usize;
            // for &o in node.outputs.iter() {
            //     if self.is_cfg(i) {
            //         if allowed_cfgs == 0 && self.is_cfg(o) {
            //             log::err!(
            //                 "multiple cfg outputs detected: {:?} -> {:?}",
            //                 node.kind,
            //                 self[o].kind
            //             );
            //             failed = true;
            //         } else {
            //             allowed_cfgs += self.is_cfg(o) as usize;
            //         }
            //     }

            //     let other = match &self.values[o as usize] {
            //         Ok(other) => other,
            //         Err(_) => {
            //             log::err!("the edge points to dropped node: {i} {:?} {o}", node.kind,);
            //             failed = true;
            //             continue;
            //         }
            //     };
            //     let occurs = self[o].inputs.iter().filter(|&&el| el == i).count();
            //     let self_occurs = self[i].outputs.iter().filter(|&&el| el == o).count();
            //     if occurs != self_occurs {
            //         log::err!(
            //             "the edge is not bidirectional: {i} {:?} {self_occurs} {o} {:?} {occurs}",
            //             node.kind,
            //             other.kind
            //         );
            //         failed = true;
            //     }
            // }
        }
        //if failed {
        //    panic!()
        //}
    }

    #[expect(dead_code)]
    fn climb_expr(&mut self, from: Nid, mut for_each: impl FnMut(Nid, &Node) -> bool) -> bool {
        fn climb_impl(
            nodes: &mut Nodes,
            from: Nid,
            for_each: &mut impl FnMut(Nid, &Node) -> bool,
        ) -> bool {
            for i in 0..nodes[from].inputs.len() {
                let n = nodes[from].inputs[i];
                if n != Nid::MAX
                    && nodes.visited.set(n)
                    && !nodes.is_cfg(n)
                    && (for_each(n, &nodes[n]) || climb_impl(nodes, n, for_each))
                {
                    return true;
                }
            }
            false
        }
        self.visited.clear(self.values.len());
        climb_impl(self, from, &mut for_each)
    }

    #[expect(dead_code)]
    fn late_peephole(&mut self, target: Nid) -> Nid {
        if let Some(id) = self.peephole(target) {
            self.replace(target, id);
            return id;
        }
        target
    }

    fn load_loop_var(&mut self, index: usize, value: &mut Nid, loops: &mut [Loop]) {
        self.load_loop_value(
            &mut |l| {
                l.scope
                    .vars
                    .get_mut(index)
                    .map_or((ty::Id::VOID, &mut l.scope.store), |v| (v.ty, &mut v.value))
            },
            value,
            loops,
        );
    }

    fn load_loop_store(&mut self, value: &mut Nid, loops: &mut [Loop]) {
        self.load_loop_value(&mut |l| (ty::Id::VOID, &mut l.scope.store), value, loops);
    }

    fn load_loop_value(
        &mut self,
        get_lvalue: &mut impl FnMut(&mut Loop) -> (ty::Id, &mut Nid),
        value: &mut Nid,
        loops: &mut [Loop],
    ) {
        if *value != VOID {
            return;
        }

        let [loob, loops @ ..] = loops else { unreachable!() };
        let node = loob.node;
        let (ty, lvalue) = get_lvalue(loob);

        self.load_loop_value(get_lvalue, lvalue, loops);

        if !self[*lvalue].is_lazy_phi() {
            self.unlock(*value);
            let inps = [node, *lvalue, VOID];
            self.unlock(*lvalue);
            *lvalue = self.new_node_nop(ty, Kind::Phi, inps);
            self[*lvalue].lock_rc += 2;
        } else {
            self.lock(*lvalue);
            self.unlock(*value);
        }
        *value = *lvalue;
    }

    fn check_dominance(&mut self, nd: Nid, min: Nid, check_outputs: bool) {
        if !cfg!(debug_assertions) {
            return;
        }

        let node = self[nd].clone();
        for &i in node.inputs.iter() {
            let dom = idom(self, i);
            debug_assert!(
                self.dominates(dom, min),
                "{dom} {min} {node:?} {:?}",
                self.basic_blocks()
            );
        }
        if check_outputs {
            for &o in node.outputs.iter() {
                let dom = use_block(nd, o, self);
                debug_assert!(
                    self.dominates(min, dom),
                    "{min} {dom} {node:?} {:?}",
                    self.basic_blocks()
                );
            }
        }
    }

    fn dominates(&mut self, dominator: Nid, mut dominated: Nid) -> bool {
        loop {
            if dominator == dominated {
                break true;
            }

            if idepth(self, dominator) > idepth(self, dominated) {
                break false;
            }

            dominated = idom(self, dominated);
        }
    }

    #[expect(dead_code)]
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.values.iter_mut().flat_map(Result::as_mut)
    }

    fn lock_scope(&mut self, scope: &Scope) {
        self.lock(scope.store);
        for &load in &scope.loads {
            self.lock(load);
        }
        for var in &scope.vars {
            self.lock(var.value);
        }
    }

    fn unlock_remove_scope(&mut self, scope: &Scope) {
        self.unlock_remove(scope.store);
        for &load in &scope.loads {
            self.unlock_remove(load);
        }
        for var in &scope.vars {
            self.unlock_remove(var.value);
        }
    }
}

impl ops::Index<Nid> for Nodes {
    type Output = Node;

    fn index(&self, index: Nid) -> &Self::Output {
        self.values[index as usize].as_ref().unwrap()
    }
}

impl ops::IndexMut<Nid> for Nodes {
    fn index_mut(&mut self, index: Nid) -> &mut Self::Output {
        self.values[index as usize].as_mut().unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Kind {
    #[default]
    Start,
    // [ctrl]
    Entry,
    Mem,
    // [terms...]
    End,
    // [ctrl, cond]
    If,
    Then,
    Else,
    // [lhs, rhs]
    Region,
    // [entry, back]
    Loop,
    // [ctrl, ?value]
    Return,
    // [ctrl]
    CInt {
        value: i64,
    },
    // [ctrl, lhs, rhs]
    Phi,
    Arg,
    // [ctrl, oper]
    Extend,
    // [ctrl, oper]
    UnOp {
        op: lexer::TokenKind,
    },
    // [ctrl, lhs, rhs]
    BinOp {
        op: lexer::TokenKind,
    },
    // [ctrl]
    Global {
        global: ty::Global,
    },
    // [ctrl, ...args]
    Call {
        func: ty::Func,
    },
    // [ctrl]
    Idk,
    // [ctrl]
    Stck,
    // [ctrl, memory]
    Load,
    // [ctrl, value, memory]
    Stre,
}

impl Kind {
    fn is_pinned(&self) -> bool {
        self.is_cfg() || matches!(self, Self::Phi | Self::Mem | Self::Arg)
    }

    fn is_cfg(&self) -> bool {
        matches!(
            self,
            Self::Start
                | Self::End
                | Self::Return
                | Self::Entry
                | Self::Then
                | Self::Else
                | Self::Call { .. }
                | Self::If
                | Self::Region
                | Self::Loop
        )
    }

    fn ends_basic_block(&self) -> bool {
        matches!(self, Self::Return | Self::If | Self::End)
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Kind::CInt { value } => write!(f, "#{value}"),
            Kind::Entry => write!(f, "ctrl[entry]"),
            Kind::Then => write!(f, "ctrl[then]"),
            Kind::Else => write!(f, "ctrl[else]"),
            Kind::BinOp { op } => write!(f, "{op}"),
            Kind::Call { func, .. } => write!(f, "call {func}"),
            slf => write!(f, "{slf:?}"),
        }
    }
}

#[derive(Debug, Default, Clone)]
//#[repr(align(64))]
pub struct Node {
    kind: Kind,
    inputs: Vc,
    outputs: Vc,
    ty: ty::Id,
    offset: Offset,
    ralloc_backref: RallocBRef,
    depth: IDomDepth,
    lock_rc: LockRc,
    loop_depth: LoopDepth,
}

impl Node {
    fn is_dangling(&self) -> bool {
        self.outputs.len() + self.lock_rc as usize == 0
    }

    fn key(&self) -> (Kind, &[Nid], ty::Id) {
        (self.kind, &self.inputs, self.ty)
    }

    fn is_lazy_phi(&self) -> bool {
        self.kind == Kind::Phi && self.inputs[2] == 0
    }

    fn is_not_gvnd(&self) -> bool {
        self.is_lazy_phi() || matches!(self.kind, Kind::Arg | Kind::Stck)
    }
}

type RallocBRef = u16;
type LoopDepth = u16;
type LockRc = u16;
type IDomDepth = u16;

struct Loop {
    node: Nid,
    ctrl: [Nid; 2],
    ctrl_scope: [Scope; 2],
    scope: Scope,
}

#[derive(Clone, Copy)]
struct Variable {
    id: Ident,
    ty: ty::Id,
    ptr: bool,
    value: Nid,
}

#[derive(Default, Clone)]
struct Scope {
    vars: Vec<Variable>,
    loads: Vec<Nid>,
    store: Nid,
}

impl Scope {
    pub fn iter_elems_mut(&mut self) -> impl Iterator<Item = (ty::Id, &mut Nid)> {
        self.vars
            .iter_mut()
            .map(|v| (v.ty, &mut v.value))
            .chain(core::iter::once((ty::Id::VOID, &mut self.store)))
    }
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    ret: Option<ty::Id>,
    task_base: usize,
    nodes: Nodes,
    ctrl: Nid,
    call_count: u16,
    loops: Vec<Loop>,
    scope: Scope,
    ret_relocs: Vec<Reloc>,
    relocs: Vec<TypedReloc>,
    jump_relocs: Vec<(Nid, Reloc)>,
    code: Vec<u8>,
}

impl ItemCtx {
    fn init(&mut self, file: FileId, ret: Option<ty::Id>, task_base: usize) {
        debug_assert_eq!(self.loops.len(), 0);
        debug_assert_eq!(self.scope.vars.len(), 0);
        debug_assert_eq!(self.ret_relocs.len(), 0);
        debug_assert_eq!(self.relocs.len(), 0);
        debug_assert_eq!(self.jump_relocs.len(), 0);
        debug_assert_eq!(self.code.len(), 0);

        self.file = file;
        self.ret = ret;
        self.task_base = task_base;

        self.call_count = 0;
        self.nodes.clear();
        self.scope.vars.clear();

        let start = self.nodes.new_node(ty::Id::VOID, Kind::Start, []);
        debug_assert_eq!(start, VOID);
        let end = self.nodes.new_node(ty::Id::NEVER, Kind::End, []);
        debug_assert_eq!(end, NEVER);
        self.nodes.lock(end);
        self.ctrl = self.nodes.new_node(ty::Id::VOID, Kind::Entry, [VOID]);
        debug_assert_eq!(self.ctrl, ENTRY);
        let mem = self.nodes.new_node(ty::Id::VOID, Kind::Mem, [VOID]);
        debug_assert_eq!(mem, MEM);
        self.nodes.lock(mem);
        self.nodes.lock(self.ctrl);
        self.scope.store = self.ctrl;
    }

    fn finalize(&mut self) {
        self.nodes.unlock(NEVER);
        self.nodes.unlock_remove_scope(&core::mem::take(&mut self.scope));
        self.nodes.unlock(MEM);
    }

    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        crate::emit(&mut self.code, instr);
    }

    fn emit_body_code(&mut self, sig: Sig, tys: &Types) -> usize {
        let mut nodes = core::mem::take(&mut self.nodes);
        nodes.visited.clear(nodes.values.len());

        let fuc = Function::new(&mut nodes, tys, sig);
        let mut ralloc = Regalloc::default(); // TODO: reuse
        log::info!("{:?}", fuc);
        if self.call_count != 0 {
            core::mem::swap(
                &mut ralloc.env.preferred_regs_by_class,
                &mut ralloc.env.non_preferred_regs_by_class,
            );
        };

        let options = regalloc2::RegallocOptions {
            verbose_log: false,
            validate_ssa: false,
            algorithm: regalloc2::Algorithm::Ion,
        };
        regalloc2::run_with_ctx(&fuc, &ralloc.env, &options, &mut ralloc.ctx)
            .unwrap_or_else(|err| panic!("{err}"));

        if self.call_count != 0 {
            core::mem::swap(
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
                match node.kind {
                    Kind::If => {
                        let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                        if let Kind::BinOp { op } = fuc.nodes[cond].kind
                            && let Some((op, swapped)) = op.cond_op(node.ty.is_signed())
                        {
                            let rel = Reloc::new(self.code.len(), 3, 2);
                            self.jump_relocs.push((node.outputs[!swapped as usize], rel));
                            let &[lhs, rhs] = allocs else { unreachable!() };
                            self.emit(op(atr(lhs), atr(rhs), 0));
                        } else {
                            todo!()
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
                        match tys.size_of(sig.ret) {
                            0..=8 => {}
                            size @ 9..=16 => {
                                self.emit(instrs::ld(reg::RET, atr(allocs[0]), 0, size as _));
                            }
                            size @ 17.. => {
                                self.emit(instrs::bmc(
                                    atr(allocs[0]),
                                    reg::RET,
                                    size.try_into().unwrap(),
                                ));
                            }
                        }

                        if i != fuc.blocks.len() - 1 {
                            let rel = Reloc::new(self.code.len(), 1, 4);
                            self.ret_relocs.push(rel);
                            self.emit(instrs::jmp(0));
                        }
                    }
                    Kind::CInt { value } => {
                        self.emit(instrs::li64(atr(allocs[0]), value as _));
                    }
                    Kind::Extend => {
                        let base = fuc.nodes[node.inputs[1]].ty;
                        let dest = node.ty;

                        match (base.is_signed(), dest.is_signed()) {
                            (true, true) => {
                                let op = [instrs::sxt8, instrs::sxt16, instrs::sxt32]
                                    [tys.size_of(base).ilog2() as usize];
                                self.emit(op(atr(allocs[0]), atr(allocs[1])))
                            }
                            _ => {
                                let mask = (1u64 << (tys.size_of(base) * 8)) - 1;
                                self.emit(instrs::andi(atr(allocs[0]), atr(allocs[1]), mask));
                            }
                        }
                    }
                    Kind::UnOp { op } => {
                        let op = op.unop().expect("TODO: unary operator not supported");
                        let &[dst, oper] = allocs else { unreachable!() };
                        self.emit(op(atr(dst), atr(oper)));
                    }
                    Kind::BinOp { op } => {
                        let &[.., rhs] = node.inputs.as_slice() else { unreachable!() };

                        if let Kind::CInt { value } = fuc.nodes[rhs].kind
                            && fuc.nodes[rhs].lock_rc != 0
                            && let Some(op) =
                                op.imm_binop(node.ty.is_signed(), fuc.tys.size_of(node.ty))
                        {
                            let &[dst, lhs] = allocs else { unreachable!() };
                            self.emit(op(atr(dst), atr(lhs), value as _));
                        } else if let Some(op) =
                            op.binop(node.ty.is_signed(), fuc.tys.size_of(node.ty))
                        {
                            let &[dst, lhs, rhs] = allocs else { unreachable!() };
                            self.emit(op(atr(dst), atr(lhs), atr(rhs)));
                        } else if op.cond_op(node.ty.is_signed()).is_some() {
                        } else {
                            todo!()
                        }
                    }
                    Kind::Call { func } => {
                        self.relocs.push(TypedReloc {
                            target: ty::Kind::Func(func).compress(),
                            reloc: Reloc::new(self.code.len(), 3, 4),
                        });
                        self.emit(instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
                        if let size @ 9..=16 = tys.size_of(node.ty) {
                            let stck = fuc.nodes[*node.inputs.last().unwrap()].offset;
                            self.emit(instrs::st(reg::RET, reg::STACK_PTR, stck as _, size as _));
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
                    Kind::Idk => {}
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
                        if size <= 8 {
                            let (base, offset) = match fuc.nodes[region].kind {
                                Kind::Stck => (reg::STACK_PTR, fuc.nodes[region].offset + offset),
                                _ => (atr(allocs[1]), offset),
                            };
                            self.emit(instrs::ld(atr(allocs[0]), base, offset as _, size as _));
                        }
                    }
                    Kind::Stre if node.inputs[2] == VOID => {}
                    Kind::Stre => {
                        let mut region = node.inputs[2];
                        let mut offset = 0;
                        let size = u16::try_from(tys.size_of(node.ty)).expect("TODO");
                        if fuc.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                            && let Kind::CInt { value } =
                                fuc.nodes[fuc.nodes[region].inputs[2]].kind
                            && size <= 8
                        {
                            region = fuc.nodes[region].inputs[1];
                            offset = value as Offset;
                        }
                        let nd = &fuc.nodes[region];
                        let (base, offset, src) = match nd.kind {
                            Kind::Stck if size <= 8 => {
                                (reg::STACK_PTR, nd.offset + offset, allocs[0])
                            }
                            _ => (atr(allocs[0]), offset, allocs[1]),
                        };
                        if size > 8 {
                            self.emit(instrs::bmc(atr(src), base, size));
                        } else {
                            self.emit(instrs::st(atr(src), base, offset as _, size));
                        }
                    }
                    Kind::Start
                    | Kind::Entry
                    | Kind::Mem
                    | Kind::End
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

    fn emit_body(&mut self, tys: &mut Types, files: &[parser::Ast], sig: Sig) {
        self.nodes.graphviz(tys, files);
        self.nodes.gcm();
        self.nodes.check_final_integrity();
        self.nodes.basic_blocks();
        self.nodes.graphviz(tys, files);

        '_open_function: {
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
            self.emit(instrs::st(reg::RET_ADDR, reg::STACK_PTR, 0, 0));
        }

        let mut stack_size = 0;
        '_compute_stack: {
            let mems = core::mem::take(&mut self.nodes[MEM].outputs);
            for &stck in mems.iter() {
                stack_size += tys.size_of(self.nodes[stck].ty);
                self.nodes[stck].offset = stack_size;
            }
            for &stck in mems.iter() {
                self.nodes[stck].offset = stack_size - self.nodes[stck].offset;
            }
            self.nodes[MEM].outputs = mems;
        }

        let saved = self.emit_body_code(sig, tys);

        if let Some(last_ret) = self.ret_relocs.last()
            && last_ret.offset as usize == self.code.len() - 5
        {
            self.code.truncate(self.code.len() - 5);
            self.ret_relocs.pop();
        }

        // FIXME: maybe do this incrementally
        for (nd, rel) in self.jump_relocs.drain(..) {
            let offset = self.nodes[nd].offset;
            rel.apply_jump(&mut self.code, offset, 0);
        }

        let end = self.code.len();
        for ret_rel in self.ret_relocs.drain(..) {
            ret_rel.apply_jump(&mut self.code, end as _, 0);
        }

        let mut stripped_prelude_size = 0;
        '_close_function: {
            let pushed = (saved as i64 + (core::mem::take(&mut self.call_count) != 0) as i64) * 8;
            let stack = stack_size as i64;

            match (pushed, stack) {
                (0, 0) => {
                    stripped_prelude_size = instrs::addi64(0, 0, 0).0 + instrs::st(0, 0, 0, 0).0;
                    self.code.drain(0..stripped_prelude_size);
                    break '_close_function;
                }
                (0, stack) => {
                    write_reloc(&mut self.code, 3, -stack, 8);
                    stripped_prelude_size = instrs::addi64(0, 0, 0).0;
                    let end = stripped_prelude_size + instrs::st(0, 0, 0, 0).0;
                    self.code.drain(stripped_prelude_size..end);
                    self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, stack as _));
                    break '_close_function;
                }
                _ => {}
            }

            write_reloc(&mut self.code, 3, -(pushed + stack), 8);
            write_reloc(&mut self.code, 3 + 8 + 3, stack, 8);
            write_reloc(&mut self.code, 3 + 8 + 3 + 8, pushed, 2);

            self.emit(instrs::ld(reg::RET_ADDR, reg::STACK_PTR, stack as _, pushed as _));
            self.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, (pushed + stack) as _));
        }
        self.relocs.iter_mut().for_each(|r| r.reloc.offset -= stripped_prelude_size as u32);
        self.emit(instrs::jala(reg::ZERO, reg::RET_ADDR, 0));
    }
}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
}

struct FTask {
    file: FileId,
    id: ty::Func,
}

#[derive(Default, Debug)]
struct Ctx {
    ty: Option<ty::Id>,
}

impl Ctx {
    pub fn with_ty(self, ty: impl Into<ty::Id>) -> Self {
        Self { ty: Some(ty.into()) }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
    used_cis: usize,
}

impl Pool {
    pub fn push_ci(
        &mut self,
        file: FileId,
        ret: Option<ty::Id>,
        task_base: usize,
        target: &mut ItemCtx,
    ) {
        if let Some(slot) = self.cis.get_mut(self.used_cis) {
            core::mem::swap(slot, target);
        } else {
            self.cis.push(ItemCtx::default());
            core::mem::swap(self.cis.last_mut().unwrap(), target);
        }
        target.init(file, ret, task_base);
        self.used_cis += 1;
    }

    pub fn pop_ci(&mut self, target: &mut ItemCtx) {
        self.used_cis -= 1;
        core::mem::swap(&mut self.cis[self.used_cis], target);
    }
}

struct Regalloc {
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

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Value {
    ty: ty::Id,
    var: bool,
    ptr: bool,
    id: Nid,
}

impl Value {
    const NEVER: Option<Value> =
        Some(Self { ty: ty::Id::NEVER, var: false, ptr: false, id: NEVER });
    const VOID: Value = Self { ty: ty::Id::VOID, var: false, ptr: false, id: VOID };

    pub fn new(id: Nid) -> Self {
        Self { id, ..Default::default() }
    }

    pub fn var(id: usize) -> Self {
        Self { id: u16::MAX - (id as Nid), var: true, ..Default::default() }
    }

    pub fn ptr(id: Nid) -> Self {
        Self { id, ptr: true, ..Default::default() }
    }

    #[inline(always)]
    pub fn ty(self, ty: impl Into<ty::Id>) -> Self {
        Self { ty: ty.into(), ..self }
    }
}

#[derive(Default)]
pub struct Codegen<'a> {
    pub files: &'a [parser::Ast],

    tasks: Vec<Option<FTask>>,
    tys: Types,
    ci: ItemCtx,
    pool: Pool,
    #[expect(dead_code)]
    ralloc: Regalloc,
    ct: Comptime,
    errors: RefCell<String>,
}

impl TypeParser for Codegen<'_> {
    fn tys(&mut self) -> &mut Types {
        &mut self.tys
    }

    #[expect(unused)]
    fn eval_const(&mut self, file: FileId, expr: &Expr, ty: ty::Id) -> u64 {
        todo!()
    }

    #[expect(unused)]
    fn infer_type(&mut self, expr: &Expr) -> ty::Id {
        todo!()
    }

    fn on_reuse(&mut self, existing: ty::Id) {
        if let ty::Kind::Func(id) = existing.expand()
            && let func = &mut self.tys.ins.funcs[id as usize]
            && let Err(idx) = task::unpack(func.offset)
            && idx < self.tasks.len()
        {
            func.offset = task::id(self.tasks.len());
            let task = self.tasks[idx].take();
            self.tasks.push(task);
        }
    }

    fn eval_global(&mut self, file: FileId, name: Ident, expr: &Expr) -> ty::Id {
        let gid = self.tys.ins.globals.len() as ty::Global;
        self.tys.ins.globals.push(Global { file, name, ..Default::default() });

        let ty = ty::Kind::Global(gid);
        self.pool.push_ci(file, None, self.tasks.len(), &mut self.ci);

        let ret = Expr::Return { pos: expr.pos(), val: Some(expr) };
        self.expr(&ret);

        self.ci.finalize();

        let ret = self.ci.ret.expect("for return type to be infered");
        if self.errors.borrow().is_empty() {
            self.ci.emit_body(&mut self.tys, self.files, Sig { args: Tuple::empty(), ret });
            self.ci.code.truncate(self.ci.code.len() - instrs::jala(0, 0, 0).0);
            self.ci.emit(instrs::tx());

            let func = Func {
                file,
                name,
                expr: ExprRef::new(expr),
                relocs: core::mem::take(&mut self.ci.relocs),
                code: core::mem::take(&mut self.ci.code),
                ..Default::default()
            };
            self.pool.pop_ci(&mut self.ci);
            self.complete_call_graph();

            let mut mem = vec![0u8; self.tys.size_of(ret) as usize];

            // TODO: return them back
            let fuc = self.tys.ins.funcs.len() as ty::Func;
            self.tys.ins.funcs.push(func);

            self.tys.dump_reachable(fuc, &mut self.ct.code);

            #[cfg(debug_assertions)]
            {
                let mut vc = String::new();
                if let Err(e) = self.tys.disasm(&self.ct.code, self.files, &mut vc, |_| {}) {
                    panic!("{e} {}", vc);
                } else {
                    log::trace!("{}", vc);
                }
            }

            self.ct.vm.write_reg(reg::RET, mem.as_mut_ptr() as u64);
            let prev_pc = self.ct.push_pc(self.tys.ins.funcs[fuc as usize].offset);
            loop {
                match self.ct.vm.run().expect("TODO") {
                    hbvm::VmRunOk::End => break,
                    hbvm::VmRunOk::Timer => todo!(),
                    hbvm::VmRunOk::Ecall => todo!(),
                    hbvm::VmRunOk::Breakpoint => todo!(),
                }
            }
            self.ct.pop_pc(prev_pc);

            match mem.len() {
                0 => unreachable!(),
                len @ 1..=8 => {
                    mem.copy_from_slice(&self.ct.vm.read_reg(reg::RET).0.to_ne_bytes()[..len])
                }
                9..=16 => todo!(),
                _ => {}
            }

            self.tys.ins.globals[gid as usize].data = mem;
        } else {
            self.pool.pop_ci(&mut self.ci);
        }
        self.tys.ins.globals[gid as usize].ty = ret;

        ty.compress()
    }

    fn report(&self, pos: Pos, msg: impl Display) -> ty::Id {
        self.report(pos, msg);
        ty::Id::NEVER
    }

    fn find_local_ty(&mut self, _: Ident) -> Option<ty::Id> {
        None
    }
}

impl<'a> Codegen<'a> {
    fn store_mem(&mut self, region: Nid, value: Nid) -> Nid {
        if value == NEVER {
            return NEVER;
        }

        let mut vc = Vc::from([VOID, value, region]);
        self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
        self.ci.nodes.unlock(self.ci.scope.store);
        if let Some(str) = self.ci.scope.store.to_store() {
            vc.push(str);
        }
        for load in self.ci.scope.loads.drain(..) {
            if load == value {
                self.ci.nodes.unlock(load);
                continue;
            }
            if !self.ci.nodes.unlock_remove(load) {
                vc.push(load);
            }
        }
        let store = self.ci.nodes.new_node(self.tof(value), Kind::Stre, vc);
        self.ci.nodes.lock(store);
        self.ci.scope.store = store;
        store
    }

    fn load_mem(&mut self, region: Nid, ty: ty::Id) -> Nid {
        debug_assert_ne!(region, VOID);
        let mut vc = Vc::from([VOID, region]);
        self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
        if let Some(str) = self.ci.scope.store.to_store() {
            vc.push(str);
        }
        let load = self.ci.nodes.new_node(ty, Kind::Load, vc);
        self.ci.nodes.lock(load);
        self.ci.scope.loads.push(load);
        load
    }

    pub fn generate(&mut self) {
        self.find_type(0, 0, Err("main"), self.files);
        self.make_func_reachable(0);
        self.complete_call_graph();
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let fuc = &mut self.tys.ins.funcs[func as usize];
        if fuc.offset == u32::MAX {
            fuc.offset = task::id(self.tasks.len() as _);
            self.tasks.push(Some(FTask { file: fuc.file, id: func }));
        }
    }

    fn raw_expr(&mut self, expr: &Expr) -> Option<Value> {
        self.raw_expr_ctx(expr, Ctx::default())
    }

    fn raw_expr_ctx(&mut self, expr: &Expr, ctx: Ctx) -> Option<Value> {
        // ordered by complexity of the expression
        match *expr {
            Expr::Idk { pos } => {
                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "resulting value cannot be inferred from context, \
                        consider using `@as(<ty>, idk)` to hint the type",
                    );
                    return Value::NEVER;
                };

                if matches!(ty.expand(), ty::Kind::Struct(_) | ty::Kind::Slice(_)) {
                    let stck = self.ci.nodes.new_node(ty, Kind::Stck, [VOID, MEM]);
                    Some(Value::ptr(stck).ty(ty))
                } else {
                    Some(self.ci.nodes.new_node_lit(ty, Kind::Idk, [VOID]))
                }
            }
            Expr::Number { value, .. } => Some(self.ci.nodes.new_node_lit(
                ctx.ty.filter(|ty| ty.is_integer() || ty.is_pointer()).unwrap_or(ty::Id::INT),
                Kind::CInt { value },
                [VOID],
            )),
            Expr::Ident { id, .. }
                if let Some(index) = self.ci.scope.vars.iter().rposition(|v| v.id == id) =>
            {
                let var = &mut self.ci.scope.vars[index];
                self.ci.nodes.load_loop_var(index, &mut var.value, &mut self.ci.loops);

                Some(Value::var(index).ty(var.ty))
            }
            Expr::Ident { id, pos, .. } => {
                let decl = self.find_type(pos, self.ci.file, Ok(id), self.files);
                match decl.expand() {
                    ty::Kind::Builtin(ty::NEVER) => Value::NEVER,
                    ty::Kind::Global(global) => {
                        let gl = &self.tys.ins.globals[global as usize];
                        let value = self.ci.nodes.new_node(gl.ty, Kind::Global { global }, [VOID]);
                        Some(Value::ptr(value).ty(gl.ty))
                    }
                    _ => self.report_unhandled_ast(
                        expr,
                        format_args!("identifier evaluated to '{}'", self.ty_display(decl)),
                    ),
                }
            }
            Expr::Comment { .. } => Some(Value::VOID),
            Expr::String { pos, literal } => {
                let literal = &literal[1..literal.len() - 1];

                let report = |bytes: &core::str::Bytes, message: &str| {
                    self.report(pos + (literal.len() - bytes.len()) as u32 - 1, message)
                };

                let mut data = Vec::<u8>::with_capacity(literal.len());
                crate::endoce_string(literal, &mut data, report).unwrap();

                let global = self.tys.ins.globals.len() as ty::Global;
                let ty = self.tys.make_ptr(ty::Id::U8);
                self.tys.ins.globals.push(Global { data, ty, ..Default::default() });
                let global = self.ci.nodes.new_node(ty, Kind::Global { global }, [VOID]);
                Some(Value::new(global).ty(ty))
            }
            Expr::Return { pos, val } => {
                let mut value = if let Some(val) = val {
                    self.expr_ctx(val, Ctx { ty: self.ci.ret })?
                } else {
                    Value { ty: ty::Id::VOID, ..Default::default() }
                };

                let expected = *self.ci.ret.get_or_insert(value.ty);
                self.assert_ty(pos, &mut value, expected, "return value");

                let mut inps = Vc::from([self.ci.ctrl, value.id]);
                self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
                if let Some(str) = self.ci.scope.store.to_store() {
                    inps.push(str);
                }

                self.ci.ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::Return, inps);

                self.ci.nodes[NEVER].inputs.push(self.ci.ctrl);
                self.ci.nodes[self.ci.ctrl].outputs.push(NEVER);

                None
            }
            Expr::Field { target, name, pos } => {
                let mut vtarget = self.raw_expr(target)?;
                self.strip_var(&mut vtarget);
                let tty = vtarget.ty;

                let ty::Kind::Struct(s) = self.tys.base_of(tty).unwrap_or(tty).expand() else {
                    self.report(
                        pos,
                        fa!(
                            "the '{}' is not a struct, or pointer to one, \
                            but accessing fields is only possible on structs",
                            self.ty_display(tty)
                        ),
                    );
                    return Value::NEVER;
                };

                let Some((offset, ty)) = OffsetIter::offset_of(&self.tys, s, name) else {
                    let field_list = self
                        .tys
                        .struct_fields(s)
                        .iter()
                        .map(|f| self.tys.names.ident_str(f.name))
                        .intersperse("', '")
                        .collect::<String>();
                    self.report(
                        pos,
                        fa!(
                            "the '{}' does not have this field, \
                            but it does have '{field_list}'",
                            self.ty_display(tty)
                        ),
                    );
                    return Value::NEVER;
                };

                Some(Value::ptr(self.offset(vtarget.id, offset)).ty(ty))
            }
            Expr::UnOp { op: TokenKind::Band, val, .. } => {
                let ctx = Ctx { ty: ctx.ty.and_then(|ty| self.tys.base_of(ty)) };

                let mut val = self.raw_expr_ctx(val, ctx)?;
                self.strip_var(&mut val);

                if val.ptr {
                    val.ptr = false;
                    val.ty = self.tys.make_ptr(val.ty);
                    return Some(val);
                }

                let stack = self.ci.nodes.new_node_nop(val.ty, Kind::Stck, [VOID, MEM]);
                self.store_mem(stack, val.id);

                Some(Value::new(stack).ty(self.tys.make_ptr(val.ty)))
            }
            Expr::UnOp { op: TokenKind::Mul, val, pos } => {
                let ctx = Ctx { ty: ctx.ty.map(|ty| self.tys.make_ptr(ty)) };
                let mut val = self.expr_ctx(val, ctx)?;

                let Some(base) = self.tys.base_of(val.ty) else {
                    self.report(
                        pos,
                        fa!("the '{}' can not be dereferneced", self.ty_display(val.ty)),
                    );
                    return Value::NEVER;
                };
                val.ptr = true;
                val.ty = base;
                Some(val)
            }
            Expr::UnOp { pos, op: op @ TokenKind::Sub, val } => {
                let val = self.expr_ctx(val, ctx)?;
                if !val.ty.is_integer() {
                    self.report(pos, fa!("cant negate '{}'", self.ty_display(val.ty)));
                }
                Some(self.ci.nodes.new_node_lit(val.ty, Kind::UnOp { op }, [VOID, val.id]))
            }
            Expr::BinOp { left, op: TokenKind::Decl, right } => {
                let right = self.raw_expr(right)?;
                self.assign_pattern(left, right);
                Some(Value::VOID)
            }
            Expr::BinOp { left, op: TokenKind::Assign, right } => {
                let dest = self.raw_expr(left)?;
                let mut value = self.expr_ctx(right, Ctx::default().with_ty(dest.ty))?;

                self.assert_ty(left.pos(), &mut value, dest.ty, "assignment source");

                if dest.var {
                    self.ci.nodes.lock(value.id);
                    let var = &mut self.ci.scope.vars[(u16::MAX - dest.id) as usize];
                    let prev = core::mem::replace(&mut var.value, value.id);
                    self.ci.nodes.unlock_remove(prev);
                } else if dest.ptr {
                    self.store_mem(dest.id, value.id);
                } else {
                    self.report(left.pos(), "cannot assign to this expression");
                }

                Some(Value::VOID)
            }
            Expr::BinOp { left, op, right }
                if !matches!(op, TokenKind::Assign | TokenKind::Decl) =>
            {
                let mut lhs = self.expr_ctx(left, ctx)?;
                self.ci.nodes.lock(lhs.id);
                let rhs = self.expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                self.ci.nodes.unlock(lhs.id);
                let mut rhs = rhs?;
                let ty = self.binop_ty(left.pos(), &mut rhs, &mut lhs, op);
                let inps = [VOID, lhs.id, rhs.id];
                Some(self.ci.nodes.new_node_lit(ty::bin_ret(ty, op), Kind::BinOp { op }, inps))
            }
            Expr::Index { base, index } => {
                let mut bs = self.raw_expr(base)?;
                self.strip_var(&mut bs);

                if let Some(base) = self.tys.base_of(bs.ty) {
                    bs.ptr = true;
                    bs.ty = base;
                }

                let ty::Kind::Slice(s) = bs.ty.expand() else {
                    self.report(
                        base.pos(),
                        fa!(
                            "cant index into '{}' which is not array nor slice",
                            self.ty_display(bs.ty)
                        ),
                    );
                    return Value::NEVER;
                };

                let elem = self.tys.ins.slices[s as usize].elem;
                let idx = self.expr_ctx(index, Ctx::default().with_ty(ty::Id::INT))?;
                let value = self.tys.size_of(elem) as i64;
                let size = self.ci.nodes.new_node_nop(ty::Id::INT, Kind::CInt { value }, [VOID]);
                let inps = [VOID, idx.id, size];
                let offset =
                    self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Mul }, inps);
                let inps = [VOID, bs.id, offset];
                let ptr =
                    self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Add }, inps);
                Some(Value::ptr(ptr).ty(elem))
            }
            Expr::Directive { name: "sizeof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(self.ci.nodes.new_node_lit(
                    ty::Id::INT,
                    Kind::CInt { value: self.tys.size_of(ty) as _ },
                    [VOID],
                ))
            }
            Expr::Directive { name: "trunc", args: [expr], pos } => {
                let val = self.expr(expr)?;

                if !val.ty.is_integer() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "only integers can be truncated ('{}' is not an integer)",
                            self.ty_display(val.ty)
                        ),
                    );
                    return Value::NEVER;
                }

                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "resulting integer cannot be inferred from context, \
                        consider using `@as(<int_ty>, @trunc(<expr>))` to hint the type",
                    );
                    return Value::NEVER;
                };

                if self.tys.size_of(val.ty) <= self.tys.size_of(ty) {
                    self.report(
                        pos,
                        fa!(
                            "truncating '{}' into '{}' has no effect",
                            self.ty_display(val.ty),
                            self.ty_display(ty)
                        ),
                    );
                }

                let value = (1i64 << self.tys.size_of(val.ty)) - 1;
                let mask = self.ci.nodes.new_node_nop(val.ty, Kind::CInt { value }, [VOID]);
                let inps = [VOID, val.id, mask];
                Some(self.ci.nodes.new_node_lit(ty, Kind::BinOp { op: TokenKind::Band }, inps))
            }
            Expr::Directive { name: "as", args: [ty, expr], .. } => {
                let ctx = Ctx::default().with_ty(self.ty(ty));
                self.raw_expr_ctx(expr, ctx)
            }
            Expr::Call { func, args, .. } => {
                self.ci.call_count += 1;
                let ty = self.ty(func);
                let ty::Kind::Func(fu) = ty.expand() else {
                    self.report(
                        func.pos(),
                        fa!("compiler cant (yet) call '{}'", self.ty_display(ty)),
                    );
                    return Value::NEVER;
                };

                self.make_func_reachable(fu);

                let fuc = &self.tys.ins.funcs[fu as usize];
                let sig = fuc.sig.expect("TODO: generic functions");
                let ast = &self.files[fuc.file as usize];
                let &Expr::Closure { args: cargs, .. } = fuc.expr.get(ast).unwrap() else {
                    unreachable!()
                };

                self.assert_report(
                    args.len() == cargs.len(),
                    func.pos(),
                    fa!(
                        "expected {} function argumenr{}, got {}",
                        cargs.len(),
                        if cargs.len() == 1 { "" } else { "s" },
                        args.len()
                    ),
                );

                let mut inps = Vc::from([self.ci.ctrl]);
                for ((arg, carg), tyx) in args.iter().zip(cargs).zip(sig.args.range()) {
                    let ty = self.tys.ins.args[tyx];
                    if self.tys.size_of(ty) == 0 {
                        continue;
                    }
                    let mut value = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                    debug_assert_ne!(value.id, 0);
                    self.assert_ty(arg.pos(), &mut value, ty, fa!("argument {}", carg.name));

                    inps.push(value.id);
                }

                if let Some(str) = self.ci.scope.store.to_store() {
                    inps.push(str);
                }
                self.ci.scope.loads.retain(|&load| {
                    if inps.contains(&load) {
                        return true;
                    }

                    if !self.ci.nodes.unlock_remove(load) {
                        inps.push(load);
                    }

                    false
                });

                let alt_value = match self.tys.size_of(sig.ret) {
                    0..=8 => None,
                    9.. => {
                        let stck = self.ci.nodes.new_node_nop(sig.ret, Kind::Stck, [VOID, MEM]);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(sig.ret))
                    }
                };

                self.ci.ctrl = self.ci.nodes.new_node(sig.ret, Kind::Call { func: fu }, inps);

                self.store_mem(VOID, VOID);

                alt_value.or(Some(Value::new(self.ci.ctrl).ty(sig.ret)))
            }
            Expr::Tupl { pos, ty, fields, .. } => {
                let Some(sty) = ty.map(|ty| self.ty(ty)).or(ctx.ty) else {
                    self.report(
                        pos,
                        "the type of struct cannot be inferred from context, \
                        use an explicit type instead: <type>.{ ... }",
                    );
                    return Value::NEVER;
                };

                match sty.expand() {
                    ty::Kind::Struct(s) => {
                        let mem = self.ci.nodes.new_node(sty, Kind::Stck, [VOID, MEM]);
                        let mut offs = OffsetIter::new(s, &self.tys);
                        for field in fields {
                            let Some((ty, offset)) = offs.next_ty(&self.tys) else {
                                self.report(
                                    field.pos(),
                                    "this init argumen overflows the field count",
                                );
                                break;
                            };

                            let value = self.expr_ctx(field, Ctx::default().with_ty(ty))?;
                            let mem = self.offset(mem, offset);
                            self.store_mem(mem, value.id);
                        }

                        let field_list = offs
                            .into_iter(&self.tys)
                            .map(|(f, ..)| self.tys.names.ident_str(f.name))
                            .intersperse(", ")
                            .collect::<String>();

                        if !field_list.is_empty() {
                            self.report(
                                pos,
                                fa!("the struct initializer is missing {field_list} \
                                    (append them to the end of the constructor)"),
                            );
                        }
                        Some(Value::ptr(mem).ty(sty))
                    }
                    ty::Kind::Slice(s) => {
                        let slice = &self.tys.ins.slices[s as usize];
                        let len = slice.len().unwrap_or(fields.len());
                        let elem = slice.elem;
                        let elem_size = self.tys.size_of(elem);
                        let aty = slice
                            .len()
                            .map_or_else(|| self.tys.make_array(elem, len as ArrayLen), |_| sty);

                        if len != fields.len() {
                            self.report(
                                pos,
                                fa!(
                                    "expected '{}' but constructor has {} elements",
                                    self.ty_display(aty),
                                    fields.len()
                                ),
                            );
                            return Value::NEVER;
                        }

                        let mem = self.ci.nodes.new_node(aty, Kind::Stck, [VOID, MEM]);

                        for (field, offset) in
                            fields.iter().zip((0u32..).step_by(elem_size as usize))
                        {
                            let mut value = self.expr_ctx(field, Ctx::default().with_ty(elem))?;
                            _ = self.assert_ty(field.pos(), &mut value, elem, "array value");
                            let mem = self.offset(mem, offset);
                            self.store_mem(mem, value.id);
                        }

                        Some(Value::ptr(mem).ty(aty))
                    }
                    _ => {
                        let inferred = if ty.is_some() { "" } else { "inferred " };
                        self.report(
                            pos,
                            fa!(
                                "the {inferred}type of the constructor is `{}`, \
                                but thats not a struct nor slice or array",
                                self.ty_display(sty)
                            ),
                        );
                        Value::NEVER
                    }
                }
            }
            Expr::Ctor { pos, ty, fields, .. } => {
                let Some(sty) = ty.map(|ty| self.ty(ty)).or(ctx.ty) else {
                    self.report(
                        pos,
                        "the type of struct cannot be inferred from context, \
                        use an explicit type instead: <type>.{ ... }",
                    );
                    return Value::NEVER;
                };

                let ty::Kind::Struct(s) = sty.expand() else {
                    let inferred = if ty.is_some() { "" } else { "inferred " };
                    self.report(
                        pos,
                        fa!(
                            "the {inferred}type of the constructor is `{}`, \
                            but thats not a struct",
                            self.ty_display(sty)
                        ),
                    );
                    return Value::NEVER;
                };

                // TODO: dont allocate
                let mut offs = OffsetIter::new(s, &self.tys)
                    .into_iter(&self.tys)
                    .map(|(f, o)| (f.ty, o))
                    .collect::<Vec<_>>();
                let mem = self.ci.nodes.new_node(sty, Kind::Stck, [VOID, MEM]);
                for field in fields {
                    let Some(index) = self.tys.find_struct_field(s, field.name) else {
                        self.report(
                            field.pos,
                            fa!("struct '{}' does not have this field", self.ty_display(sty)),
                        );
                        continue;
                    };

                    let (ty, offset) =
                        core::mem::replace(&mut offs[index], (ty::Id::UNDECLARED, field.pos));

                    if ty == ty::Id::UNDECLARED {
                        self.report(field.pos, "the struct field is already initialized");
                        self.report(offset, "previous initialization is here");
                        continue;
                    }

                    let value = self.expr_ctx(&field.value, Ctx::default().with_ty(ty))?;
                    let mem = self.offset(mem, offset);
                    self.store_mem(mem, value.id);
                }

                let field_list = self
                    .tys
                    .struct_fields(s)
                    .iter()
                    .zip(offs)
                    .filter(|&(_, (ty, _))| ty != ty::Id::UNDECLARED)
                    .map(|(f, _)| self.tys.names.ident_str(f.name))
                    .intersperse(", ")
                    .collect::<String>();

                if !field_list.is_empty() {
                    self.report(pos, fa!("the struct initializer is missing {field_list}"));
                }

                Some(Value::ptr(mem).ty(sty))
            }
            Expr::Block { stmts, .. } => {
                let base = self.ci.scope.vars.len();

                let mut ret = Some(Value::VOID);
                for stmt in stmts {
                    ret = ret.and(self.expr(stmt));
                    if let Some(mut id) = ret {
                        self.assert_ty(stmt.pos(), &mut id, ty::Id::VOID, "statement");
                    } else {
                        break;
                    }
                }

                self.ci.nodes.lock(self.ci.ctrl);
                for var in self.ci.scope.vars.drain(base..) {
                    self.ci.nodes.unlock_remove(var.value);
                }
                self.ci.nodes.unlock(self.ci.ctrl);

                ret
            }
            Expr::Loop { body, .. } => {
                self.ci.ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::Loop, [self.ci.ctrl; 2]);
                self.ci.loops.push(Loop {
                    node: self.ci.ctrl,
                    ctrl: [Nid::MAX; 2],
                    ctrl_scope: core::array::from_fn(|_| Default::default()),
                    scope: self.ci.scope.clone(),
                });

                for (_, var) in &mut self.ci.scope.iter_elems_mut() {
                    *var = VOID;
                }
                self.ci.nodes.lock_scope(&self.ci.scope);

                self.expr(body);

                let Loop {
                    node,
                    ctrl: [mut con, bre],
                    ctrl_scope: [mut cons, mut bres],
                    mut scope,
                } = self.ci.loops.pop().unwrap();

                if con != Nid::MAX {
                    con = self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [con, self.ci.ctrl]);
                    Self::merge_scopes(
                        &mut self.ci.nodes,
                        &mut self.ci.loops,
                        con,
                        &mut self.ci.scope,
                        &mut cons,
                        true,
                    );
                    self.ci.ctrl = con;
                }

                self.ci.nodes.modify_input(node, 1, self.ci.ctrl);

                let idx = self.ci.nodes[node]
                    .outputs
                    .iter()
                    .position(|&n| self.ci.nodes.is_cfg(n))
                    .unwrap();
                self.ci.nodes[node].outputs.swap(idx, 0);

                if bre == Nid::MAX {
                    self.ci.ctrl = NEVER;
                    return None;
                }
                self.ci.ctrl = bre;

                self.ci.nodes.lock(self.ci.ctrl);

                core::mem::swap(&mut self.ci.scope, &mut bres);

                for (((_, dest_value), (_, &mut mut scope_value)), (_, &mut loop_value)) in self
                    .ci
                    .scope
                    .iter_elems_mut()
                    .zip(scope.iter_elems_mut())
                    .zip(bres.iter_elems_mut())
                {
                    self.ci.nodes.unlock(loop_value);

                    if loop_value != VOID {
                        self.ci.nodes.unlock(scope_value);
                        if loop_value != scope_value {
                            scope_value = self.ci.nodes.modify_input(scope_value, 2, loop_value);
                            self.ci.nodes.lock(scope_value);
                        } else {
                            if *dest_value == scope_value {
                                self.ci.nodes.unlock(*dest_value);
                                *dest_value = VOID;
                                self.ci.nodes.lock(*dest_value);
                            }
                            let phi = &self.ci.nodes[scope_value];
                            debug_assert_eq!(phi.kind, Kind::Phi);
                            debug_assert_eq!(phi.inputs[2], VOID);
                            let prev = phi.inputs[1];
                            self.ci.nodes.replace(scope_value, prev);
                            scope_value = prev;
                            self.ci.nodes.lock(prev);
                        }
                    }

                    if *dest_value == VOID {
                        self.ci.nodes.unlock(*dest_value);
                        *dest_value = scope_value;
                        self.ci.nodes.lock(*dest_value);
                    }

                    debug_assert!(
                        self.ci.nodes[*dest_value].kind != Kind::Phi
                            || self.ci.nodes[*dest_value].inputs[2] != 0
                    );

                    self.ci.nodes.unlock_remove(scope_value);
                }

                scope.loads.iter().for_each(|&n| _ = self.ci.nodes.unlock_remove(n));
                bres.loads.iter().for_each(|&n| _ = self.ci.nodes.unlock_remove(n));

                self.ci.nodes.unlock(self.ci.ctrl);

                Some(Value::VOID)
            }
            Expr::Break { pos } => self.jump_to(pos, 1),
            Expr::Continue { pos } => self.jump_to(pos, 0),
            Expr::If { cond, then, else_, .. } => {
                let cond = self.expr_ctx(cond, Ctx::default().with_ty(ty::BOOL))?;

                let if_node =
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::If, [self.ci.ctrl, cond.id]);

                'b: {
                    let branch = match self.tof(if_node).expand().inner() {
                        ty::LEFT_UNREACHABLE => else_,
                        ty::RIGHT_UNREACHABLE => Some(then),
                        _ => break 'b,
                    };

                    self.ci.nodes.lock(self.ci.ctrl);
                    self.ci.nodes.remove(if_node);
                    self.ci.nodes.unlock(self.ci.ctrl);

                    if let Some(branch) = branch {
                        return self.expr(branch);
                    } else {
                        return Some(Value::VOID);
                    }
                }

                self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
                let orig_store = self.ci.scope.store;
                self.ci.nodes.lock(orig_store);
                let else_scope = self.ci.scope.clone();
                self.ci.nodes.lock_scope(&else_scope);

                self.ci.ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::Then, [if_node]);
                let lcntrl = self.expr(then).map_or(Nid::MAX, |_| self.ci.ctrl);

                let mut then_scope = core::mem::replace(&mut self.ci.scope, else_scope);
                self.ci.ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::Else, [if_node]);
                let rcntrl = if let Some(else_) = else_ {
                    self.expr(else_).map_or(Nid::MAX, |_| self.ci.ctrl)
                } else {
                    self.ci.ctrl
                };

                self.ci.nodes.unlock_remove(orig_store);

                if lcntrl == Nid::MAX && rcntrl == Nid::MAX {
                    self.ci.nodes.unlock_remove_scope(&then_scope);
                    return None;
                } else if lcntrl == Nid::MAX {
                    self.ci.nodes.unlock_remove_scope(&then_scope);
                    return Some(Value::VOID);
                } else if rcntrl == Nid::MAX {
                    self.ci.nodes.unlock_remove_scope(&self.ci.scope);
                    self.ci.scope = then_scope;
                    self.ci.ctrl = lcntrl;
                    return Some(Value::VOID);
                }

                self.ci.ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [lcntrl, rcntrl]);

                Self::merge_scopes(
                    &mut self.ci.nodes,
                    &mut self.ci.loops,
                    self.ci.ctrl,
                    &mut self.ci.scope,
                    &mut then_scope,
                    true,
                );

                Some(Value::VOID)
            }
            ref e => self.report_unhandled_ast(e, "bruh"),
        }
    }

    fn assign_pattern(&mut self, pat: &Expr, mut right: Value) {
        match *pat {
            Expr::Ident { id, .. } => {
                self.strip_var(&mut right);
                self.ci.nodes.lock(right.id);
                self.ci.scope.vars.push(Variable {
                    id,
                    value: right.id,
                    ptr: right.ptr,
                    ty: right.ty,
                });
            }
            Expr::Ctor { pos, fields, .. } => {
                let ty::Kind::Struct(idx) = right.ty.expand() else {
                    self.report(pos, "can't use struct destruct on non struct value (TODO: shold work with modules)");
                    return;
                };

                for &CtorField { pos, name, ref value } in fields {
                    let Some((offset, ty)) = OffsetIter::offset_of(&self.tys, idx, name) else {
                        self.report(pos, format_args!("field not found: {name:?}"));
                        continue;
                    };
                    let off = self.offset(right.id, offset);
                    self.assign_pattern(value, Value::ptr(off).ty(ty));
                }
            }
            ref pat => self.report_unhandled_ast(pat, "pattern"),
        }
    }

    fn expr_ctx(&mut self, expr: &Expr, ctx: Ctx) -> Option<Value> {
        let mut n = self.raw_expr_ctx(expr, ctx)?;
        self.strip_var(&mut n);
        if core::mem::take(&mut n.ptr) {
            n.id = self.load_mem(n.id, n.ty);
        }
        Some(n)
    }

    fn offset(&mut self, val: Nid, off: Offset) -> Nid {
        if off == 0 {
            return val;
        }

        let off = self.ci.nodes.new_node_nop(ty::Id::INT, Kind::CInt { value: off as i64 }, [VOID]);
        let inps = [VOID, val, off];
        self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Add }, inps)
    }

    fn strip_var(&mut self, n: &mut Value) {
        if core::mem::take(&mut n.var) {
            let id = (u16::MAX - n.id) as usize;
            self.ci.nodes.load_loop_var(id, &mut self.ci.scope.vars[id].value, &mut self.ci.loops);
            n.ptr = self.ci.scope.vars[id].ptr;
            n.id = self.ci.scope.vars[id].value;
        }
    }

    fn expr(&mut self, expr: &Expr) -> Option<Value> {
        self.expr_ctx(expr, Default::default())
    }

    fn jump_to(&mut self, pos: Pos, id: usize) -> Option<Value> {
        let Some(mut loob) = self.ci.loops.last_mut() else {
            self.report(pos, "break outside a loop");
            return None;
        };

        if loob.ctrl[id] == Nid::MAX {
            loob.ctrl[id] = self.ci.ctrl;
            loob.ctrl_scope[id] = self.ci.scope.clone();
            loob.ctrl_scope[id].vars.truncate(loob.scope.vars.len());
            self.ci.nodes.lock_scope(&loob.ctrl_scope[id]);
        } else {
            let reg =
                self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [self.ci.ctrl, loob.ctrl[id]]);
            let mut scope = core::mem::take(&mut loob.ctrl_scope[id]);

            Self::merge_scopes(
                &mut self.ci.nodes,
                &mut self.ci.loops,
                reg,
                &mut scope,
                &mut self.ci.scope,
                false,
            );

            loob = self.ci.loops.last_mut().unwrap();
            loob.ctrl_scope[id] = scope;
            loob.ctrl[id] = reg;
        }

        self.ci.ctrl = NEVER;
        None
    }

    fn merge_scopes(
        nodes: &mut Nodes,
        loops: &mut [Loop],
        ctrl: Nid,
        to: &mut Scope,
        from: &mut Scope,
        drop_from: bool,
    ) {
        for (i, ((ty, to_value), (_, from_value))) in
            to.iter_elems_mut().zip(from.iter_elems_mut()).enumerate()
        {
            if to_value != from_value {
                nodes.load_loop_var(i, from_value, loops);
                nodes.load_loop_var(i, to_value, loops);
                if to_value != from_value {
                    let inps = [ctrl, *from_value, *to_value];
                    nodes.unlock(*to_value);
                    *to_value = nodes.new_node(ty, Kind::Phi, inps);
                    nodes.lock(*to_value);
                }
            }
        }

        for load in to.loads.drain(..) {
            nodes.unlock_remove(load);
        }
        for load in from.loads.drain(..) {
            nodes.unlock_remove(load);
        }

        if drop_from {
            nodes.unlock_remove_scope(from);
        }
    }

    #[inline(always)]
    fn tof(&self, id: Nid) -> ty::Id {
        self.ci.nodes[id].ty
    }

    fn complete_call_graph(&mut self) {
        while self.ci.task_base < self.tasks.len()
            && let Some(task_slot) = self.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.emit_func(task);
        }
    }

    fn emit_func(&mut self, FTask { file, id }: FTask) {
        let func = &mut self.tys.ins.funcs[id as usize];
        debug_assert_eq!(func.file, file);
        func.offset = u32::MAX - 1;
        let sig = func.sig.expect("to emmit only concrete functions");
        let ast = &self.files[file as usize];
        let expr = func.expr.get(ast).unwrap();

        self.pool.push_ci(file, Some(sig.ret), 0, &mut self.ci);

        let &Expr::Closure { body, args, .. } = expr else {
            unreachable!("{}", self.ast_display(expr))
        };

        let mut sig_args = sig.args.range();
        for arg in args.iter() {
            let ty = self.tys.ins.args[sig_args.next().unwrap()];
            let value = self.ci.nodes.new_node_nop(ty, Kind::Arg, [VOID]);
            self.ci.nodes.lock(value);
            let sym = parser::find_symbol(&ast.symbols, arg.id);
            assert!(sym.flags & idfl::COMPTIME == 0, "TODO");
            self.ci.scope.vars.push(Variable { id: arg.id, value, ty, ptr: false });
        }

        if self.expr(body).is_some() && sig.ret == ty::Id::VOID {
            self.report(
                body.pos(),
                "expected all paths in the fucntion to return \
                or the return type to be 'void'",
            );
        }

        self.ci.finalize();

        if self.errors.borrow().is_empty() {
            self.ci.emit_body(&mut self.tys, self.files, sig);
            self.tys.ins.funcs[id as usize].code.append(&mut self.ci.code);
            self.tys.ins.funcs[id as usize].relocs.append(&mut self.ci.relocs);
        }

        self.pool.pop_ci(&mut self.ci);
    }

    fn ty(&mut self, expr: &Expr) -> ty::Id {
        self.parse_ty(self.ci.file, expr, None, self.files)
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, self.files, ty)
    }

    fn ast_display(&self, ast: &'a Expr<'a>) -> parser::Display<'a> {
        parser::Display::new(&self.cfile().file, ast)
    }

    #[must_use]
    #[track_caller]
    fn binop_ty(&mut self, pos: Pos, lhs: &mut Value, rhs: &mut Value, op: TokenKind) -> ty::Id {
        if let Some(upcasted) = lhs.ty.try_upcast(rhs.ty, ty::TyCheck::BinOp) {
            if lhs.ty != upcasted {
                lhs.ty = upcasted;
                lhs.id = self.ci.nodes.new_node(upcasted, Kind::Extend, [VOID, lhs.id]);
            } else if rhs.ty != upcasted {
                rhs.ty = upcasted;
                rhs.id = self.ci.nodes.new_node(upcasted, Kind::Extend, [VOID, rhs.id]);
            }
            upcasted
        } else {
            let ty = self.ty_display(lhs.ty);
            let expected = self.ty_display(rhs.ty);
            self.report(pos, fa!("'{ty} {op} {expected}' is not supported"));
            ty::Id::NEVER
        }
    }

    #[track_caller]
    fn assert_ty(
        &mut self,
        pos: Pos,
        src: &mut Value,
        expected: ty::Id,
        hint: impl fmt::Display,
    ) -> bool {
        if let Some(upcasted) = src.ty.try_upcast(expected, ty::TyCheck::BinOp)
            && upcasted == expected
        {
            if src.ty != upcasted {
                src.ty = upcasted;
                src.id = self.ci.nodes.new_node(upcasted, Kind::Extend, [VOID, src.id]);
            }
            true
        } else {
            let ty = self.ty_display(src.ty);
            let expected = self.ty_display(expected);
            self.report(pos, fa!("expected {hint} to be of type {expected}, got {ty}"));
            false
        }
    }

    #[track_caller]
    fn assert_report(&self, cond: bool, pos: Pos, msg: impl core::fmt::Display) {
        if !cond {
            self.report(pos, msg);
        }
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl core::fmt::Display) {
        let mut buf = self.errors.borrow_mut();
        write!(buf, "{}", self.cfile().report(pos, msg)).unwrap();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: impl Display) -> ! {
        log::info!("{ast:#?}");
        self.fatal_report(ast.pos(), fa!("compiler does not (yet) know how to handle ({hint})"));
    }

    fn cfile(&self) -> &'a parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn fatal_report(&self, pos: Pos, msg: impl Display) -> ! {
        self.report(pos, msg);
        panic!("{}", self.errors.borrow());
    }
}

// FIXME: make this more efficient (allocated with arena)

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

struct Function<'a> {
    sig: Sig,
    nodes: &'a mut Nodes,
    tys: &'a Types,
    blocks: Vec<Block>,
    instrs: Vec<Instr>,
}

impl Debug for Function<'_> {
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

    fn def_nid(&mut self, _nid: Nid) {}

    fn drg(&mut self, nid: Nid) -> regalloc2::Operand {
        self.def_nid(nid);
        regalloc2::Operand::reg_def(self.rg(nid))
    }

    fn rg(&self, nid: Nid) -> VReg {
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

        let node = self.nodes[nid].clone();
        match node.kind {
            Kind::Start => {
                debug_assert_matches!(self.nodes[node.outputs[0]].kind, Kind::Entry);
                self.emit_node(node.outputs[0], VOID)
            }
            Kind::End => {}
            Kind::If => {
                self.nodes[nid].ralloc_backref = self.nodes[prev].ralloc_backref;

                let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Kind::BinOp { op } = self.nodes[cond].kind
                    && let Some((_, swapped)) = op.cond_op(node.ty.is_signed())
                {
                    if swapped {
                        core::mem::swap(&mut then, &mut else_);
                    }
                    let &[_, lhs, rhs] = self.nodes[cond].inputs.as_slice() else { unreachable!() };
                    let ops = vec![self.urg(lhs), self.urg(rhs)];
                    self.add_instr(nid, ops);
                } else {
                    todo!()
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
                    self.def_nid(ph);
                    block.push(self.rg(ph));
                }
                self.blocks[self.nodes[nid].ralloc_backref as usize].params = block;
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::Return => {
                let ops = match self.tys.size_of(self.sig.ret) {
                    0 => vec![],
                    1..=8 => {
                        vec![regalloc2::Operand::reg_fixed_use(
                            self.rg(node.inputs[1]),
                            regalloc2::PReg::new(1, regalloc2::RegClass::Int),
                        )]
                    }
                    9..=16 => {
                        vec![self.urg(self.nodes[node.inputs[1]].inputs[1])]
                    }
                    17.. => {
                        vec![self.urg(node.inputs[1])]
                    }
                };

                self.add_instr(nid, ops);
                self.emit_node(node.outputs[0], nid);
            }
            Kind::CInt { .. }
                if node.outputs.iter().all(|&o| {
                    let ond = &self.nodes[o];
                    matches!(ond.kind, Kind::BinOp { op }
                        if op.imm_binop(ond.ty.is_signed(), 8).is_some()
                            && self.nodes.is_const(ond.inputs[2])
                            && op.cond_op(ond.ty.is_signed()).is_none())
                }) =>
            {
                self.nodes.lock(nid)
            }
            Kind::CInt { .. } => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Extend => {
                let ops = vec![self.drg(nid), self.urg(node.inputs[1])];
                self.add_instr(nid, ops);
            }
            Kind::Entry => {
                self.nodes[nid].ralloc_backref = self.add_block(nid);

                let mut parama = self.tys.parama(self.sig.ret);
                for (arg, ti) in
                    self.nodes[VOID].clone().outputs.into_iter().skip(2).zip(self.sig.args.range())
                {
                    let ty = self.tys.ins.args[ti];
                    match self.tys.size_of(ty) {
                        0 => continue,
                        1..=8 => {
                            self.def_nid(arg);
                            self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                                self.rg(arg),
                                regalloc2::PReg::new(parama.next() as _, regalloc2::RegClass::Int),
                            )]);
                        }
                        9..=16 => todo!(),
                        _ => {
                            self.def_nid(arg);
                            self.add_instr(NEVER, vec![regalloc2::Operand::reg_fixed_def(
                                self.rg(arg),
                                regalloc2::PReg::new(parama.next() as _, regalloc2::RegClass::Int),
                            )]);
                        }
                    }
                }

                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::Then | Kind::Else => {
                self.nodes[nid].ralloc_backref = self.add_block(nid);
                self.bridge(prev, nid);
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, nid);
                }
            }
            Kind::BinOp { op: TokenKind::Add }
                if self.nodes.is_const(node.inputs[2])
                    && node.outputs.iter().all(|&n| {
                        matches!(self.nodes[n].kind, Kind::Stre | Kind::Load)
                            && self.tys.size_of(self.nodes[n].ty) <= 8
                    }) =>
            {
                self.nodes.lock(nid)
            }
            Kind::BinOp { op } => {
                let &[_, lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                let ops = if let Kind::CInt { .. } = self.nodes[rhs].kind
                    && self.nodes[rhs].lock_rc != 0
                {
                    vec![self.drg(nid), self.urg(lhs)]
                } else if op.binop(node.ty.is_signed(), 8).is_some() {
                    vec![self.drg(nid), self.urg(lhs), self.urg(rhs)]
                } else if op.cond_op(node.ty.is_signed()).is_some() {
                    return;
                } else {
                    todo!("{op}")
                };
                self.add_instr(nid, ops);
            }
            Kind::UnOp { .. } => {
                let ops = vec![self.drg(nid), self.urg(node.inputs[1])];
                self.add_instr(nid, ops);
            }
            Kind::Call { func } => {
                self.nodes[nid].ralloc_backref = self.nodes[prev].ralloc_backref;
                let mut ops = vec![];

                let fuc = self.tys.ins.funcs[func as usize].sig.unwrap();
                if self.tys.size_of(fuc.ret) != 0 {
                    self.def_nid(nid);
                    ops.push(regalloc2::Operand::reg_fixed_def(
                        self.rg(nid),
                        regalloc2::PReg::new(1, regalloc2::RegClass::Int),
                    ));
                }

                let mut parama = self.tys.parama(fuc.ret);
                for (&(mut i), ti) in node.inputs[1..].iter().zip(fuc.args.range()) {
                    let ty = self.tys.ins.args[ti];
                    match self.tys.size_of(ty) {
                        0 => continue,
                        1..=8 => {
                            ops.push(regalloc2::Operand::reg_fixed_use(
                                self.rg(i),
                                regalloc2::PReg::new(parama.next() as _, regalloc2::RegClass::Int),
                            ));
                        }
                        9..=16 => todo!("pass in two register"),
                        _ => {
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
                                regalloc2::PReg::new(parama.next() as _, regalloc2::RegClass::Int),
                            ));
                        }
                    }
                }

                match self.tys.size_of(fuc.ret) {
                    0..=16 => {}
                    17.. => {
                        ops.push(regalloc2::Operand::reg_fixed_use(
                            self.rg(*node.inputs.last().unwrap()),
                            regalloc2::PReg::new(1, regalloc2::RegClass::Int),
                        ));
                    }
                }

                self.add_instr(nid, ops);

                for o in node.outputs.into_iter().rev() {
                    if self.nodes[o].inputs[0] == nid {
                        self.emit_node(o, nid);
                    }
                }
            }
            Kind::Global { .. } => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            //Kind::Stck
            //    if node.outputs.iter().all(|&n| {
            //        matches!(self.nodes[n].kind, Kind::Stre | Kind::Load)
            //            || matches!(self.nodes[n].kind, Kind::BinOp { op: TokenKind::Add }
            //    if self.nodes.is_const(self.nodes[n].inputs[2])
            //        && self.nodes[n]
            //            .outputs
            //            .iter()
            //            .all(|&n| matches!(self.nodes[n].kind, Kind::Stre | Kind::Load)))
            //    }) => {}
            Kind::Stck => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Idk => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Phi | Kind::Arg | Kind::Mem => {}
            Kind::Load { .. } => {
                let mut region = node.inputs[1];
                if self.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                    && self.nodes.is_const(self.nodes[region].inputs[2])
                {
                    region = self.nodes[region].inputs[1]
                }
                if self.tys.size_of(node.ty) <= 8 {
                    let ops = match self.nodes[region].kind {
                        Kind::Stck => vec![self.drg(nid)],
                        _ => vec![self.drg(nid), self.urg(region)],
                    };
                    self.add_instr(nid, ops);
                }
            }
            Kind::Stre if node.inputs[2] == VOID => self.nodes.lock(nid),
            Kind::Stre => {
                let mut region = node.inputs[2];
                if self.nodes[region].kind == (Kind::BinOp { op: TokenKind::Add })
                    && self.nodes.is_const(self.nodes[region].inputs[2])
                {
                    region = self.nodes[region].inputs[1]
                }
                let ops = match self.nodes[region].kind {
                    _ if self.tys.size_of(node.ty) > 8 => {
                        vec![self.urg(region), self.urg(self.nodes[node.inputs[1]].inputs[1])]
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
        if matches!(self.nodes[self.instrs[insn.index()].nid].kind, Kind::Call { .. }) {
            let mut set = regalloc2::PRegSet::default();
            for i in 2..13 {
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

fn loop_depth(target: Nid, nodes: &mut Nodes) -> LoopDepth {
    if nodes[target].loop_depth != 0 {
        return nodes[target].loop_depth;
    }

    nodes[target].loop_depth = match nodes[target].kind {
        Kind::Entry | Kind::Then | Kind::Else | Kind::Call { .. } | Kind::Return | Kind::If => {
            let dpth = loop_depth(nodes[target].inputs[0], nodes);
            if nodes[target].loop_depth != 0 {
                return nodes[target].loop_depth;
            }
            dpth
        }
        Kind::Region => {
            let l = loop_depth(nodes[target].inputs[0], nodes);
            let r = loop_depth(nodes[target].inputs[1], nodes);
            debug_assert_eq!(l, r);
            l
        }
        Kind::Loop => {
            let depth = loop_depth(nodes[target].inputs[0], nodes) + 1;
            nodes[target].loop_depth = depth;
            let mut cursor = nodes[target].inputs[1];
            while cursor != target {
                nodes[cursor].loop_depth = depth;
                let next = if nodes[cursor].kind == Kind::Region {
                    loop_depth(nodes[cursor].inputs[0], nodes);
                    nodes[cursor].inputs[1]
                } else {
                    idom(nodes, cursor)
                };
                debug_assert_ne!(next, VOID);
                if matches!(nodes[cursor].kind, Kind::Then | Kind::Else) {
                    let other = *nodes[next]
                        .outputs
                        .iter()
                        .find(|&&n| nodes[n].kind != nodes[cursor].kind)
                        .unwrap();
                    if nodes[other].loop_depth == 0 {
                        nodes[other].loop_depth = depth - 1;
                    }
                }
                cursor = next;
            }
            depth
        }
        Kind::Start | Kind::End => 1,
        u => unreachable!("{u:?}"),
    };

    nodes[target].loop_depth
}

fn idepth(nodes: &mut Nodes, target: Nid) -> IDomDepth {
    if target == VOID {
        return 0;
    }
    if nodes[target].depth == 0 {
        nodes[target].depth = match nodes[target].kind {
            Kind::End | Kind::Start => unreachable!("{:?}", nodes[target].kind),
            Kind::Region => {
                idepth(nodes, nodes[target].inputs[0]).max(idepth(nodes, nodes[target].inputs[1]))
            }
            _ => idepth(nodes, nodes[target].inputs[0]),
        } + 1;
    }
    nodes[target].depth
}

fn push_up(nodes: &mut Nodes) {
    fn collect_rpo(node: Nid, nodes: &mut Nodes, rpo: &mut Vec<Nid>) {
        if !nodes.is_cfg(node) || !nodes.visited.set(node) {
            return;
        }

        for i in 0..nodes[node].outputs.len() {
            collect_rpo(nodes[node].outputs[i], nodes, rpo);
        }
        rpo.push(node);
    }

    fn push_up_impl(node: Nid, nodes: &mut Nodes) {
        if !nodes.visited.set(node) {
            return;
        }

        for i in 0..nodes[node].inputs.len() {
            let inp = nodes[node].inputs[i];
            if !nodes[inp].kind.is_pinned() {
                push_up_impl(inp, nodes);
            }
        }

        if nodes[node].kind.is_pinned() {
            return;
        }

        let mut deepest = VOID;
        for i in 1..nodes[node].inputs.len() {
            let inp = nodes[node].inputs[i];
            if idepth(nodes, inp) > idepth(nodes, deepest) {
                deepest = idom(nodes, inp);
            }
        }

        if deepest == VOID {
            return;
        }

        let index = nodes[0].outputs.iter().position(|&p| p == node).unwrap();
        nodes[0].outputs.remove(index);
        nodes[node].inputs[0] = deepest;
        debug_assert!(
            !nodes[deepest].outputs.contains(&node)
                || matches!(nodes[deepest].kind, Kind::Call { .. }),
            "{node} {:?} {deepest} {:?}",
            nodes[node],
            nodes[deepest]
        );
        nodes[deepest].outputs.push(node);
    }

    let mut rpo = vec![];

    collect_rpo(VOID, nodes, &mut rpo);

    for node in rpo.into_iter().rev() {
        loop_depth(node, nodes);
        for i in 0..nodes[node].inputs.len() {
            push_up_impl(nodes[node].inputs[i], nodes);
        }

        if matches!(nodes[node].kind, Kind::Loop | Kind::Region) {
            for i in 0..nodes[node].outputs.len() {
                let usage = nodes[node].outputs[i];
                if nodes[usage].kind == Kind::Phi {
                    push_up_impl(usage, nodes);
                }
            }
        }
    }
}

fn push_down(nodes: &mut Nodes, node: Nid) {
    fn is_forward_edge(usage: Nid, def: Nid, nodes: &mut Nodes) -> bool {
        match nodes[usage].kind {
            Kind::Phi => {
                nodes[usage].inputs[2] != def || nodes[nodes[usage].inputs[0]].kind != Kind::Loop
            }
            Kind::Loop => nodes[usage].inputs[1] != def,
            _ => true,
        }
    }

    fn better(nodes: &mut Nodes, is: Nid, then: Nid) -> bool {
        loop_depth(is, nodes) < loop_depth(then, nodes)
            || idepth(nodes, is) > idepth(nodes, then)
            || nodes[then].kind == Kind::If
    }

    if !nodes.visited.set(node) {
        return;
    }

    for usage in nodes[node].outputs.clone() {
        if is_forward_edge(usage, node, nodes) {
            push_down(nodes, usage);
        }
    }

    if nodes[node].kind.is_pinned() {
        return;
    }

    let mut min = None::<Nid>;
    for i in 0..nodes[node].outputs.len() {
        let usage = nodes[node].outputs[i];
        let ub = use_block(node, usage, nodes);
        min = min.map(|m| common_dom(ub, m, nodes)).or(Some(ub));
    }
    let mut min = min.unwrap();

    debug_assert!(nodes.dominates(nodes[node].inputs[0], min));

    let mut cursor = min;
    loop {
        if better(nodes, cursor, min) {
            min = cursor;
        }
        if cursor == nodes[node].inputs[0] {
            break;
        }
        cursor = idom(nodes, cursor);
    }

    if nodes[min].kind.ends_basic_block() {
        min = idom(nodes, min);
    }

    nodes.check_dominance(node, min, true);

    let prev = nodes[node].inputs[0];
    debug_assert!(idepth(nodes, min) >= idepth(nodes, prev));
    let index = nodes[prev].outputs.iter().position(|&p| p == node).unwrap();
    nodes[prev].outputs.remove(index);
    nodes[node].inputs[0] = min;
    nodes[min].outputs.push(node);
}

fn use_block(target: Nid, from: Nid, nodes: &mut Nodes) -> Nid {
    if nodes[from].kind != Kind::Phi {
        return idom(nodes, from);
    }

    let index = nodes[from].inputs.iter().position(|&n| n == target).unwrap();
    nodes[nodes[from].inputs[0]].inputs[index - 1]
}

fn idom(nodes: &mut Nodes, target: Nid) -> Nid {
    match nodes[target].kind {
        Kind::Start => VOID,
        Kind::End => unreachable!(),
        Kind::Region => {
            let &[lcfg, rcfg] = nodes[target].inputs.as_slice() else { unreachable!() };
            common_dom(lcfg, rcfg, nodes)
        }
        _ => nodes[target].inputs[0],
    }
}

fn common_dom(mut a: Nid, mut b: Nid, nodes: &mut Nodes) -> Nid {
    while a != b {
        let [ldepth, rdepth] = [idepth(nodes, a), idepth(nodes, b)];
        if ldepth >= rdepth {
            a = idom(nodes, a);
        }
        if ldepth <= rdepth {
            b = idom(nodes, b);
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use {
        alloc::{string::String, vec::Vec},
        core::fmt::Write,
    };

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        _ = log::set_logger(&crate::fs::Logger);
        log::set_max_level(log::LevelFilter::Info);

        let (ref files, _embeds) = crate::test_parse_files(ident, input);
        let mut codegen = super::Codegen { files, ..Default::default() };

        codegen.generate();

        {
            let errors = codegen.errors.borrow();
            if !errors.is_empty() {
                output.push_str(&errors);
                return;
            }
        }

        let mut out = Vec::new();
        codegen.tys.reassemble(&mut out);

        let err = codegen.tys.disasm(&out, codegen.files, output, |_| {});
        if let Err(e) = err {
            writeln!(output, "!!! asm is invalid: {e}").unwrap();
            return;
        }

        crate::test_run_vm(&out, output);
    }

    crate::run_tests! { generate:
        // Tour Examples
        main_fn;
        arithmetic;
        functions;
        comments;
        if_statements;
        variables;
        loops;
        pointers;
        structs;
        hex_octal_binary_literals;
        //struct_operators;
        global_variables;
        //directives;
        c_strings;
        struct_patterns;
        arrays;
        //inline;
        idk;

        // Incomplete Examples;
        //comptime_pointers;
        //generic_types;
        //generic_functions;
        fb_driver;

        // Purely Testing Examples;
        wide_ret;
        comptime_min_reg_leak;
        //different_types;
        //struct_return_from_module_function;
        sort_something_viredly;
        //structs_in_registers;
        comptime_function_from_another_file;
        //inline_test;
        //inlined_generic_functions;
        //some_generic_code;
        //integer_inference_issues;
        writing_into_string;
        //request_page;
        //tests_ptr_to_ptr_copy;

        // Just Testing Optimizations;
        const_folding_with_arg;
        branch_assignments;
        exhaustive_loop_testing;
        pointer_opts;
        conditional_stores;
        loop_stores;
    }
}
