use {
    self::{
        hbvm::{Comptime, HbvmBackend},
        strong_ref::StrongRef,
    },
    crate::{
        ctx_map::CtxEntry,
        debug,
        lexer::{self, TokenKind},
        parser::{
            self,
            idfl::{self},
            CtorField, Expr, Pos,
        },
        ty::{self, Arg, ArrayLen, Loc, Module, Tuple},
        utils::{BitSet, Ent, Vc},
        CompState, FTask, Func, Global, Ident, Offset, OffsetIter, OptLayout, Sig, StringRef,
        SymKey, TypeParser, Types,
    },
    alloc::{string::String, vec::Vec},
    core::{
        assert_matches::debug_assert_matches,
        cell::{Cell, RefCell},
        fmt::{self, Debug, Display, Write},
        format_args as fa, mem,
        ops::{self, Range},
    },
    hashbrown::hash_map,
    hbbytecode::DisasmError,
};

const VOID: Nid = 0;
const NEVER: Nid = 1;
const ENTRY: Nid = 2;
const MEM: Nid = 3;
const LOOPS: Nid = 4;
const ARG_START: usize = 3;
const DEFAULT_ACLASS: usize = 0;
const GLOBAL_ACLASS: usize = 1;

pub mod hbvm;

type Nid = u16;
type AClassId = i16;

pub struct AssemblySpec {
    entry: u32,
    code_length: u64,
    data_length: u64,
}

pub trait Backend {
    fn assemble_reachable(
        &mut self,
        from: ty::Func,
        types: &Types,
        to: &mut Vec<u8>,
    ) -> AssemblySpec;
    fn disasm<'a>(
        &'a self,
        sluce: &[u8],
        eca_handler: &mut dyn FnMut(&mut &[u8]),
        types: &'a Types,
        files: &'a [parser::Ast],
        output: &mut String,
    ) -> Result<(), hbbytecode::DisasmError<'a>>;
    fn emit_body(&mut self, id: ty::Func, ci: &mut Nodes, tys: &Types, files: &[parser::Ast]);

    fn emit_ct_body(&mut self, id: ty::Func, ci: &mut Nodes, tys: &Types, files: &[parser::Ast]) {
        self.emit_body(id, ci, tys, files);
    }

    fn assemble_bin(&mut self, from: ty::Func, types: &Types, to: &mut Vec<u8>) {
        self.assemble_reachable(from, types, to);
    }
}

type Lookup = crate::ctx_map::CtxMap<Nid>;

impl crate::ctx_map::CtxEntry for Nid {
    type Ctx = [Result<Node, (Nid, debug::Trace)>];
    type Key<'a> = (Kind, &'a [Nid], ty::Id);

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        ctx[*self as usize].as_ref().unwrap_or_else(|(_, t)| panic!("{t:#?}")).key()
    }
}

macro_rules! inference {
    ($ty:ident, $ctx:expr, $self:expr, $pos:expr, $subject:literal, $example:literal) => {
        let Some($ty) = $ctx.ty else {
            $self.report(
                $pos,
                concat!(
                    "resulting ",
                    $subject,
                    " cannot be inferred from context, consider using `",
                    $example,
                    "` to hint the type",
                ),
            );
            return Value::NEVER;
        };
    };
}

#[derive(Clone)]
pub struct Nodes {
    values: Vec<Result<Node, (Nid, debug::Trace)>>,
    queued_peeps: Vec<Nid>,
    free: Nid,
    lookup: Lookup,
}

impl Default for Nodes {
    fn default() -> Self {
        Self {
            values: Default::default(),
            queued_peeps: Default::default(),
            free: Nid::MAX,
            lookup: Default::default(),
        }
    }
}

impl Nodes {
    fn loop_depth(&self, target: Nid) -> LoopDepth {
        self[target].loop_depth.set(match self[target].kind {
            Kind::Region | Kind::Entry | Kind::Then | Kind::Else | Kind::Call { .. } | Kind::If => {
                if self[target].loop_depth.get() != 0 {
                    return self[target].loop_depth.get();
                }
                self.loop_depth(self[target].inputs[0])
            }
            Kind::Loop => {
                if self[target].loop_depth.get() == self.loop_depth(self[target].inputs[0]) + 1 {
                    return self[target].loop_depth.get();
                }
                let depth = self.loop_depth(self[target].inputs[0]) + 1;
                self[target].loop_depth.set(depth);
                let mut cursor = self[target].inputs[1];
                while cursor != target {
                    self[cursor].loop_depth.set(depth);
                    let next = self.idom(cursor);
                    debug_assert_ne!(next, 0);
                    if matches!(self[cursor].kind, Kind::Then | Kind::Else) {
                        debug_assert_eq!(self[next].kind, Kind::If);
                        let other = self[next].outputs[(self[next].outputs[0] == cursor) as usize];
                        self[other].loop_depth.set(depth - 1);
                    }
                    cursor = next;
                }
                depth
            }
            Kind::Start | Kind::End | Kind::Die | Kind::Return => 1,
            u => unreachable!("{u:?}"),
        });

        self[target].loop_depth.get()
    }

    fn idepth(&self, target: Nid) -> IDomDepth {
        if target == VOID {
            return 0;
        }
        if self[target].depth.get() == 0 {
            let depth = match self[target].kind {
                Kind::End | Kind::Start => unreachable!("{:?}", self[target].kind),
                Kind::Region => {
                    self.idepth(self[target].inputs[0]).max(self.idepth(self[target].inputs[1]))
                }
                _ => self.idepth(self[target].inputs[0]),
            } + 1;
            self[target].depth.set(depth);
        }
        self[target].depth.get()
    }

    fn fix_loops(&mut self, stack: &mut Vec<Nid>, seen: &mut BitSet) {
        debug_assert!(stack.is_empty());

        stack.push(NEVER);

        while let Some(node) = stack.pop() {
            if seen.set(node) && self.is_cfg(node) {
                stack.extend(self[node].inputs.iter());
            }
        }

        for l in self[LOOPS].outputs.clone() {
            if !seen.get(l) {
                self[l].outputs.push(NEVER);
                self[NEVER].inputs.push(l);
            }
        }
    }

    fn push_up_impl(&mut self, node: Nid, visited: &mut BitSet) {
        if !visited.set(node) {
            return;
        }

        for i in 1..self[node].inputs.len() {
            let inp = self[node].inputs[i];
            if !self[inp].kind.is_pinned() {
                self.push_up_impl(inp, visited);
            }
        }

        if self[node].kind.is_pinned() {
            return;
        }

        let mut deepest = self[node].inputs[0];
        for &inp in self[node].inputs[1..].iter() {
            if self.idepth(inp) > self.idepth(deepest) {
                if self[inp].kind.is_call() {
                    deepest = inp;
                } else {
                    debug_assert!(!self.is_cfg(inp));
                    deepest = self.idom(inp);
                }
            }
        }

        if deepest == self[node].inputs[0] {
            return;
        }

        let current = self[node].inputs[0];

        let index = self[current].outputs.iter().position(|&p| p == node).unwrap();
        self[current].outputs.remove(index);
        self[node].inputs[0] = deepest;
        debug_assert!(
            !self[deepest].outputs.contains(&node) || self[deepest].kind.is_call(),
            "{node} {:?} {deepest} {:?}",
            self[node],
            self[deepest]
        );
        self[deepest].outputs.push(node);
    }

    fn collect_rpo(&self, node: Nid, rpo: &mut Vec<Nid>, visited: &mut BitSet) {
        if !self.is_cfg(node) || !visited.set(node) {
            return;
        }

        for &n in self[node].outputs.iter() {
            self.collect_rpo(n, rpo, visited);
        }

        rpo.push(node);
    }

    fn push_up(&mut self, rpo: &mut Vec<Nid>, visited: &mut BitSet) {
        debug_assert!(rpo.is_empty());
        self.collect_rpo(VOID, rpo, visited);

        for &node in rpo.iter().rev() {
            self.loop_depth(node);
            for i in 0..self[node].inputs.len() {
                self.push_up_impl(self[node].inputs[i], visited);
            }

            if matches!(self[node].kind, Kind::Loop | Kind::Region) {
                for i in 0..self[node].outputs.len() {
                    let usage = self[node].outputs[i];
                    if self[usage].kind == Kind::Phi {
                        self.push_up_impl(usage, visited);
                    }
                }
            }
        }

        debug_assert_eq!(
            self.iter()
                .map(|(n, _)| n)
                .filter(|&n| !visited.get(n)
                    && !matches!(self[n].kind, Kind::Arg | Kind::Mem | Kind::Loops))
                .collect::<Vec<_>>(),
            vec![],
            "{:?}",
            self.iter()
                .filter(|&(n, nod)| !visited.get(n)
                    && !matches!(nod.kind, Kind::Arg | Kind::Mem | Kind::Loops))
                .collect::<Vec<_>>()
        );

        rpo.clear();
    }

    fn better(&mut self, is: Nid, then: Nid) -> bool {
        debug_assert_ne!(self.idepth(is), self.idepth(then), "{is} {then}");
        self.loop_depth(is) < self.loop_depth(then)
            || self.idepth(is) > self.idepth(then)
            || self[then].kind == Kind::If
    }

    fn is_forward_edge(&mut self, usage: Nid, def: Nid) -> bool {
        match self[usage].kind {
            Kind::Phi => {
                self[usage].inputs[2] != def || self[self[usage].inputs[0]].kind != Kind::Loop
            }
            Kind::Loop => self[usage].inputs[1] != def,
            _ => true,
        }
    }

    fn push_down(&mut self, node: Nid, visited: &mut BitSet, antideps: &mut [Nid]) {
        if !visited.set(node) {
            return;
        }

        for usage in self[node].outputs.clone() {
            if self.is_forward_edge(usage, node) && self[node].kind == Kind::Stre {
                self.push_down(usage, visited, antideps);
            }
        }

        for usage in self[node].outputs.clone() {
            if self.is_forward_edge(usage, node) {
                self.push_down(usage, visited, antideps);
            }
        }

        if self[node].kind.is_pinned() {
            return;
        }

        let mut min = None::<Nid>;
        for i in 0..self[node].outputs.len() {
            let usage = self[node].outputs[i];
            let ub = self.use_block(node, usage);
            min = min.map(|m| self.common_dom(ub, m)).or(Some(ub));
        }
        let mut min = min.unwrap();

        debug_assert!(self.dominates(self[node].inputs[0], min));

        let mut cursor = min;
        while cursor != self[node].inputs[0] {
            cursor = self.idom(cursor);
            if self.better(cursor, min) {
                min = cursor;
            }
        }

        if self[node].kind == Kind::Load {
            min = self.find_antideps(node, min, antideps);
        }

        if self[node].kind == Kind::Stre {
            antideps[node as usize] = self[node].inputs[0];
        }

        if self[min].kind.ends_basic_block() {
            min = self.idom(min);
        }

        self.assert_dominance(node, min, true);

        let prev = self[node].inputs[0];
        debug_assert!(self.idepth(min) >= self.idepth(prev));
        let index = self[prev].outputs.iter().position(|&p| p == node).unwrap();
        self[prev].outputs.remove(index);
        self[node].inputs[0] = min;
        self[min].outputs.push(node);
    }

    fn find_antideps(&mut self, load: Nid, mut min: Nid, antideps: &mut [Nid]) -> Nid {
        debug_assert!(self[load].kind == Kind::Load);

        let (aclass, _) = self.aclass_index(self[load].inputs[1]);

        let mut cursor = min;
        while cursor != self[load].inputs[0] {
            antideps[cursor as usize] = load;
            if self[cursor].clobbers.get(aclass as _) {
                min = self[cursor].inputs[0];
                break;
            }
            cursor = self.idom(cursor);
        }

        if self[load].inputs[2] == MEM {
            return min;
        }

        for out in self[self[load].inputs[2]].outputs.clone() {
            match self[out].kind {
                Kind::Stre => {
                    let mut cursor = self[out].inputs[0];
                    while cursor != antideps[out as usize] {
                        if antideps[cursor as usize] == load {
                            min = self.common_dom(min, cursor);
                            if min == cursor {
                                self.bind(load, out);
                            }
                            break;
                        }
                        cursor = self.idom(cursor);
                    }
                    break;
                }
                Kind::Phi => {
                    let n = self[out].inputs[1..]
                        .iter()
                        .position(|&n| n == self[load].inputs[2])
                        .unwrap();
                    let mut cursor = self[self[out].inputs[0]].inputs[n];
                    while cursor != antideps[out as usize] {
                        if antideps[cursor as usize] == load {
                            min = self.common_dom(min, cursor);
                            break;
                        }
                        cursor = self.idom(cursor);
                    }
                }
                _ => {}
            }
        }

        min
    }

    fn bind(&mut self, from: Nid, to: Nid) {
        debug_assert_ne!(to, 0);
        debug_assert_ne!(self[to].kind, Kind::Phi);
        self[from].outputs.push(to);
        self[to].inputs.push(from);
    }

    fn use_block(&self, target: Nid, from: Nid) -> Nid {
        if self[from].kind != Kind::Phi {
            return self.idom(from);
        }

        let index = self[from].inputs.iter().position(|&n| n == target).unwrap_or_else(|| {
            panic!("from {from} {:?} target {target} {:?}", self[from], self[target])
        });
        self[self[from].inputs[0]].inputs[index - 1]
    }

    fn idom(&self, target: Nid) -> Nid {
        match self[target].kind {
            Kind::Start => VOID,
            Kind::End => unreachable!(),
            Kind::Region => {
                let &[lcfg, rcfg] = self[target].inputs.as_slice() else { unreachable!() };
                self.common_dom(lcfg, rcfg)
            }
            _ => self[target].inputs[0],
        }
    }

    fn common_dom(&self, mut a: Nid, mut b: Nid) -> Nid {
        while a != b {
            let [ldepth, rdepth] = [self.idepth(a), self.idepth(b)];
            if ldepth >= rdepth {
                a = self.idom(a);
            }
            if ldepth <= rdepth {
                b = self.idom(b);
            }
        }
        a
    }

    fn merge_scopes(
        &mut self,
        loops: &mut [Loop],
        ctrl: &StrongRef,
        to: &mut Scope,
        from: &mut Scope,
        tys: &Types,
    ) {
        for (i, (to_value, from_value)) in to.vars.iter_mut().zip(from.vars.iter_mut()).enumerate()
        {
            debug_assert_eq!(to_value.ty, from_value.ty);
            if to_value.value() != from_value.value() {
                self.load_loop_var(i, from_value, loops);
                self.load_loop_var(i, to_value, loops);
                if to_value.value() != from_value.value() {
                    debug_assert!(!to_value.ptr);
                    debug_assert!(!from_value.ptr);
                    let inps = [ctrl.get(), from_value.value(), to_value.value()];
                    to_value
                        .set_value_remove(self.new_node(from_value.ty, Kind::Phi, inps, tys), self);
                }
            }
        }

        for (i, (to_class, from_class)) in
            to.aclasses.iter_mut().zip(from.aclasses.iter_mut()).enumerate()
        {
            if to_class.last_store.get() != from_class.last_store.get() {
                self.load_loop_aclass(i, from_class, loops);
                self.load_loop_aclass(i, to_class, loops);
                if to_class.last_store.get() != from_class.last_store.get() {
                    let inps = [ctrl.get(), from_class.last_store.get(), to_class.last_store.get()];
                    to_class
                        .last_store
                        .set_remove(self.new_node(ty::Id::VOID, Kind::Phi, inps, tys), self);
                }
            }
        }
    }

    fn graphviz_low(&self, disp: ty::Display, out: &mut String) -> core::fmt::Result {
        use core::fmt::Write;

        writeln!(out)?;
        writeln!(out, "digraph G {{")?;
        writeln!(out, "rankdir=BT;")?;
        writeln!(out, "concentrate=true;")?;
        writeln!(out, "compound=true;")?;

        for (i, node) in self.iter() {
            let color = match () {
                _ if node.lock_rc.get() == Nid::MAX => "orange",
                _ if node.lock_rc.get() == Nid::MAX - 1 => "blue",
                _ if node.lock_rc.get() != 0 => "red",
                _ if node.outputs.is_empty() => "purple",
                _ if node.is_mem() => "green",
                _ if self.is_cfg(i) => "yellow",
                _ => "white",
            };

            if node.ty != ty::Id::VOID {
                writeln!(
                    out,
                    " node{i}[label=\"{i} {} {} {}\" color={color}]",
                    node.kind,
                    disp.rety(node.ty),
                    node.aclass,
                )?;
            } else {
                writeln!(
                    out,
                    " node{i}[label=\"{i} {} {}\" color={color}]",
                    node.kind, node.aclass,
                )?;
            }

            for (j, &o) in node.outputs.iter().enumerate() {
                let color = if self.is_cfg(i) && self.is_cfg(o) { "red" } else { "lightgray" };
                let index = self[o].inputs.iter().position(|&inp| i == inp).unwrap();
                let style = if index == 0 && !self.is_cfg(o) { "style=dotted" } else { "" };
                writeln!(
                    out,
                    " node{o} -> node{i}[color={color} taillabel={index} headlabel={j} {style}]",
                )?;
            }
        }

        writeln!(out, "}}")?;

        Ok(())
    }

    fn graphviz(&self, disp: ty::Display) {
        let out = &mut String::new();
        _ = self.graphviz_low(disp, out);
        log::info!("{out}");
    }

    fn graphviz_in_browser(&self, _disp: ty::Display) {
        #[cfg(all(test, feature = "std"))]
        {
            let out = &mut String::new();
            _ = self.graphviz_low(_disp, out);
            if !std::process::Command::new("brave")
                .arg(format!("https://dreampuf.github.io/GraphvizOnline/#{out}"))
                .status()
                .unwrap()
                .success()
            {
                log::error!("{out}");
            }
        }
    }

    fn gcm(&mut self, rpo: &mut Vec<Nid>, visited: &mut BitSet) {
        visited.clear(self.values.len());
        self.fix_loops(rpo, visited);
        visited.clear(self.values.len());
        self.push_up(rpo, visited);
        visited.clear(self.values.len());
        debug_assert!(rpo.is_empty());
        rpo.resize(self.values.len(), VOID);
        self.push_down(VOID, visited, rpo);
        rpo.clear();
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = Nid::MAX;
    }

    fn new_node_nop(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Nid {
        let node = Node { inputs: inps.into(), kind, ty, ..Default::default() };

        if node.kind == Kind::Phi && node.ty != ty::Id::VOID {
            debug_assert_ne!(
                self[node.inputs[1]].ty,
                ty::Id::VOID,
                "{:?} {:?}",
                self[node.inputs[1]],
                node.ty.expand(),
            );

            if self[node.inputs[0]].kind != Kind::Loop {
                debug_assert_ne!(
                    self[node.inputs[2]].ty,
                    ty::Id::VOID,
                    "{:?} {:?}",
                    self[node.inputs[2]],
                    node.ty.expand(),
                );
            }

            debug_assert!(!matches!(node.ty.expand(), ty::Kind::Struct(_)));
        }

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
            self.values.push(Err((Nid::MAX, debug::trace())));
        }

        let free = self.free;
        for &d in node.inputs.as_slice() {
            debug_assert_ne!(d, free);
            self.values[d as usize].as_mut().unwrap_or_else(|_| panic!("{d}")).outputs.push(free);
        }
        self.free = mem::replace(&mut self.values[free as usize], Ok(node)).unwrap_err().0;

        if let Some((entry, hash)) = lookup_meta {
            entry.insert(crate::ctx_map::Key { value: free, hash }, ());
        }
        free
    }

    fn remove_node_lookup(&mut self, target: Nid) {
        if !self[target].is_not_gvnd() {
            self.lookup
                .remove(&target, &self.values)
                .unwrap_or_else(|| panic!("{:?}", self[target]));
        }
    }

    fn new_node(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>, tys: &Types) -> Nid {
        let id = self.new_node_nop(ty, kind, inps);
        if let Some(opt) = self.peephole(id, tys) {
            debug_assert_ne!(opt, id);
            self.queued_peeps.clear();
            self.lock(opt);
            self.remove(id);
            self.unlock(opt);
            opt
        } else {
            id
        }
    }

    fn new_const(&mut self, ty: ty::Id, value: impl Into<i64>) -> Nid {
        self.new_node_nop(ty, Kind::CInt { value: value.into() }, [VOID])
    }

    fn new_const_lit(&mut self, ty: ty::Id, value: impl Into<i64>) -> Value {
        Value::new(self.new_const(ty, value)).ty(ty)
    }

    fn new_node_lit(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>, tys: &Types) -> Value {
        Value::new(self.new_node(ty, kind, inps, tys)).ty(ty)
    }

    fn is_locked(&self, target: Nid) -> bool {
        self[target].lock_rc.get() != 0
    }

    fn is_unlocked(&self, target: Nid) -> bool {
        self[target].lock_rc.get() == 0
    }

    fn lock(&self, target: Nid) {
        self[target].lock_rc.set(self[target].lock_rc.get() + 1);
    }

    #[track_caller]
    fn unlock(&self, target: Nid) {
        self[target].lock_rc.set(self[target].lock_rc.get() - 1);
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

        if cfg!(debug_assertions) {
            mem::replace(&mut self.values[target as usize], Err((Nid::MAX, debug::trace())))
                .unwrap();
        } else {
            mem::replace(&mut self.values[target as usize], Err((self.free, debug::trace())))
                .unwrap();
            self.free = target;
        }

        true
    }

    fn late_peephole(&mut self, target: Nid, tys: &Types) -> Option<Nid> {
        if let Some(id) = self.peephole(target, tys) {
            self.queued_peeps.clear();
            self.replace(target, id);
            return None;
        }
        None
    }

    fn iter_peeps(&mut self, mut fuel: usize, stack: &mut Vec<Nid>, tys: &Types) {
        debug_assert!(stack.is_empty());

        self.iter()
            .filter_map(|(id, node)| node.kind.is_peeped().then_some(id))
            .collect_into(stack);
        stack.iter().for_each(|&s| self.lock(s));

        while fuel != 0
            && let Some(node) = stack.pop()
        {
            fuel -= 1;

            if self.unlock_remove(node) {
                continue;
            }

            if let Some(new) = self.peephole(node, tys) {
                let plen = stack.len();
                stack.append(&mut self.queued_peeps);
                for &p in &stack[plen..] {
                    self.lock(p);
                }
                self.replace(node, new);
                self.push_adjacent_nodes(new, stack);
            }

            //debug_assert_matches!(
            //    self.iter()
            //        .find(|(i, n)| n.lock_rc != 0 && n.kind.is_peeped() && !stack.contains(i)),
            //    None
            //);
        }

        stack.drain(..).for_each(|s| _ = self.unlock_remove(s));
    }

    fn push_adjacent_nodes(&mut self, of: Nid, stack: &mut Vec<Nid>) {
        let prev_len = stack.len();
        for &i in self[of]
            .outputs
            .iter()
            .chain(self[of].inputs.iter())
            .chain(self[of].peep_triggers.iter())
        {
            if self.values[i as usize].is_ok()
                && self[i].kind.is_peeped()
                && self[i].lock_rc.get() == 0
            {
                stack.push(i);
            }
        }

        self[of].peep_triggers = Vc::default();
        stack.iter().skip(prev_len).for_each(|&n| self.lock(n));
    }

    fn aclass_index(&self, region: Nid) -> (usize, Nid) {
        if self[region].aclass >= 0 {
            (self[region].aclass as _, region)
        } else {
            (
                self[self[region].aclass.unsigned_abs() - 1].aclass as _,
                self[region].aclass.unsigned_abs() - 1,
            )
        }
    }

    fn pass_aclass(&mut self, from: Nid, to: Nid) {
        debug_assert!(self[from].aclass >= 0);
        if from != to {
            self[to].aclass = -(from as AClassId + 1);
        }
    }

    fn peephole(&mut self, target: Nid, tys: &Types) -> Option<Nid> {
        use {Kind as K, TokenKind as T};
        match self[target].kind {
            K::BinOp { op } => {
                let &[ctrl, mut lhs, mut rhs] = self[target].inputs.as_slice() else {
                    unreachable!()
                };
                let ty = self[target].ty;

                let is_float = self[lhs].ty.is_float();

                if let (&K::CInt { value: a }, &K::CInt { value: b }) =
                    (&self[lhs].kind, &self[rhs].kind)
                {
                    return Some(self.new_const(ty, op.apply_binop(a, b, is_float)));
                }

                if lhs == rhs {
                    match op {
                        T::Sub => return Some(self.new_const(ty, 0)),
                        T::Add => {
                            let rhs = self.new_const(ty, 2);
                            return Some(self.new_node(
                                ty,
                                K::BinOp { op: T::Mul },
                                [ctrl, lhs, rhs],
                                tys,
                            ));
                        }
                        _ => {}
                    }
                }

                // this is more general the pushing constants to left to help deduplicate expressions more
                let mut changed = false;
                if op.is_comutative() && self[lhs].key() < self[rhs].key() {
                    mem::swap(&mut lhs, &mut rhs);
                    changed = true;
                }

                if let K::CInt { value } = self[rhs].kind {
                    match (op, value) {
                        (T::Eq, 0) if self[lhs].ty.is_pointer() || self[lhs].kind == Kind::Stck => {
                            return Some(self.new_const(ty::Id::BOOL, 0));
                        }
                        (T::Ne, 0) if self[lhs].ty.is_pointer() || self[lhs].kind == Kind::Stck => {
                            return Some(self.new_const(ty::Id::BOOL, 1));
                        }
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
                        let new_rhs = self.new_const(ty, op.apply_binop(av, bv, is_float));
                        return Some(self.new_node(ty, K::BinOp { op }, [ctrl, a, new_rhs], tys));
                    }

                    if self.is_const(b) {
                        // (a op #b) op c => (a op c) op #b
                        let new_lhs = self.new_node(ty, K::BinOp { op }, [ctrl, a, rhs], tys);
                        return Some(self.new_node(ty, K::BinOp { op }, [ctrl, new_lhs, b], tys));
                    }

                    self.add_trigger(b, target);
                }

                if op == T::Add
                    && self[lhs].kind == (K::BinOp { op: T::Mul })
                    && self[lhs].inputs[1] == rhs
                    && let K::CInt { value } = self[self[lhs].inputs[2]].kind
                {
                    // a * #n + a => a * (#n + 1)
                    let new_rhs = self.new_const(ty, value + 1);
                    return Some(self.new_node(
                        ty,
                        K::BinOp { op: T::Mul },
                        [ctrl, rhs, new_rhs],
                        tys,
                    ));
                }

                if op == T::Sub
                    && self[lhs].kind == (K::BinOp { op: T::Add })
                    && let K::CInt { value: a } = self[rhs].kind
                    && let K::CInt { value: b } = self[self[lhs].inputs[2]].kind
                {
                    let new_rhs = self.new_const(ty, b - a);
                    return Some(self.new_node(
                        ty,
                        K::BinOp { op: T::Add },
                        [ctrl, self[lhs].inputs[1], new_rhs],
                        tys,
                    ));
                }

                if op == T::Sub && self[lhs].kind == (K::BinOp { op }) {
                    // (a - b) - c => a - (b + c)
                    let &[_, a, b] = self[lhs].inputs.as_slice() else { unreachable!() };
                    let c = rhs;
                    let new_rhs = self.new_node(ty, K::BinOp { op: T::Add }, [ctrl, b, c], tys);
                    return Some(self.new_node(ty, K::BinOp { op }, [ctrl, a, new_rhs], tys));
                }

                if changed {
                    return Some(self.new_node(ty, self[target].kind, [ctrl, lhs, rhs], tys));
                }
            }
            K::UnOp { op } => {
                let &[_, oper] = self[target].inputs.as_slice() else { unreachable!() };
                let ty = self[target].ty;

                if matches!(op, TokenKind::Number | TokenKind::Float)
                    && tys.size_of(self[oper].ty) == tys.size_of(ty)
                    && self[oper].ty.is_integer()
                    && ty.is_integer()
                {
                    return Some(oper);
                }

                if let K::CInt { value } = self[oper].kind {
                    let is_float = self[oper].ty.is_float();
                    return Some(self.new_const(ty, op.apply_unop(value, is_float)));
                }
            }
            K::If => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }

                if self[target].ty == ty::Id::VOID {
                    match self.try_match_cond(target) {
                        CondOptRes::Unknown => {}
                        CondOptRes::Known { value, .. } => {
                            let ty = if value {
                                ty::Id::RIGHT_UNREACHABLE
                            } else {
                                ty::Id::LEFT_UNREACHABLE
                            };
                            return Some(self.new_node_nop(ty, K::If, self[target].inputs.clone()));
                        }
                    }
                }
            }
            K::Then => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }

                if self[self[target].inputs[0]].ty == ty::Id::LEFT_UNREACHABLE {
                    return Some(NEVER);
                } else if self[self[target].inputs[0]].ty == ty::Id::RIGHT_UNREACHABLE {
                    return Some(self[self[target].inputs[0]].inputs[0]);
                }
            }
            K::Else => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }

                if self[self[target].inputs[0]].ty == ty::Id::RIGHT_UNREACHABLE {
                    return Some(NEVER);
                } else if self[self[target].inputs[0]].ty == ty::Id::LEFT_UNREACHABLE {
                    return Some(self[self[target].inputs[0]].inputs[0]);
                }
            }
            K::Region => {
                let (ctrl, side) = match self[target].inputs.as_slice() {
                    [NEVER, NEVER] => return Some(NEVER),
                    &[NEVER, ctrl] => (ctrl, 2),
                    &[ctrl, NEVER] => (ctrl, 1),
                    _ => return None,
                };

                self.lock(target);
                for i in self[target].outputs.clone() {
                    if self[i].kind == Kind::Phi {
                        self.replace(i, self[i].inputs[side]);
                    }
                }
                self.unlock(target);

                return Some(ctrl);
            }
            K::Call { .. } => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }
            }
            K::Return => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }

                let mut new_inps = Vc::from(&self[target].inputs[..2]);
                'a: for &n in self[target].inputs.clone().iter().skip(2) {
                    if self[n].kind != Kind::Stre {
                        new_inps.push(n);
                        continue;
                    }

                    if let Some(&load) =
                        self[n].outputs.iter().find(|&&n| self[n].kind == Kind::Load)
                    {
                        self.add_trigger(load, target);
                        continue;
                    }

                    let mut cursor = n;
                    let class = self.aclass_index(self[cursor].inputs[2]);

                    if self[class.1].kind != Kind::Stck {
                        new_inps.push(n);
                        continue;
                    }

                    if self[class.1].outputs.iter().any(|&n| {
                        self[n].kind != Kind::Stre
                            && self[n].outputs.iter().any(|&n| self[n].kind != Kind::Stre)
                    }) {
                        new_inps.push(n);
                        continue;
                    }

                    cursor = self[cursor].inputs[3];
                    while cursor != MEM {
                        debug_assert_eq!(self[cursor].kind, Kind::Stre);
                        if self.aclass_index(self[cursor].inputs[2]) != class {
                            new_inps.push(n);
                            continue 'a;
                        }

                        if let Some(&load) =
                            self[cursor].outputs.iter().find(|&&n| self[n].kind == Kind::Load)
                        {
                            self.add_trigger(load, target);
                            continue 'a;
                        }

                        cursor = self[cursor].inputs[3];
                    }
                }

                if new_inps.as_slice() != self[target].inputs.as_slice() {
                    return Some(self.new_node_nop(ty::Id::VOID, Kind::Return, new_inps));
                }
            }
            K::Phi => {
                let &[ctrl, lhs, rhs] = self[target].inputs.as_slice() else { unreachable!() };

                if rhs == target || lhs == rhs {
                    return Some(lhs);
                }

                if self[lhs].kind == Kind::Stre
                    && self[rhs].kind == Kind::Stre
                    && self[lhs].ty == self[rhs].ty
                    && self[lhs].ty.loc(tys) == Loc::Reg
                    && self[lhs].inputs[2] == self[rhs].inputs[2]
                    && self[lhs].inputs[3] == self[rhs].inputs[3]
                {
                    let pick_value = self.new_node(
                        self[lhs].ty,
                        Kind::Phi,
                        [ctrl, self[lhs].inputs[1], self[rhs].inputs[1]],
                        tys,
                    );
                    let mut vc = self[lhs].inputs.clone();
                    vc[1] = pick_value;
                    return Some(self.new_node(self[lhs].ty, Kind::Stre, vc, tys));
                }

                // broken
                //let ty = self[target].ty;
                //if let Kind::BinOp { op } = self[lhs].kind
                //    && self[rhs].kind == (Kind::BinOp { op })
                //{
                //    debug_assert!(ty != ty::Id::VOID);
                //    debug_assert_eq!(
                //        self[lhs].ty.simple_size(),
                //        ty.simple_size(),
                //        "{:?} {:?}",
                //        self[lhs].ty.expand(),
                //        ty.expand()
                //    );
                //    debug_assert_eq!(
                //        self[rhs].ty.simple_size(),
                //        ty.simple_size(),
                //        "{:?} {:?}",
                //        self[rhs].ty.expand(),
                //        ty.expand()
                //    );
                //    let inps = [ctrl, self[lhs].inputs[1], self[rhs].inputs[1]];
                //    let nlhs = self.new_node(ty, Kind::Phi, inps, tys);
                //    let inps = [ctrl, self[lhs].inputs[2], self[rhs].inputs[2]];
                //    let nrhs = self.new_node(ty, Kind::Phi, inps, tys);
                //    return Some(self.new_node(ty, Kind::BinOp { op }, [VOID, nlhs, nrhs], tys));
                //}
            }
            K::Stck => {
                if let &[mut a, mut b] = self[target].outputs.as_slice() {
                    if self[a].kind == Kind::Load {
                        mem::swap(&mut a, &mut b);
                    }

                    if self[a].kind.is_call()
                        && self[a].inputs.last() == Some(&target)
                        && self[b].kind == Kind::Load
                        && let &[store] = self[b].outputs.as_slice()
                        && self[store].kind == Kind::Stre
                    {
                        let len = self[a].inputs.len();
                        let stre = self[store].inputs[3];
                        if stre != MEM {
                            self[a].inputs.push(stre);
                            self[a].inputs.swap(len - 1, len);
                            self[stre].outputs.push(a);
                        }
                        return Some(self[store].inputs[2]);
                    }
                }
            }
            K::Stre => {
                let &[_, value, region, store, ..] = self[target].inputs.as_slice() else {
                    unreachable!()
                };

                if self[value].kind == Kind::Load && self[value].inputs[1] == region {
                    return Some(store);
                }

                let mut cursor = target;
                while self[cursor].kind == Kind::Stre
                    && self[cursor].inputs[1] != VOID
                    && let &[next_store] = self[cursor].outputs.as_slice()
                {
                    if self[next_store].inputs[2] == region
                        && self[next_store].ty == self[target].ty
                    {
                        return Some(store);
                    }
                    cursor = next_store;
                }

                'eliminate: {
                    if self[target].outputs.is_empty() {
                        break 'eliminate;
                    }

                    if self[value].kind != Kind::Load
                        || self[value].outputs.iter().any(|&n| self[n].kind != Kind::Stre)
                    {
                        for &ele in self[value].outputs.clone().iter().filter(|&&n| n != target) {
                            self.add_trigger(ele, target);
                        }
                        break 'eliminate;
                    }

                    let &[_, stack, last_store] = self[value].inputs.as_slice() else {
                        unreachable!()
                    };

                    if self[stack].ty != self[value].ty || self[stack].kind != Kind::Stck {
                        break 'eliminate;
                    }

                    let mut unidentifed = self[stack].outputs.clone();
                    let load_idx = unidentifed.iter().position(|&n| n == value).unwrap();
                    unidentifed.swap_remove(load_idx);

                    let mut saved = Vc::default();
                    let mut cursor = last_store;
                    let mut first_store = last_store;
                    while cursor != MEM && self[cursor].kind == Kind::Stre {
                        let mut contact_point = cursor;
                        let mut region = self[cursor].inputs[2];
                        if let Kind::BinOp { op } = self[region].kind {
                            debug_assert_matches!(op, TokenKind::Add | TokenKind::Sub);
                            contact_point = region;
                            region = self[region].inputs[1]
                        }

                        if region != stack {
                            break;
                        }
                        let Some(index) = unidentifed.iter().position(|&n| n == contact_point)
                        else {
                            break 'eliminate;
                        };
                        if self[self[cursor].inputs[1]].kind == Kind::Load
                            && self[value].outputs.iter().any(|&n| {
                                self.aclass_index(self[self[cursor].inputs[1]].inputs[1]).0
                                    == self.aclass_index(self[n].inputs[2]).0
                            })
                        {
                            break 'eliminate;
                        }
                        unidentifed.remove(index);
                        saved.push(contact_point);
                        first_store = cursor;
                        cursor = *self[cursor].inputs.get(3).unwrap_or(&MEM);

                        if unidentifed.is_empty() {
                            break;
                        }
                    }

                    if !unidentifed.is_empty() {
                        break 'eliminate;
                    }

                    debug_assert_matches!(
                        self[last_store].kind,
                        Kind::Stre | Kind::Mem,
                        "{:?}",
                        self[last_store]
                    );
                    debug_assert_matches!(
                        self[first_store].kind,
                        Kind::Stre | Kind::Mem,
                        "{:?}",
                        self[first_store]
                    );

                    // FIXME: when the loads and stores become parallel we will need to get saved
                    // differently
                    let mut prev_store = store;
                    for mut oper in saved.into_iter().rev() {
                        let mut region = region;
                        if let Kind::BinOp { op } = self[oper].kind {
                            debug_assert_eq!(self[oper].outputs.len(), 1);
                            debug_assert_eq!(self[self[oper].outputs[0]].kind, Kind::Stre);
                            let new_region = self.new_node(
                                self[oper].ty,
                                Kind::BinOp { op },
                                [VOID, region, self[oper].inputs[2]],
                                tys,
                            );
                            self.pass_aclass(self.aclass_index(region).1, new_region);
                            region = new_region;
                            oper = self[oper].outputs[0];
                        }

                        let mut inps = self[oper].inputs.clone();
                        debug_assert_eq!(inps.len(), 4);
                        inps[2] = region;
                        inps[3] = prev_store;
                        prev_store = self.new_node_nop(self[oper].ty, Kind::Stre, inps);
                        self.queued_peeps.push(prev_store);
                    }

                    return Some(prev_store);
                }

                if let Some(&load) =
                    self[target].outputs.iter().find(|&&n| self[n].kind == Kind::Load)
                {
                    self.add_trigger(load, target);
                } else if value != VOID
                    && self[value].kind != Kind::Load
                    && self[store].kind == Kind::Stre
                    && self[store].inputs[2] == region
                {
                    if self[store].inputs[1] == value {
                        return Some(store);
                    }

                    let mut inps = self[target].inputs.clone();
                    inps[3] = self[store].inputs[3];
                    return Some(self.new_node_nop(self[target].ty, Kind::Stre, inps));
                }
            }
            K::Load => {
                fn range_of(s: &Nodes, mut region: Nid, ty: ty::Id, tys: &Types) -> Range<usize> {
                    let loc = s.aclass_index(region).1;
                    let full_size = tys.size_of(
                        if matches!(s[loc].kind, Kind::Stck | Kind::Arg | Kind::Global { .. }) {
                            s[loc].ty
                        } else if let Some(ptr) = tys.base_of(s[loc].ty) {
                            ptr
                        } else {
                            return 0..usize::MAX;
                        },
                    );
                    let size = tys.size_of(ty);
                    let mut offset = 0;
                    loop {
                        match s[region].kind {
                            _ if region == loc => {
                                break offset as usize..offset as usize + size as usize
                            }
                            Kind::Assert { kind: AssertKind::NullCheck, .. } => {
                                region = s[region].inputs[2]
                            }
                            Kind::BinOp { op: TokenKind::Add | TokenKind::Sub }
                                if let Kind::CInt { value } = s[s[region].inputs[2]].kind =>
                            {
                                offset += value;
                                region = s[region].inputs[1];
                            }
                            _ => break 0..full_size as usize,
                        };
                    }
                }

                let &[ctrl, region, store] = self[target].inputs.as_slice() else { unreachable!() };
                let load_range = range_of(self, region, self[target].ty, tys);

                let mut cursor = store;
                while cursor != MEM && self[cursor].kind != Kind::Phi {
                    if self[cursor].inputs[0] == ctrl
                        && self[cursor].inputs[2] == region
                        && self[cursor].ty == self[target].ty
                        && (self[self[cursor].inputs[1]].kind != Kind::Load
                            || (!self[target].outputs.is_empty()
                                && self[target].outputs.iter().all(|&n| {
                                    self[n].kind != Kind::Stre
                                        || self
                                            .aclass_index(self[self[cursor].inputs[1]].inputs[1])
                                            .0
                                            != self.aclass_index(self[n].inputs[2]).0
                                })))
                    {
                        return Some(self[cursor].inputs[1]);
                    }
                    let range = range_of(self, self[cursor].inputs[2], self[cursor].ty, tys);
                    if range.start >= load_range.end || range.end <= load_range.start {
                        cursor = self[cursor].inputs[3];
                    } else {
                        let reg = self.aclass_index(self[cursor].inputs[2]).1;
                        self.add_trigger(reg, target);
                        break;
                    }
                }

                if store != cursor {
                    return Some(self.new_node(
                        self[target].ty,
                        Kind::Load,
                        [ctrl, region, cursor],
                        tys,
                    ));
                }
            }
            K::Loop => {
                if self[target].inputs[1] == NEVER || self[target].inputs[0] == NEVER {
                    self.lock(target);
                    for o in self[target].outputs.clone() {
                        if self[o].kind == Kind::Phi {
                            self.remove_node_lookup(target);

                            let prev = self[o].inputs[2];
                            self[o].inputs[2] = VOID;
                            self[VOID].outputs.push(o);
                            let index = self[prev].outputs.iter().position(|&n| n == o).unwrap();
                            self[prev].outputs.swap_remove(index);
                            self.lock(o);
                            self.remove(prev);
                            self.unlock(o);

                            self.replace(o, self[o].inputs[1]);
                        }
                    }
                    self.unlock(target);
                    return Some(self[target].inputs[0]);
                }
            }
            K::Die => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }
            }
            K::Assert { kind, .. } => 'b: {
                let pin = match (kind, self.try_match_cond(target)) {
                    (AssertKind::NullCheck, CondOptRes::Known { value: false, pin }) => pin,
                    (AssertKind::UnwrapCheck, CondOptRes::Unknown) => None,
                    _ => break 'b,
                }
                .unwrap_or(self[target].inputs[0]);

                for out in self[target].outputs.clone() {
                    if !self[out].kind.is_pinned() && self[out].inputs[0] != pin {
                        self.modify_input(out, 0, pin);
                    }
                }
                return Some(self[target].inputs[2]);
            }
            _ if self.is_cfg(target) && self.idom(target) == NEVER => panic!(),
            K::Start
            | K::Entry
            | K::Mem
            | K::Loops
            | K::End
            | K::CInt { .. }
            | K::Arg
            | K::Global { .. }
            | K::Join => {}
        }

        None
    }

    fn try_match_cond(&self, target: Nid) -> CondOptRes {
        let &[ctrl, cond, ..] = self[target].inputs.as_slice() else { unreachable!() };
        if let Kind::CInt { value } = self[cond].kind {
            return CondOptRes::Known { value: value != 0, pin: None };
        }

        let mut cursor = ctrl;
        while cursor != ENTRY {
            let ctrl = &self[cursor];
            // TODO: do more inteligent checks on the condition
            if matches!(ctrl.kind, Kind::Then | Kind::Else) {
                let other_cond = self[ctrl.inputs[0]].inputs[1];
                if let Some(value) = self.matches_cond(cond, other_cond) {
                    return CondOptRes::Known {
                        value: (ctrl.kind == Kind::Then) ^ !value,
                        pin: Some(cursor),
                    };
                }
            }

            cursor = self.idom(cursor);
        }

        CondOptRes::Unknown
    }

    fn matches_cond(&self, to_match: Nid, matches: Nid) -> Option<bool> {
        use TokenKind as K;
        let [tn, mn] = [&self[to_match], &self[matches]];
        match (tn.kind, mn.kind) {
            _ if to_match == matches => Some(true),
            (Kind::BinOp { op: K::Ne }, Kind::BinOp { op: K::Eq })
            | (Kind::BinOp { op: K::Eq }, Kind::BinOp { op: K::Ne })
                if tn.inputs[1..] == mn.inputs[1..] =>
            {
                Some(false)
            }
            (_, Kind::BinOp { op: K::Band }) => self
                .matches_cond(to_match, mn.inputs[1])
                .or(self.matches_cond(to_match, mn.inputs[2])),
            (_, Kind::BinOp { op: K::Bor }) => match (
                self.matches_cond(to_match, mn.inputs[1]),
                self.matches_cond(to_match, mn.inputs[2]),
            ) {
                (None, Some(a)) | (Some(a), None) => Some(a),
                (Some(b), Some(a)) if a == b => Some(a),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_const(&self, id: Nid) -> bool {
        matches!(self[id].kind, Kind::CInt { .. })
    }

    fn replace(&mut self, target: Nid, with: Nid) {
        debug_assert_ne!(target, with, "{:?}", self[target]);
        for out in self[target].outputs.clone() {
            let index = self[out].inputs.iter().position(|&p| p == target).unwrap();
            self.modify_input(out, index, with);
        }
    }

    fn modify_input(&mut self, target: Nid, inp_index: usize, with: Nid) -> Nid {
        self.remove_node_lookup(target);
        debug_assert_ne!(
            self[target].inputs[inp_index], with,
            "{:?} {:?}",
            self[target], self[with]
        );

        if self[target].is_not_gvnd() && (self[target].kind != Kind::Phi || with == 0) {
            let prev = self[target].inputs[inp_index];
            self[target].inputs[inp_index] = with;
            self[with].outputs.push(target);
            let index = self[prev].outputs.iter().position(|&o| o == target).unwrap();
            self[prev].outputs.swap_remove(index);
            self.remove(prev);
            target
        } else {
            let prev = self[target].inputs[inp_index];
            self[target].inputs[inp_index] = with;
            let (entry, hash) = self.lookup.entry(target.key(&self.values), &self.values);
            match entry {
                hash_map::RawEntryMut::Occupied(other) => {
                    let rpl = other.get_key_value().0.value;
                    self[target].inputs[inp_index] = prev;
                    self.lookup.insert(target.key(&self.values), target, &self.values);
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
    }

    #[track_caller]
    fn unlock_remove(&mut self, id: Nid) -> bool {
        self.unlock(id);
        self.remove(id)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = (Nid, &Node)> {
        self.values.iter().enumerate().filter_map(|(i, s)| Some((i as _, s.as_ref().ok()?)))
    }

    #[expect(clippy::format_in_format_args)]
    fn basic_blocks_instr(&mut self, out: &mut String, node: Nid) -> core::fmt::Result {
        match self[node].kind {
            Kind::Assert { .. } | Kind::Start => unreachable!("{} {out}", self[node].kind),
            Kind::End => return Ok(()),
            Kind::If => write!(out, "  if:      "),
            Kind::Region | Kind::Loop => writeln!(out, "      goto: {node}"),
            Kind::Return => write!(out, " ret:      "),
            Kind::Die => write!(out, " die:      "),
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
            Kind::Call { func, args: _ } => {
                write!(out, "call: {func} {}  ", self[node].depth.get())
            }
            Kind::Global { global } => write!(out, "glob: {global:<5}"),
            Kind::Entry => write!(out, "ctrl: {:<5}", "entry"),
            Kind::Then => write!(out, "ctrl: {:<5}", "then"),
            Kind::Else => write!(out, "ctrl: {:<5}", "else"),
            Kind::Stck => write!(out, "stck:      "),
            Kind::Load => write!(out, "load:      "),
            Kind::Stre => write!(out, "stre:      "),
            Kind::Mem => write!(out, " mem:      "),
            Kind::Loops => write!(out, "loops:      "),
            Kind::Join => write!(out, "join:     "),
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

    fn basic_blocks_low(
        &mut self,
        out: &mut String,
        mut node: Nid,
        visited: &mut BitSet,
    ) -> core::fmt::Result {
        let iter = |nodes: &Nodes, node| nodes[node].outputs.clone().into_iter().rev();
        while visited.set(node) {
            match self[node].kind {
                Kind::Start => {
                    writeln!(out, "start: {}", self[node].depth.get())?;
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
                    self.basic_blocks_low(out, self[node].outputs[0], visited)?;
                    node = self[node].outputs[1];
                }
                Kind::Region => {
                    writeln!(
                        out,
                        "region{node}: {} {} {:?}",
                        self[node].depth.get(),
                        self[node].loop_depth.get(),
                        self[node].inputs
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
                        self[node].depth.get(),
                        self[node].loop_depth.get(),
                        self[node].outputs
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
                Kind::Return | Kind::Die => {
                    node = self[node].outputs[0];
                }
                Kind::Then | Kind::Else | Kind::Entry => {
                    writeln!(
                        out,
                        "b{node}: {} {} {:?}",
                        self[node].depth.get(),
                        self[node].loop_depth.get(),
                        self[node].outputs
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
                            && (self[node].outputs[0] != o || mem::take(&mut print_ret))
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
        let mut visited = BitSet::default();
        self.basic_blocks_low(&mut out, VOID, &mut visited).unwrap();
        log::info!("{out}");
    }

    fn is_cfg(&self, o: Nid) -> bool {
        self[o].kind.is_cfg()
    }

    fn check_final_integrity(&self, disp: ty::Display) {
        if !cfg!(debug_assertions) {
            return;
        }

        let mut failed = false;
        for (id, node) in self.iter() {
            if self.is_locked(id) {
                log::error!("{} {} {:?}", node.lock_rc.get(), 0, node.kind);
                failed = true;
            }
            if !matches!(node.kind, Kind::End | Kind::Mem | Kind::Arg | Kind::Loops)
                && node.outputs.is_empty()
            {
                log::error!("outputs are empry {id} {:?}", node);
                failed = true;
            }
            if node.inputs.first() == Some(&NEVER) && id != NEVER {
                log::error!("is unreachable but still present {id} {:?}", node.kind);
                failed = true;
            }
            if node.outputs.contains(&id) && !matches!(node.kind, Kind::Loop | Kind::End) {
                log::error!("node depends on it self and its not a loop {id} {:?}", node);
                failed = true;
            }
        }

        if failed {
            self.graphviz_in_browser(disp);
            panic!()
        }
    }

    fn check_loop_depth_integrity(&self, disp: ty::Display) {
        if !cfg!(debug_assertions) {
            return;
        }

        let mut failed = false;
        for &loob in self[LOOPS].outputs.iter() {
            let mut stack = vec![self[loob].inputs[1]];
            let mut seen = BitSet::default();
            seen.set(loob);
            let depth = self.loop_depth(loob);
            while let Some(nid) = stack.pop() {
                if seen.set(nid) {
                    if depth > self.loop_depth(nid) {
                        failed = true;
                        log::error!("{depth} {} {nid} {:?}", self.loop_depth(nid), self[nid]);
                    }

                    match self[nid].kind {
                        Kind::Loop | Kind::Region => {
                            stack.extend(&self[nid].inputs[..2]);
                        }
                        _ => stack.push(self[nid].inputs[0]),
                    }
                }
            }
        }

        if failed {
            self.graphviz_in_browser(disp);
            panic!()
        }
    }

    fn load_loop_var(&mut self, index: usize, var: &mut Variable, loops: &mut [Loop]) {
        if var.value() != VOID {
            return;
        }

        debug_assert!(!var.ptr);

        let [loops @ .., loob] = loops else { unreachable!() };
        let node = loob.node;
        let lvar = &mut loob.scope.vars[index];

        debug_assert!(!lvar.ptr);

        self.load_loop_var(index, lvar, loops);

        if !self[lvar.value()].is_lazy_phi(node) {
            let lvalue = lvar.value();
            let inps = [node, lvalue, VOID];
            lvar.set_value(self.new_node_nop(lvar.ty, Kind::Phi, inps), self);

            self.pass_aclass(self.aclass_index(lvalue).1, lvar.value());
        }
        var.set_value(lvar.value(), self);
    }

    fn load_loop_aclass(&mut self, index: usize, aclass: &mut AClass, loops: &mut [Loop]) {
        if aclass.last_store.get() != VOID {
            return;
        }

        let [loops @ .., loob] = loops else { unreachable!() };
        let node = loob.node;
        let lvar = &mut loob.scope.aclasses[index];

        self.load_loop_aclass(index, lvar, loops);

        if !self[lvar.last_store.get()].is_lazy_phi(node) {
            let inps = [node, lvar.last_store.get(), VOID];
            lvar.last_store.set(self.new_node_nop(ty::Id::VOID, Kind::Phi, inps), self);
        }
        aclass.last_store.set(lvar.last_store.get(), self);
    }

    fn assert_dominance(&mut self, nd: Nid, min: Nid, check_outputs: bool) {
        if !cfg!(debug_assertions) {
            return;
        }

        let node = self[nd].clone();
        for &i in node.inputs.iter() {
            let dom = self.idom(i);
            debug_assert!(
                self.dominates(dom, min),
                "{dom} {min} {node:?} {:?}",
                self.basic_blocks()
            );
        }
        if check_outputs {
            for &o in node.outputs.iter() {
                let dom = self.use_block(nd, o);
                debug_assert!(
                    self.dominates(min, dom),
                    "{min} {dom} {node:?} {:?}",
                    self.basic_blocks()
                );
            }
        }
    }

    fn dominates(&self, dominator: Nid, mut dominated: Nid) -> bool {
        loop {
            if dominator == dominated {
                break true;
            }

            if self.idepth(dominator) > self.idepth(dominated) {
                break false;
            }

            dominated = self.idom(dominated);
        }
    }

    fn is_data_dep(&self, val: Nid, user: Nid) -> bool {
        match self[user].kind {
            Kind::Return => self[user].inputs[1] == val,
            _ if self.is_cfg(user) && !matches!(self[user].kind, Kind::Call { .. } | Kind::If) => {
                false
            }
            Kind::Join => false,
            Kind::Stre => self[user].inputs[3] != val,
            Kind::Load => self[user].inputs[2] != val,
            _ => self[user].inputs[0] != val || self[user].inputs[1..].contains(&val),
        }
    }

    fn this_or_delegates<'a>(&'a self, source: Nid, target: &'a Nid) -> (Nid, &'a [Nid]) {
        if self.is_unlocked(*target) {
            (source, core::slice::from_ref(target))
        } else {
            (*target, self[*target].outputs.as_slice())
        }
    }

    fn is_hard_zero(&self, nid: Nid) -> bool {
        self[nid].kind == Kind::CInt { value: 0 }
            && self[nid].outputs.iter().all(|&n| self[n].kind != Kind::Phi)
    }

    fn add_trigger(&mut self, blocker: Nid, target: Nid) {
        if !self[blocker].peep_triggers.contains(&target) {
            self[blocker].peep_triggers.push(target);
        }
    }
}

enum CondOptRes {
    Unknown,
    Known { value: bool, pin: Option<Nid> },
}

impl ops::Index<Nid> for Nodes {
    type Output = Node;

    fn index(&self, index: Nid) -> &Self::Output {
        self.values[index as usize].as_ref().unwrap_or_else(|(_, bt)| panic!("{index} {bt:#?}"))
    }
}

impl ops::IndexMut<Nid> for Nodes {
    fn index_mut(&mut self, index: Nid) -> &mut Self::Output {
        self.values[index as usize].as_mut().unwrap_or_else(|(_, bt)| panic!("{index} {bt:#?}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AssertKind {
    NullCheck,
    UnwrapCheck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Kind {
    #[default]
    Start,
    // [ctrl]
    Entry,
    // [VOID]
    Mem,
    // [VOID]
    Loops,
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
    Die,
    // [ctrl]
    CInt {
        value: i64,
    },
    // [ctrl, lhs, rhs]
    Phi,
    Arg,
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
        args: ty::Tuple,
    },
    // [ctrl, cond, value]
    Assert {
        kind: AssertKind,
        pos: Pos,
    },
    // [ctrl]
    Stck,
    // [ctrl, memory]
    Load,
    // [ctrl, value, memory]
    Stre,
    // [ctrl, a, b]
    Join,
}

impl Kind {
    fn is_call(&self) -> bool {
        matches!(self, Kind::Call { .. })
    }

    fn is_eca(&self) -> bool {
        matches!(self, Kind::Call { func: ty::Func::ECA, .. })
    }

    fn is_pinned(&self) -> bool {
        self.is_cfg()
            || matches!(self, Self::Phi | Self::Arg | Self::Mem | Self::Loops | Kind::Assert { .. })
    }

    fn is_cfg(&self) -> bool {
        matches!(
            self,
            Self::Start
                | Self::End
                | Self::Return
                | Self::Die
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
        matches!(self, Self::Return | Self::If | Self::End | Self::Die)
    }

    fn starts_basic_block(&self) -> bool {
        matches!(self, Self::Region | Self::Loop | Self::Start | Kind::Then | Kind::Else)
    }

    fn is_peeped(&self) -> bool {
        !matches!(self, Self::End | Self::Arg | Self::Mem | Self::Loops)
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
pub struct Node {
    kind: Kind,
    inputs: Vc,
    outputs: Vc,
    peep_triggers: Vc,
    clobbers: BitSet,
    ty: ty::Id,
    pos: Pos,
    depth: Cell<IDomDepth>,
    lock_rc: Cell<LockRc>,
    loop_depth: Cell<LoopDepth>,
    aclass: AClassId,
}

impl Node {
    fn is_dangling(&self) -> bool {
        self.outputs.is_empty() && self.lock_rc.get() == 0 && self.kind != Kind::Arg
    }

    fn key(&self) -> (Kind, &[Nid], ty::Id) {
        (self.kind, &self.inputs, self.ty)
    }

    fn is_lazy_phi(&self, loob: Nid) -> bool {
        self.kind == Kind::Phi && self.inputs[2] == 0 && self.inputs[0] == loob
    }

    fn is_not_gvnd(&self) -> bool {
        (self.kind == Kind::Phi && self.inputs[2] == 0)
            || matches!(self.kind, Kind::Arg | Kind::Stck | Kind::Stre)
            || self.kind.is_cfg()
    }

    fn is_mem(&self) -> bool {
        matches!(self.kind, Kind::Stre | Kind::Load | Kind::Stck)
    }

    fn is_data_phi(&self) -> bool {
        self.kind == Kind::Phi && self.ty != ty::Id::VOID
    }

    fn has_no_value(&self) -> bool {
        (self.kind.is_cfg() && (!self.kind.is_call() || self.ty == ty::Id::VOID))
            || matches!(self.kind, Kind::Stre)
    }
}

type LoopDepth = u16;
type LockRc = u16;
type IDomDepth = u16;

#[derive(Clone)]
struct Loop {
    node: Nid,
    ctrl: [StrongRef; 2],
    ctrl_scope: [Scope; 2],
    scope: Scope,
}

mod strong_ref {
    use {
        super::{Kind, Nid, Nodes},
        crate::debug,
        core::ops::Not,
    };

    #[derive(Clone)]
    pub struct StrongRef(Nid);

    impl StrongRef {
        pub const DEFAULT: Self = Self(Nid::MAX);

        pub fn new(value: Nid, nodes: &mut Nodes) -> Self {
            nodes.lock(value);
            Self(value)
        }

        pub fn get(&self) -> Nid {
            debug_assert!(self.0 != Nid::MAX);
            self.0
        }

        pub fn unwrap(self, nodes: &mut Nodes) -> Option<Nid> {
            let nid = self.0;
            if nid != Nid::MAX {
                nodes.unlock(nid);
                core::mem::forget(self);
                Some(nid)
            } else {
                None
            }
        }

        pub fn set(&mut self, mut new_value: Nid, nodes: &mut Nodes) -> Nid {
            nodes.unlock(self.0);
            core::mem::swap(&mut self.0, &mut new_value);
            nodes.lock(self.0);
            new_value
        }

        pub fn dup(&self, nodes: &mut Nodes) -> Self {
            nodes.lock(self.0);
            Self(self.0)
        }

        pub fn remove(self, nodes: &mut Nodes) -> Option<Nid> {
            let ret = nodes.unlock_remove(self.0).not().then_some(self.0);
            core::mem::forget(self);
            ret
        }

        pub fn set_remove(&mut self, new_value: Nid, nodes: &mut Nodes) {
            let old = self.set(new_value, nodes);
            nodes.remove(old);
        }

        pub fn remove_ignore_arg(self, nodes: &mut Nodes) {
            if nodes[self.0].kind == Kind::Arg {
                nodes.unlock(self.0);
            } else {
                nodes.unlock_remove(self.0);
            }
            core::mem::forget(self);
        }

        pub fn soft_remove(self, nodes: &mut Nodes) -> Nid {
            let nid = self.0;
            nodes.unlock(self.0);
            core::mem::forget(self);
            nid
        }

        pub fn is_live(&self) -> bool {
            self.0 != Nid::MAX
        }
    }

    impl Default for StrongRef {
        fn default() -> Self {
            Self::DEFAULT
        }
    }

    impl Drop for StrongRef {
        fn drop(&mut self) {
            if self.0 != Nid::MAX && !debug::panicking() {
                panic!("variable unproperly deinitialized")
            }
        }
    }
}

// makes sure value inside is laways locked for this instance of variable
#[derive(Default, Clone)]
struct Variable {
    id: Ident,
    ty: ty::Id,
    ptr: bool,
    value: StrongRef,
}

impl Variable {
    fn new(id: Ident, ty: ty::Id, ptr: bool, value: Nid, nodes: &mut Nodes) -> Self {
        Self { id, ty, ptr, value: StrongRef::new(value, nodes) }
    }

    fn value(&self) -> Nid {
        self.value.get()
    }

    fn set_value(&mut self, new_value: Nid, nodes: &mut Nodes) -> Nid {
        self.value.set(new_value, nodes)
    }

    fn dup(&self, nodes: &mut Nodes) -> Self {
        Self { id: self.id, ty: self.ty, ptr: self.ptr, value: self.value.dup(nodes) }
    }

    fn remove(self, nodes: &mut Nodes) {
        self.value.remove(nodes);
    }

    fn set_value_remove(&mut self, new_value: Nid, nodes: &mut Nodes) {
        self.value.set_remove(new_value, nodes);
    }

    fn remove_ignore_arg(self, nodes: &mut Nodes) {
        self.value.remove_ignore_arg(nodes);
    }
}

#[derive(Default, Clone)]
pub struct AClass {
    last_store: StrongRef,
    clobber: StrongRef,
}

impl AClass {
    fn dup(&self, nodes: &mut Nodes) -> Self {
        Self { last_store: self.last_store.dup(nodes), clobber: self.clobber.dup(nodes) }
    }

    fn remove(self, nodes: &mut Nodes) {
        self.last_store.remove(nodes);
        self.clobber.remove(nodes);
    }

    fn new(nodes: &mut Nodes) -> Self {
        Self { last_store: StrongRef::new(MEM, nodes), clobber: StrongRef::new(VOID, nodes) }
    }
}

#[derive(Default, Clone)]
pub struct Scope {
    vars: Vec<Variable>,
    aclasses: Vec<AClass>,
}

impl Scope {
    fn dup(&self, nodes: &mut Nodes) -> Self {
        Self {
            vars: self.vars.iter().map(|v| v.dup(nodes)).collect(),
            aclasses: self.aclasses.iter().map(|v| v.dup(nodes)).collect(),
        }
    }

    fn clear(&mut self, nodes: &mut Nodes) {
        self.vars.drain(..).for_each(|n| n.remove(nodes));
        self.aclasses.drain(..).for_each(|l| l.remove(nodes));
    }
}

#[derive(Default, Clone)]
pub struct ItemCtx {
    file: Module,
    pos: Vec<Pos>,
    ret: Option<ty::Id>,
    task_base: usize,
    inline_var_base: usize,
    inline_aclass_base: usize,
    inline_depth: u16,
    inline_ret: Option<(Value, StrongRef, Scope, Option<AClass>)>,
    nodes: Nodes,
    ctrl: StrongRef,
    loops: Vec<Loop>,
    scope: Scope,
}

impl ItemCtx {
    fn init(&mut self, file: Module, ret: Option<ty::Id>, task_base: usize) {
        debug_assert_eq!(self.loops.len(), 0);
        debug_assert_eq!(self.scope.vars.len(), 0);
        debug_assert_eq!(self.scope.aclasses.len(), 0);
        debug_assert!(self.inline_ret.is_none());
        debug_assert_eq!(self.inline_depth, 0);
        debug_assert_eq!(self.inline_var_base, 0);
        debug_assert_eq!(self.inline_aclass_base, 0);

        self.file = file;
        self.ret = ret;
        self.task_base = task_base;

        self.nodes.clear();
        self.scope.vars.clear();

        let start = self.nodes.new_node_nop(ty::Id::VOID, Kind::Start, []);
        debug_assert_eq!(start, VOID);
        let end = self.nodes.new_node_nop(ty::Id::NEVER, Kind::End, []);
        debug_assert_eq!(end, NEVER);
        self.nodes.lock(end);
        self.ctrl = StrongRef::new(
            self.nodes.new_node_nop(ty::Id::VOID, Kind::Entry, [VOID]),
            &mut self.nodes,
        );
        debug_assert_eq!(self.ctrl.get(), ENTRY);
        let mem = self.nodes.new_node_nop(ty::Id::VOID, Kind::Mem, [VOID]);
        debug_assert_eq!(mem, MEM);
        self.nodes.lock(mem);
        let loops = self.nodes.new_node_nop(ty::Id::VOID, Kind::Loops, [VOID]);
        debug_assert_eq!(loops, LOOPS);
        self.nodes.lock(loops);
        self.scope.aclasses.push(AClass::new(&mut self.nodes)); // DEFAULT
        self.scope.aclasses.push(AClass::new(&mut self.nodes)); // GLOBAL
    }

    fn finalize(&mut self, stack: &mut Vec<Nid>, tys: &Types, _files: &[parser::Ast]) {
        self.scope.clear(&mut self.nodes);
        mem::take(&mut self.ctrl).soft_remove(&mut self.nodes);

        self.nodes.iter_peeps(1000, stack, tys);

        self.nodes.unlock(MEM);
        self.nodes.unlock(NEVER);
        self.nodes.unlock(LOOPS);
    }
}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
}

#[derive(Default, Debug)]
struct Ctx {
    ty: Option<ty::Id>,
}

impl Ctx {
    fn with_ty(self, ty: ty::Id) -> Self {
        Self { ty: Some(ty) }
    }
}

#[derive(Default)]
pub struct Pool {
    cis: Vec<ItemCtx>,
    used_cis: usize,
    nid_stack: Vec<Nid>,
    nid_set: BitSet,
}

impl Pool {
    fn push_ci(
        &mut self,
        file: Module,
        ret: Option<ty::Id>,
        task_base: usize,
        target: &mut ItemCtx,
    ) {
        if let Some(slot) = self.cis.get_mut(self.used_cis) {
            mem::swap(slot, target);
        } else {
            self.cis.push(ItemCtx::default());
            mem::swap(self.cis.last_mut().unwrap(), target);
        }
        target.init(file, ret, task_base);
        self.used_cis += 1;
    }

    fn pop_ci(&mut self, target: &mut ItemCtx) {
        self.used_cis -= 1;
        mem::swap(&mut self.cis[self.used_cis], target);
    }

    fn save_ci(&mut self, ci: &ItemCtx) {
        if let Some(slot) = self.cis.get_mut(self.used_cis) {
            slot.clone_from(ci);
        } else {
            self.cis.push(ci.clone());
        }
        self.used_cis += 1;
    }

    fn restore_ci(&mut self, dst: &mut ItemCtx) {
        self.used_cis -= 1;
        dst.scope.clear(&mut dst.nodes);
        mem::take(&mut dst.ctrl).remove(&mut dst.nodes);
        *dst = mem::take(&mut self.cis[self.used_cis]);
    }

    fn clear(&mut self) {
        debug_assert_eq!(self.used_cis, 0);
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

    fn new(id: Nid) -> Self {
        Self { id, ..Default::default() }
    }

    fn var(id: usize) -> Self {
        Self { id: u16::MAX - (id as Nid), var: true, ..Default::default() }
    }

    fn ptr(id: Nid) -> Self {
        Self { id, ptr: true, ..Default::default() }
    }

    #[inline(always)]
    fn ty(self, ty: ty::Id) -> Self {
        Self { ty, ..self }
    }
}

#[derive(Default)]
pub struct CodegenCtx {
    pub parser: parser::Ctx,
    tys: Types,
    pool: Pool,
    ct: Comptime,
    ct_backend: HbvmBackend,
}

impl CodegenCtx {
    pub fn clear(&mut self) {
        self.parser.clear();
        self.tys.clear();
        self.pool.clear();
        self.ct.clear();
    }
}

pub struct Codegen<'a> {
    pub files: &'a [parser::Ast],
    pub errors: &'a RefCell<String>,
    tys: &'a mut Types,
    ci: ItemCtx,
    pool: &'a mut Pool,
    ct: &'a mut Comptime,
    ct_backend: &'a mut HbvmBackend,
    backend: &'a mut dyn Backend,
}

impl Drop for Codegen<'_> {
    fn drop(&mut self) {
        if debug::panicking() {
            if let Some(&pos) = self.ci.pos.last() {
                self.report(pos, "panic occured here");
            }

            if !self.errors.borrow().is_empty() {
                log::error!("{}", self.errors.borrow());
            }
        }
    }
}

impl<'a> Codegen<'a> {
    pub fn new(
        backend: &'a mut dyn Backend,
        files: &'a [parser::Ast],
        ctx: &'a mut CodegenCtx,
    ) -> Self {
        Self {
            files,
            errors: &ctx.parser.errors,
            tys: &mut ctx.tys,
            ci: Default::default(),
            pool: &mut ctx.pool,
            ct: &mut ctx.ct,
            ct_backend: &mut ctx.ct_backend,
            backend,
        }
    }

    pub fn generate(&mut self, entry: Module) {
        self.find_type(0, entry, entry, Err("main"), self.files);
        if self.tys.ins.funcs.is_empty() {
            return;
        }
        self.make_func_reachable(ty::Func::MAIN);
        self.complete_call_graph();
    }

    pub fn assemble_comptime(&mut self) -> Comptime {
        self.ct.code.clear();
        self.backend.assemble_bin(ty::Func::MAIN, self.tys, &mut self.ct.code);
        self.ct.reset();
        core::mem::take(self.ct)
    }

    pub fn assemble(&mut self, buf: &mut Vec<u8>) {
        self.backend.assemble_bin(ty::Func::MAIN, self.tys, buf);
    }

    pub fn disasm(&mut self, output: &mut String, bin: &[u8]) -> Result<(), DisasmError> {
        self.backend.disasm(bin, &mut |_| {}, self.tys, self.files, output)
    }

    pub fn push_embeds(&mut self, embeds: Vec<Vec<u8>>) {
        for data in embeds {
            let g = Global {
                ty: self.tys.make_array(ty::Id::U8, data.len() as _),
                data,
                ..Default::default()
            };
            self.tys.ins.globals.push(g);
        }
    }

    fn emit_and_eval(&mut self, file: Module, ret: ty::Id, ret_loc: &mut [u8]) -> u64 {
        let mut rets =
            self.ci.nodes[NEVER].inputs.iter().filter(|&&i| self.ci.nodes[i].kind == Kind::Return);
        if let Some(&ret) = rets.next()
            && rets.next().is_none()
            && let Kind::CInt { value } = self.ci.nodes[self.ci.nodes[ret].inputs[1]].kind
        {
            if let len @ 1..=8 = ret_loc.len() {
                ret_loc.copy_from_slice(&value.to_ne_bytes()[..len])
            }
            return value as _;
        }

        if !self.complete_call_graph() {
            return 1;
        }

        let fuc = self.tys.ins.funcs.push(Func {
            file,
            sig: Some(Sig { args: Tuple::empty(), ret }),
            ..Default::default()
        });

        self.ct_backend.emit_ct_body(fuc, &mut self.ci.nodes, self.tys, self.files);

        // TODO: return them back

        let entry = self.ct_backend.assemble_reachable(fuc, self.tys, &mut self.ct.code).entry;

        #[cfg(debug_assertions)]
        {
            let mut vc = String::new();
            if let Err(e) =
                self.ct_backend.disasm(&self.ct.code, &mut |_| {}, self.tys, self.files, &mut vc)
            {
                panic!("{e} {}", vc);
            } else {
                log::info!("{}", vc);
            }
        }

        self.ct.run(ret_loc, entry)
    }

    fn new_stack(&mut self, pos: Pos, ty: ty::Id) -> Nid {
        let stck = self.ci.nodes.new_node_nop(ty, Kind::Stck, [VOID, MEM]);
        self.ci.nodes[stck].aclass = self.ci.scope.aclasses.len() as _;
        self.ci.nodes[stck].pos = pos;
        self.ci.scope.aclasses.push(AClass::new(&mut self.ci.nodes));
        stck
    }

    fn store_mem(&mut self, region: Nid, ty: ty::Id, value: Nid) -> Nid {
        if value == NEVER {
            return NEVER;
        }

        debug_assert!(
            self.ci.nodes[region].kind != Kind::Load || self.ci.nodes[region].ty.is_pointer()
        );
        debug_assert!(self.ci.nodes[region].kind != Kind::Stre);

        let (value_index, value_region) = self.ci.nodes.aclass_index(value);
        if value_index != 0 {
            self.ci.nodes[value_region].aclass = 0;
            self.ci.nodes.load_loop_aclass(0, &mut self.ci.scope.aclasses[0], &mut self.ci.loops);
            self.ci.nodes.load_loop_aclass(
                value_index,
                &mut self.ci.scope.aclasses[value_index],
                &mut self.ci.loops,
            );
            let base_class = self.ci.scope.aclasses[0].last_store.get();
            let last_store = self.ci.scope.aclasses[value_index].last_store.get();
            match [base_class, last_store] {
                [_, MEM] => {}
                [MEM, a] => {
                    self.ci.scope.aclasses[0].last_store.set(a, &mut self.ci.nodes);
                }
                [a, b] => {
                    let a = self.ci.nodes.new_node_nop(ty::Id::VOID, Kind::Join, [0, a, b]);
                    self.ci.scope.aclasses[0].last_store.set(a, &mut self.ci.nodes);
                }
            }
        }

        let (index, _) = self.ci.nodes.aclass_index(region);
        if self.ci.nodes[value].kind == Kind::Load {
            let (lindex, ..) = self.ci.nodes.aclass_index(self.ci.nodes[value].inputs[1]);
            let clobber = self.ci.scope.aclasses[lindex].clobber.get();
            if self.ci.nodes.idepth(clobber)
                > self.ci.nodes.idepth(self.ci.scope.aclasses[index].clobber.get())
            {
                self.ci.scope.aclasses[index].clobber.set(clobber, &mut self.ci.nodes);
            }
        }
        let aclass = &mut self.ci.scope.aclasses[index];
        self.ci.nodes.load_loop_aclass(index, aclass, &mut self.ci.loops);
        let vc = Vc::from([aclass.clobber.get(), value, region, aclass.last_store.get()]);
        mem::take(&mut aclass.last_store).soft_remove(&mut self.ci.nodes);
        let store = self.ci.nodes.new_node(ty, Kind::Stre, vc, self.tys);
        aclass.last_store = StrongRef::new(store, &mut self.ci.nodes);
        store
    }

    fn load_mem(&mut self, region: Nid, ty: ty::Id) -> Nid {
        debug_assert_ne!(region, VOID);
        debug_assert_ne!({ self.ci.nodes[region].ty }, ty::Id::VOID, "{:?}", {
            self.ci.nodes[region].lock_rc.set(Nid::MAX);
            self.ci.nodes.graphviz_in_browser(self.ty_display(ty::Id::VOID));
        });
        debug_assert!(
            self.ci.nodes[region].kind != Kind::Load
                || self.ci.nodes[region].kind == Kind::Stck
                || self.ci.nodes[region].ty.is_pointer(),
            "{:?} {} {}",
            self.ci.nodes.graphviz_in_browser(self.ty_display(ty::Id::VOID)),
            self.file().path,
            self.ty_display(self.ci.nodes[region].ty)
        );
        debug_assert!(self.ci.nodes[region].kind != Kind::Stre);
        let (index, _) = self.ci.nodes.aclass_index(region);
        let aclass = &mut self.ci.scope.aclasses[index];
        self.ci.nodes.load_loop_aclass(index, aclass, &mut self.ci.loops);
        let vc = [aclass.clobber.get(), region, aclass.last_store.get()];
        self.ci.nodes.new_node(ty, Kind::Load, vc, self.tys)
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let state_slot = self.ct.active() as usize;
        let fuc = &mut self.tys.ins.funcs[func];
        if fuc.comp_state[state_slot] == CompState::Dead {
            fuc.comp_state[state_slot] = CompState::Queued(self.tys.tasks.len() as _);
            self.tys.tasks.push(Some(FTask { file: fuc.file, id: func, ct: self.ct.active() }));
        }
    }

    fn raw_expr(&mut self, expr: &Expr) -> Option<Value> {
        self.raw_expr_ctx(expr, Ctx::default())
    }

    fn raw_expr_ctx(&mut self, expr: &Expr, ctx: Ctx) -> Option<Value> {
        self.ci.pos.push(expr.pos());
        let res = self.raw_expr_ctx_low(expr, ctx);
        self.ci.pos.pop().unwrap();
        res
    }

    fn raw_expr_ctx_low(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Value> {
        // ordered by complexity of the expression
        match *expr {
            Expr::Null { pos } => {
                inference!(oty, ctx, self, pos, "null pointer", "@as(^<ty>, null)");

                let Some(ty) = self.tys.inner_of(oty) else {
                    self.report(
                        pos,
                        fa!(
                            "'null' expression was inferred to be '{}',
                            which is not optional",
                            self.ty_display(oty)
                        ),
                    );
                    return Value::NEVER;
                };

                match oty.loc(self.tys) {
                    Loc::Reg => Some(self.ci.nodes.new_const_lit(oty, 0)),
                    Loc::Stack => {
                        let OptLayout { flag_ty, flag_offset, .. } = self.tys.opt_layout(ty);
                        let stack = self.new_stack(pos, oty);
                        let offset = self.offset(stack, flag_offset);
                        let value = self.ci.nodes.new_const(flag_ty, 0);
                        self.store_mem(offset, flag_ty, value);
                        Some(Value::ptr(stack).ty(oty))
                    }
                }
            }
            Expr::Idk { pos } => {
                inference!(ty, ctx, self, pos, "value", "@as(<ty>, idk)");

                if ty.loc(self.tys) == Loc::Stack {
                    Some(Value::ptr(self.new_stack(pos, ty)).ty(ty))
                } else {
                    Some(self.ci.nodes.new_const_lit(ty, 0))
                }
            }
            Expr::Bool { value, .. } => Some(self.ci.nodes.new_const_lit(ty::Id::BOOL, value)),
            Expr::Number { value, .. }
                if let Some(ty) = ctx.ty
                    && ty.is_float() =>
            {
                Some(self.ci.nodes.new_const_lit(ty, (value as f64).to_bits() as i64))
            }
            Expr::Number { value, .. } => {
                self.gen_inferred_const(ctx, ty::Id::DINT, value, ty::Id::is_integer)
            }
            Expr::Float { value, .. } => {
                self.gen_inferred_const(ctx, ty::Id::F32, value as i64, ty::Id::is_float)
            }
            Expr::Ident { id, .. }
                if let Some(index) = self.ci.scope.vars.iter().rposition(|v| v.id == id) =>
            {
                let var = &mut self.ci.scope.vars[index];
                self.ci.nodes.load_loop_var(index, var, &mut self.ci.loops);

                Some(Value::var(index).ty(var.ty))
            }
            Expr::Ident { id, pos, .. } => {
                let decl = self.find_type(pos, self.ci.file, self.ci.file, Ok(id), self.files);
                match decl.expand() {
                    ty::Kind::NEVER => Value::NEVER,
                    ty::Kind::Global(global) => self.gen_global(global),
                    ty::Kind::Const(cnst) => self.gen_const(cnst, ctx),
                    _ => Some(Value::new(Nid::MAX).ty(decl)),
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

                let ty = self.tys.make_ptr(ty::Id::U8);
                let global = match self.tys.strings.entry(&data, &self.tys.ins.globals) {
                    (hash_map::RawEntryMut::Occupied(occupied_entry), _) => {
                        occupied_entry.get_key_value().0.value.0
                    }
                    (hash_map::RawEntryMut::Vacant(vacant_entry), hash) => {
                        let global =
                            self.tys.ins.globals.push(Global { data, ty, ..Default::default() });
                        vacant_entry
                            .insert(crate::ctx_map::Key { value: StringRef(global), hash }, ())
                            .0
                            .value
                            .0
                    }
                };
                let global = self.ci.nodes.new_node_nop(ty, Kind::Global { global }, [VOID]);
                self.ci.nodes[global].aclass = GLOBAL_ACLASS as _;
                Some(Value::new(global).ty(ty))
            }
            Expr::Return { pos, val } => {
                let mut value = if let Some(val) = val {
                    self.raw_expr_ctx(val, Ctx { ty: self.ci.ret })?
                } else {
                    Value { ty: ty::Id::VOID, ..Default::default() }
                };
                self.strip_var(&mut value);

                let expected = *self.ci.ret.get_or_insert(value.ty);
                self.assert_ty(pos, &mut value, expected, "return value");
                self.strip_ptr(&mut value);

                if self.ci.inline_depth == 0 {
                    debug_assert_ne!(self.ci.ctrl.get(), VOID);
                    let mut inps = Vc::from([self.ci.ctrl.get(), value.id]);
                    for (i, aclass) in self.ci.scope.aclasses.iter_mut().enumerate() {
                        self.ci.nodes.load_loop_aclass(i, aclass, &mut self.ci.loops);
                        if aclass.last_store.get() != MEM {
                            inps.push(aclass.last_store.get());
                        }
                    }

                    let ret = self.ci.nodes.new_node_nop(ty::Id::VOID, Kind::Return, inps);
                    self.ci.ctrl.set(NEVER, &mut self.ci.nodes);
                    self.ci.nodes[ret].pos = pos;
                    self.ci.nodes.bind(ret, NEVER);
                } else if let Some((pv, ctrl, scope, aclass)) = &mut self.ci.inline_ret {
                    debug_assert!(
                        aclass.is_none(),
                        "TODO: oh no, we cant return structs from divergent branches"
                    );
                    ctrl.set(
                        self.ci.nodes.new_node(
                            ty::Id::VOID,
                            Kind::Region,
                            [self.ci.ctrl.get(), ctrl.get()],
                            self.tys,
                        ),
                        &mut self.ci.nodes,
                    );
                    self.ci.nodes.merge_scopes(
                        &mut self.ci.loops,
                        ctrl,
                        scope,
                        &mut self.ci.scope,
                        self.tys,
                    );
                    self.ci.nodes.unlock(pv.id);
                    pv.id = self.ci.nodes.new_node(
                        value.ty,
                        Kind::Phi,
                        [ctrl.get(), value.id, pv.id],
                        self.tys,
                    );
                    self.ci.nodes.lock(pv.id);
                    self.ci.ctrl.set(NEVER, &mut self.ci.nodes);
                } else {
                    for (i, aclass) in self.ci.scope.aclasses[..2].iter_mut().enumerate() {
                        self.ci.nodes.load_loop_aclass(i, aclass, &mut self.ci.loops);
                    }

                    self.ci.nodes.lock(value.id);
                    let mut scope = self.ci.scope.dup(&mut self.ci.nodes);
                    scope
                        .vars
                        .drain(self.ci.inline_var_base..)
                        .for_each(|v| v.remove(&mut self.ci.nodes));
                    scope
                        .aclasses
                        .drain(self.ci.inline_aclass_base..)
                        .for_each(|v| v.remove(&mut self.ci.nodes));

                    let repl = StrongRef::new(NEVER, &mut self.ci.nodes);
                    let (index, _) = self
                        .ci
                        .nodes
                        .aclass_index(*self.ci.nodes[value.id].inputs.get(1).unwrap_or(&VOID));
                    let aclass = (self.ci.inline_aclass_base <= index)
                        .then(|| self.ci.scope.aclasses[index].dup(&mut self.ci.nodes));
                    self.ci.inline_ret =
                        Some((value, mem::replace(&mut self.ci.ctrl, repl), scope, aclass));
                }

                None
            }
            Expr::Die { .. } => {
                self.ci.ctrl.set(
                    self.ci.nodes.new_node_nop(ty::Id::VOID, Kind::Die, [self.ci.ctrl.get()]),
                    &mut self.ci.nodes,
                );

                self.ci.nodes[NEVER].inputs.push(self.ci.ctrl.get());
                self.ci.nodes[self.ci.ctrl.get()].outputs.push(NEVER);
                None
            }
            Expr::Field { target, name, pos } => {
                let mut vtarget = self.raw_expr(target)?;
                self.strip_var(&mut vtarget);
                self.implicit_unwrap(pos, &mut vtarget);
                let tty = vtarget.ty;

                if let ty::Kind::Module(m) = tty.expand() {
                    return match self
                        .find_type(pos, self.ci.file, m, Err(name), self.files)
                        .expand()
                    {
                        ty::Kind::NEVER => Value::NEVER,
                        ty::Kind::Global(global) => self.gen_global(global),
                        ty::Kind::Const(cnst) => self.gen_const(cnst, ctx),
                        v => Some(Value::new(Nid::MAX).ty(v.compress())),
                    };
                }

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

                let Some((offset, ty)) = OffsetIter::offset_of(self.tys, s, name) else {
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
            Expr::UnOp { op: TokenKind::Band, val, pos } => {
                let ctx = Ctx { ty: ctx.ty.and_then(|ty| self.tys.base_of(ty)) };

                let mut val = self.raw_expr_ctx(val, ctx)?;
                self.strip_var(&mut val);

                if val.ptr {
                    val.ptr = false;
                    val.ty = self.tys.make_ptr(val.ty);
                    return Some(val);
                }

                let stack = self.new_stack(pos, val.ty);
                self.store_mem(stack, val.ty, val.id);

                Some(Value::new(stack).ty(self.tys.make_ptr(val.ty)))
            }
            Expr::UnOp { op: TokenKind::Mul, val, pos } => {
                let ctx = Ctx { ty: ctx.ty.map(|ty| self.tys.make_ptr(ty)) };
                let mut vl = self.expr_ctx(val, ctx)?;

                self.implicit_unwrap(val.pos(), &mut vl);

                let Some(base) = self.tys.base_of(vl.ty) else {
                    self.report(
                        pos,
                        fa!("the '{}' can not be dereferneced", self.ty_display(vl.ty)),
                    );
                    return Value::NEVER;
                };
                vl.ptr = true;
                vl.ty = base;
                Some(vl)
            }
            Expr::UnOp { pos, op: op @ TokenKind::Sub, val } => {
                let val =
                    self.expr_ctx(val, Ctx::default().with_ty(ctx.ty.unwrap_or(ty::Id::INT)))?;
                if val.ty.is_integer() {
                    Some(self.ci.nodes.new_node_lit(
                        val.ty,
                        Kind::UnOp { op },
                        [VOID, val.id],
                        self.tys,
                    ))
                } else if val.ty.is_float() {
                    let value = self.ci.nodes.new_const(val.ty, (-1f64).to_bits() as i64);
                    Some(self.ci.nodes.new_node_lit(
                        val.ty,
                        Kind::BinOp { op: TokenKind::Mul },
                        [VOID, val.id, value],
                        self.tys,
                    ))
                } else {
                    self.report(pos, fa!("cant negate '{}'", self.ty_display(val.ty)));
                    Value::NEVER
                }
            }
            Expr::UnOp { pos, op: op @ TokenKind::Not, val } => {
                let val =
                    self.expr_ctx(val, Ctx::default().with_ty(ctx.ty.unwrap_or(ty::Id::INT)))?;
                if val.ty == ty::Id::BOOL {
                    Some(self.ci.nodes.new_node_lit(
                        val.ty,
                        Kind::UnOp { op },
                        [VOID, val.id],
                        self.tys,
                    ))
                } else {
                    self.report(pos, fa!("cant logically negate '{}'", self.ty_display(val.ty)));
                    Value::NEVER
                }
            }
            Expr::BinOp { left, op: TokenKind::Decl, right, pos } => {
                let mut right = self.expr(right)?;

                if right.ty.loc(self.tys) == Loc::Stack {
                    let stck = self.new_stack(pos, right.ty);
                    self.store_mem(stck, right.ty, right.id);
                    right.id = stck;
                    right.ptr = true;
                }
                self.assign_pattern(left, right);
                Some(Value::VOID)
            }
            Expr::BinOp { left: Expr::Wildcard { .. }, op: TokenKind::Assign, right, .. } => {
                self.expr(right)?;
                Some(Value::VOID)
            }
            Expr::BinOp { left, pos, op: TokenKind::Assign, right } => {
                let dest = self.raw_expr(left)?;
                let mut value = self.expr_ctx(right, Ctx::default().with_ty(dest.ty))?;

                self.assert_ty(pos, &mut value, dest.ty, "assignment source");

                if dest.var {
                    let var = &mut self.ci.scope.vars[(u16::MAX - dest.id) as usize];

                    if var.ptr {
                        let val = var.value();
                        let ty = var.ty;
                        self.store_mem(val, ty, value.id);
                    } else {
                        var.set_value_remove(value.id, &mut self.ci.nodes);
                    }
                } else if dest.ptr {
                    self.store_mem(dest.id, dest.ty, value.id);
                } else {
                    self.report(pos, "cannot assign to this expression");
                }

                Some(Value::VOID)
            }
            Expr::BinOp { left: &Expr::Null { pos }, .. } => {
                self.report(pos, "'null' must always be no the right side of an expression");
                Value::NEVER
            }
            Expr::BinOp {
                left,
                op: op @ (TokenKind::Eq | TokenKind::Ne),
                right: Expr::Null { .. },
                ..
            } => {
                let mut cmped = self.raw_expr(left)?;
                self.strip_var(&mut cmped);

                let Some(ty) = self.tys.inner_of(cmped.ty) else {
                    self.report(
                        left.pos(),
                        fa!("'{}' is never null, remove this check", self.ty_display(cmped.ty)),
                    );
                    return Value::NEVER;
                };

                Some(Value::new(self.gen_null_check(cmped, ty, op)).ty(ty::Id::BOOL))
            }
            Expr::BinOp { left, pos, op, right }
                if !matches!(op, TokenKind::Assign | TokenKind::Decl) =>
            {
                let mut lhs = self.raw_expr_ctx(left, ctx)?;
                self.strip_var(&mut lhs);
                self.implicit_unwrap(left.pos(), &mut lhs);

                match lhs.ty.expand() {
                    _ if lhs.ty.is_pointer()
                        || lhs.ty.is_integer()
                        || lhs.ty == ty::Id::BOOL
                        || (lhs.ty.is_float() && op.is_supported_float_op()) =>
                    {
                        self.strip_ptr(&mut lhs);
                        self.ci.nodes.lock(lhs.id);
                        let rhs = self.expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                        self.ci.nodes.unlock(lhs.id);
                        let mut rhs = rhs?;
                        self.strip_var(&mut rhs);
                        self.implicit_unwrap(right.pos(), &mut rhs);
                        let (ty, aclass) = self.binop_ty(pos, &mut lhs, &mut rhs, op);
                        if matches!(
                            op,
                            TokenKind::Lt
                                | TokenKind::Gt
                                | TokenKind::Ge
                                | TokenKind::Le
                                | TokenKind::Ne
                                | TokenKind::Eq
                        ) {
                            if lhs.ty.is_float() {
                            } else {
                                self.ci.nodes.lock(rhs.id);
                                let lty = lhs.ty.extend();
                                if lty != lhs.ty {
                                    self.extend(&mut lhs, lty);
                                }
                                self.ci.nodes.unlock(rhs.id);
                                let rty = rhs.ty.extend();
                                if rty != rhs.ty {
                                    self.extend(&mut rhs, rty);
                                }
                            }
                        }
                        let bop = self.ci.nodes.new_node_lit(
                            ty.bin_ret(op),
                            Kind::BinOp { op },
                            [VOID, lhs.id, rhs.id],
                            self.tys,
                        );
                        self.ci.nodes.pass_aclass(aclass, bop.id);
                        Some(bop)
                    }
                    ty::Kind::Struct(s) if op.is_homogenous() => {
                        self.ci.nodes.lock(lhs.id);
                        let rhs = self.raw_expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                        self.ci.nodes.unlock(lhs.id);
                        let mut rhs = rhs?;
                        self.strip_var(&mut rhs);
                        self.assert_ty(pos, &mut rhs, lhs.ty, "struct operand");
                        let dst = self.new_stack(pos, lhs.ty);
                        self.struct_op(left.pos(), op, s, dst, lhs.id, rhs.id);
                        Some(Value::ptr(dst).ty(lhs.ty))
                    }
                    _ => {
                        self.report(
                            pos,
                            fa!("'{} {op} _' is not supported", self.ty_display(lhs.ty)),
                        );
                        Value::NEVER
                    }
                }
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

                let elem = self.tys.ins.slices[s].elem;
                let mut idx = self.expr_ctx(index, Ctx::default().with_ty(ty::Id::DINT))?;
                self.assert_ty(index.pos(), &mut idx, ty::Id::DINT, "subscript");
                let size = self.ci.nodes.new_const(ty::Id::INT, self.tys.size_of(elem));
                let inps = [VOID, idx.id, size];
                let offset = self.ci.nodes.new_node(
                    ty::Id::INT,
                    Kind::BinOp { op: TokenKind::Mul },
                    inps,
                    self.tys,
                );
                let aclass = self.ci.nodes.aclass_index(bs.id).1;
                let inps = [VOID, bs.id, offset];
                let ptr = self.ci.nodes.new_node(
                    ty::Id::INT,
                    Kind::BinOp { op: TokenKind::Add },
                    inps,
                    self.tys,
                );
                self.ci.nodes.pass_aclass(aclass, ptr);

                Some(Value::ptr(ptr).ty(elem))
            }
            Expr::Embed { id, .. } => {
                let glob = &self.tys.ins.globals[id];
                let g =
                    self.ci.nodes.new_node(glob.ty, Kind::Global { global: id }, [VOID], self.tys);
                Some(Value::ptr(g).ty(glob.ty))
            }
            Expr::Directive { name: "sizeof", args: [ty], .. } => {
                let ty = self.ty(ty);
                self.gen_inferred_const(ctx, ty::Id::DINT, self.tys.size_of(ty), ty::Id::is_integer)
            }
            Expr::Directive { name: "alignof", args: [ty], .. } => {
                let ty = self.ty(ty);
                let align = self.tys.align_of(ty);
                self.gen_inferred_const(ctx, ty::Id::DINT, align, ty::Id::is_integer)
            }
            Expr::Directive { name: "bitcast", args: [val], pos } => {
                let mut val = self.raw_expr(val)?;
                self.strip_var(&mut val);

                inference!(ty, ctx, self, pos, "type", "@as(<ty>, @bitcast(<expr>))");

                let (got, expected) = (self.tys.size_of(val.ty), self.tys.size_of(ty));
                if got != expected {
                    self.report(
                        pos,
                        fa!(
                            "cast from '{}' to '{}' is not supported, \
                            sizes dont match ({got} != {expected})",
                            self.ty_display(val.ty),
                            self.ty_display(ty)
                        ),
                    );
                }

                match ty.loc(self.tys) {
                    Loc::Reg if mem::take(&mut val.ptr) => val.id = self.load_mem(val.id, ty),
                    Loc::Stack if !val.ptr => {
                        let stack = self.new_stack(pos, ty);
                        self.store_mem(stack, val.ty, val.id);
                        val.id = stack;
                        val.ptr = true;
                    }
                    _ => {}
                }

                val.ty = ty;
                Some(val)
            }
            Expr::Directive { name: "unwrap", args: [expr], .. } => {
                let mut val = self.raw_expr(expr)?;
                self.strip_var(&mut val);

                if !val.ty.is_optional() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "only optional types can be unwrapped ('{}' is not optional)",
                            self.ty_display(val.ty)
                        ),
                    );
                    return Value::NEVER;
                };

                self.explicit_unwrap(expr.pos(), &mut val);
                Some(val)
            }
            Expr::Directive { name: "intcast", args: [expr], pos } => {
                let mut val = self.expr(expr)?;

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

                inference!(ty, ctx, self, pos, "integer", "@as(<ty>, @intcast(<expr>))");

                if !ty.is_integer() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "intcast is inferred to output '{}', which is not an integer",
                            self.ty_display(ty)
                        ),
                    );
                }

                if self.tys.size_of(val.ty) < self.tys.size_of(ty) {
                    self.extend(&mut val, ty);
                    Some(val)
                } else {
                    Some(val.ty(ty))
                }
            }
            Expr::Directive { pos, name: "floatcast", args: [expr] } => {
                let val = self.expr(expr)?;

                if !val.ty.is_float() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "only floats can be truncated ('{}' is not a float)",
                            self.ty_display(val.ty)
                        ),
                    );
                    return Value::NEVER;
                }

                inference!(ty, ctx, self, pos, "float", "@as(<floaty>, @floatcast(<expr>))");

                if !ty.is_float() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "floatcast is inferred to output '{}', which is not a float",
                            self.ty_display(ty)
                        ),
                    );
                }

                if self.tys.size_of(val.ty) != self.tys.size_of(ty) {
                    Some(self.ci.nodes.new_node_lit(
                        ty,
                        Kind::UnOp { op: TokenKind::Float },
                        [VOID, val.id],
                        self.tys,
                    ))
                } else {
                    Some(val.ty(ty))
                }
            }
            Expr::Directive { name: "fti", args: [expr], .. } => {
                let val = self.expr(expr)?;

                let ret_ty = match val.ty {
                    ty::Id::F32 | ty::Id::F64 => ty::Id::INT,
                    _ => {
                        self.report(
                            expr.pos(),
                            fa!("expected float ('{}' is not a float)", self.ty_display(val.ty)),
                        );
                        return Value::NEVER;
                    }
                };

                Some(self.ci.nodes.new_node_lit(
                    ret_ty,
                    Kind::UnOp { op: TokenKind::Number },
                    [VOID, val.id],
                    self.tys,
                ))
            }
            Expr::Directive { name: "itf", args: [expr], .. } => {
                let mut val = self.expr_ctx(expr, Ctx::default().with_ty(ty::Id::INT))?;

                let (ret_ty, expected) = match val.ty.simple_size().unwrap() {
                    8 => (ty::Id::F64, ty::Id::INT),
                    _ => (ty::Id::F32, ty::Id::INT),
                };

                self.assert_ty(expr.pos(), &mut val, expected, "converted integer");

                Some(self.ci.nodes.new_node_lit(
                    ret_ty,
                    Kind::UnOp { op: TokenKind::Float },
                    [VOID, val.id],
                    self.tys,
                ))
            }
            Expr::Directive { name: "as", args: [ty, expr], pos } => {
                let ty = self.ty(ty);
                let mut val = self.raw_expr_ctx(expr, Ctx::default().with_ty(ty))?;

                if let Some(ity) = ctx.ty
                    && ity.try_upcast(ty) == Some(ty)
                    && val.ty == ity
                {
                    self.report(pos, "the type is known at this point, remove the hint");
                }
                self.strip_var(&mut val);
                self.assert_ty(expr.pos(), &mut val, ty, "hinted expr");
                Some(val)
            }
            Expr::Directive { pos, name: "eca", args } => {
                inference!(ty, ctx, self, pos, "return type", "@as(<return_ty>, @eca(<expr>...))");

                let mut inps = Vc::from([NEVER]);
                let arg_base = self.tys.tmp.args.len();
                let mut clobbered_aliases = BitSet::default();
                for arg in args {
                    let value = self.expr(arg)?;
                    self.add_clobbers(value, &mut clobbered_aliases);
                    self.tys.tmp.args.push(value.ty);
                    debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                    self.ci.nodes.lock(value.id);
                    inps.push(value.id);
                }

                let args = self.tys.pack_args(arg_base).expect("TODO");

                for &n in inps.iter().skip(1) {
                    self.ci.nodes.unlock(n);
                }

                self.append_clobbers(&mut inps, &mut clobbered_aliases);

                let alt_value = match ty.loc(self.tys) {
                    Loc::Reg => None,
                    Loc::Stack => {
                        let stck = self.new_stack(pos, ty);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(ty))
                    }
                };

                inps[0] = self.ci.ctrl.get();
                self.ci.ctrl.set(
                    self.ci.nodes.new_node_nop(ty, Kind::Call { func: ty::Func::ECA, args }, inps),
                    &mut self.ci.nodes,
                );

                self.add_clobber_stores(clobbered_aliases);

                alt_value.or(Some(Value::new(self.ci.ctrl.get()).ty(ty)))
            }
            Expr::Call { func, args, .. } => self.gen_call(func, args, false),
            Expr::Directive { name: "inline", args: [func, args @ ..], .. } => {
                self.gen_call(func, args, true)
            }
            Expr::Tupl { pos, ty, fields, .. } => {
                ctx.ty = ty
                    .map(|ty| self.ty(ty))
                    .or(ctx.ty.map(|ty| self.tys.inner_of(ty).unwrap_or(ty)));
                inference!(sty, ctx, self, pos, "struct or slice", "<struct_ty>.(...)");

                match sty.expand() {
                    ty::Kind::Struct(s) => {
                        let mem = self.new_stack(pos, sty);
                        let mut offs = OffsetIter::new(s, self.tys);
                        for field in fields {
                            let Some((ty, offset)) = offs.next_ty(self.tys) else {
                                self.report(
                                    field.pos(),
                                    "this init argumen overflows the field count",
                                );
                                break;
                            };

                            let mut value = self.expr_ctx(field, Ctx::default().with_ty(ty))?;
                            _ = self.assert_ty(field.pos(), &mut value, ty, "tuple field");
                            let mem = self.offset(mem, offset);

                            self.store_mem(mem, ty, value.id);
                        }

                        let field_list = offs
                            .into_iter(self.tys)
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
                        let slice = &self.tys.ins.slices[s];
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

                        let mem = self.new_stack(pos, aty);

                        for (field, offset) in
                            fields.iter().zip((0u32..).step_by(elem_size as usize))
                        {
                            let mut value = self.expr_ctx(field, Ctx::default().with_ty(elem))?;
                            _ = self.assert_ty(field.pos(), &mut value, elem, "array value");
                            let mem = self.offset(mem, offset);
                            self.store_mem(mem, elem, value.id);
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
            Expr::Struct { .. } => {
                let value = self.ty(expr).repr();
                Some(self.ci.nodes.new_const_lit(ty::Id::TYPE, value))
            }
            Expr::Ctor { pos, ty, fields, .. } => {
                ctx.ty = ty
                    .map(|ty| self.ty(ty))
                    .or(ctx.ty.map(|ty| self.tys.inner_of(ty).unwrap_or(ty)));
                inference!(sty, ctx, self, pos, "struct", "<struct_ty>.{...}");

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
                let mut offs = OffsetIter::new(s, self.tys)
                    .into_iter(self.tys)
                    .map(|(f, o)| (f.ty, o))
                    .collect::<Vec<_>>();
                let mem = self.new_stack(pos, sty);
                for field in fields {
                    let Some(index) = self.tys.find_struct_field(s, field.name) else {
                        self.report(
                            field.pos,
                            fa!("struct '{}' does not have this field", self.ty_display(sty)),
                        );
                        continue;
                    };

                    let (ty, offset) =
                        mem::replace(&mut offs[index], (ty::Id::UNDECLARED, field.pos));

                    if ty == ty::Id::UNDECLARED {
                        self.report(field.pos, "the struct field is already initialized");
                        self.report(offset, "previous initialization is here");
                        continue;
                    }

                    let mut value = self.expr_ctx(&field.value, Ctx::default().with_ty(ty))?;
                    self.assert_ty(field.pos, &mut value, ty, fa!("field {}", field.name));
                    let mem = self.offset(mem, offset);
                    self.store_mem(mem, ty, value.id);
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
                let aclass_base = self.ci.scope.aclasses.len();

                let mut ret = Some(Value::VOID);
                for stmt in stmts {
                    ret = ret.and(self.expr(stmt));
                    if let Some(mut id) = ret {
                        self.assert_ty(stmt.pos(), &mut id, ty::Id::VOID, "statement");
                    } else {
                        break;
                    }
                }

                for var in self.ci.scope.vars.drain(base..) {
                    var.remove(&mut self.ci.nodes);
                }

                for aclass in self.ci.scope.aclasses.drain(aclass_base..) {
                    aclass.remove(&mut self.ci.nodes);
                }

                ret
            }
            Expr::Loop { body, .. } => {
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(
                        ty::Id::VOID,
                        Kind::Loop,
                        [self.ci.ctrl.get(), self.ci.ctrl.get(), LOOPS],
                        self.tys,
                    ),
                    &mut self.ci.nodes,
                );

                self.ci.loops.push(Loop {
                    node: self.ci.ctrl.get(),
                    ctrl: [StrongRef::DEFAULT; 2],
                    ctrl_scope: core::array::from_fn(|_| Default::default()),
                    scope: self.ci.scope.dup(&mut self.ci.nodes),
                });

                for var in self.ci.scope.vars.iter_mut().skip(self.ci.inline_var_base) {
                    if !var.ptr && var.value() != NEVER {
                        var.set_value(VOID, &mut self.ci.nodes);
                    }
                }

                for aclass in self.ci.scope.aclasses[..2].iter_mut() {
                    aclass.last_store.set(VOID, &mut self.ci.nodes);
                }
                for aclass in self.ci.scope.aclasses.iter_mut().skip(self.ci.inline_aclass_base) {
                    aclass.last_store.set(VOID, &mut self.ci.nodes);
                }

                self.expr(body);

                let Loop { ctrl: [con, ..], ctrl_scope: [cons, ..], .. } =
                    self.ci.loops.last_mut().unwrap();
                let mut cons = mem::take(cons);

                if let Some(con) = mem::take(con).unwrap(&mut self.ci.nodes) {
                    self.ci.ctrl.set(
                        self.ci.nodes.new_node(
                            ty::Id::VOID,
                            Kind::Region,
                            [con, self.ci.ctrl.get()],
                            self.tys,
                        ),
                        &mut self.ci.nodes,
                    );
                    self.ci.nodes.merge_scopes(
                        &mut self.ci.loops,
                        &self.ci.ctrl,
                        &mut self.ci.scope,
                        &mut cons,
                        self.tys,
                    );
                    cons.clear(&mut self.ci.nodes);
                }

                let Loop { node, ctrl: [.., bre], ctrl_scope: [.., mut bres], mut scope } =
                    self.ci.loops.pop().unwrap();

                self.ci.nodes.modify_input(node, 1, self.ci.ctrl.get());

                if let Some(idx) =
                    self.ci.nodes[node].outputs.iter().position(|&n| self.ci.nodes.is_cfg(n))
                {
                    self.ci.nodes[node].outputs.swap(idx, 0);
                }

                let Some(bre) = bre.unwrap(&mut self.ci.nodes) else {
                    for (loop_var, scope_var) in
                        self.ci.scope.vars.iter_mut().zip(scope.vars.iter_mut())
                    {
                        if self.ci.nodes[scope_var.value()].is_lazy_phi(node) {
                            if loop_var.value() != scope_var.value() {
                                scope_var.set_value(
                                    self.ci.nodes.modify_input(
                                        scope_var.value(),
                                        2,
                                        loop_var.value(),
                                    ),
                                    &mut self.ci.nodes,
                                );
                            } else {
                                let phi = &self.ci.nodes[scope_var.value()];
                                let prev = phi.inputs[1];
                                self.ci.nodes.replace(scope_var.value(), prev);
                                scope_var.set_value(prev, &mut self.ci.nodes);
                            }
                        }
                    }

                    for (loop_class, scope_class) in
                        self.ci.scope.aclasses.iter_mut().zip(scope.aclasses.iter_mut())
                    {
                        if self.ci.nodes[scope_class.last_store.get()].is_lazy_phi(node) {
                            if loop_class.last_store.get() != scope_class.last_store.get()
                                && loop_class.last_store.get() != 0
                            {
                                scope_class.last_store.set(
                                    self.ci.nodes.modify_input(
                                        scope_class.last_store.get(),
                                        2,
                                        loop_class.last_store.get(),
                                    ),
                                    &mut self.ci.nodes,
                                );
                            } else {
                                let phi = &self.ci.nodes[scope_class.last_store.get()];
                                let prev = phi.inputs[1];
                                self.ci.nodes.replace(scope_class.last_store.get(), prev);
                                scope_class.last_store.set(prev, &mut self.ci.nodes);
                            }
                        }

                        if loop_class.last_store.get() == 0 {
                            loop_class
                                .last_store
                                .set(scope_class.last_store.get(), &mut self.ci.nodes);
                        }
                    }

                    debug_assert!(self.ci.scope.aclasses.iter().all(|a| a.last_store.get() != 0));

                    scope.clear(&mut self.ci.nodes);
                    self.ci.ctrl.set(NEVER, &mut self.ci.nodes);

                    return None;
                };

                self.ci.ctrl.set(bre, &mut self.ci.nodes);

                mem::swap(&mut self.ci.scope, &mut bres);

                debug_assert_eq!(self.ci.scope.vars.len(), scope.vars.len());
                debug_assert_eq!(self.ci.scope.vars.len(), bres.vars.len());

                self.ci.nodes.lock(node);

                for ((dest_var, scope_var), loop_var) in self
                    .ci
                    .scope
                    .vars
                    .iter_mut()
                    .zip(scope.vars.iter_mut())
                    .zip(bres.vars.iter_mut())
                {
                    if self.ci.nodes[scope_var.value()].is_lazy_phi(node) {
                        if loop_var.value() != scope_var.value() && loop_var.value() != 0 {
                            scope_var.set_value(
                                self.ci.nodes.modify_input(scope_var.value(), 2, loop_var.value()),
                                &mut self.ci.nodes,
                            );
                        } else {
                            if dest_var.value() == scope_var.value() {
                                dest_var.set_value(VOID, &mut self.ci.nodes);
                            }
                            let phi = &self.ci.nodes[scope_var.value()];
                            let prev = phi.inputs[1];
                            self.ci.nodes.replace(scope_var.value(), prev);
                            scope_var.set_value(prev, &mut self.ci.nodes);
                        }
                    }

                    if dest_var.value() == VOID {
                        dest_var.set_value(scope_var.value(), &mut self.ci.nodes);
                    }

                    debug_assert!(!self.ci.nodes[dest_var.value()].is_lazy_phi(node));
                }

                for ((dest_class, scope_class), loop_class) in self
                    .ci
                    .scope
                    .aclasses
                    .iter_mut()
                    .zip(scope.aclasses.iter_mut())
                    .zip(bres.aclasses.iter_mut())
                {
                    if self.ci.nodes[scope_class.last_store.get()].is_lazy_phi(node) {
                        if loop_class.last_store.get() != scope_class.last_store.get()
                            && loop_class.last_store.get() != 0
                        {
                            scope_class.last_store.set(
                                self.ci.nodes.modify_input(
                                    scope_class.last_store.get(),
                                    2,
                                    loop_class.last_store.get(),
                                ),
                                &mut self.ci.nodes,
                            );
                        } else {
                            if dest_class.last_store.get() == scope_class.last_store.get() {
                                dest_class.last_store.set(VOID, &mut self.ci.nodes);
                            }
                            let phi = &self.ci.nodes[scope_class.last_store.get()];
                            let prev = phi.inputs[1];
                            self.ci.nodes.replace(scope_class.last_store.get(), prev);
                            scope_class.last_store.set(prev, &mut self.ci.nodes);
                        }
                    }

                    if dest_class.last_store.get() == VOID {
                        dest_class.last_store.set(scope_class.last_store.get(), &mut self.ci.nodes);
                    }

                    debug_assert!(
                        !self.ci.nodes[dest_class.last_store.get()].is_lazy_phi(node),
                        "{:?}",
                        self.ci.nodes[dest_class.last_store.get()]
                    );
                }

                scope.clear(&mut self.ci.nodes);
                bres.clear(&mut self.ci.nodes);

                self.ci.nodes.unlock(node);
                let rpl = self.ci.nodes.late_peephole(node, self.tys).unwrap_or(node);
                if self.ci.ctrl.get() == node {
                    self.ci.ctrl.set_remove(rpl, &mut self.ci.nodes);
                }

                Some(Value::VOID)
            }
            Expr::Break { pos } => self.jump_to(pos, 1),
            Expr::Continue { pos } => self.jump_to(pos, 0),
            Expr::If { cond, then, else_, .. } => {
                let mut cnd = self.expr_ctx(cond, Ctx::default().with_ty(ty::Id::BOOL))?;
                self.assert_ty(cond.pos(), &mut cnd, ty::Id::BOOL, "condition");

                let if_node = self.ci.nodes.new_node(
                    ty::Id::VOID,
                    Kind::If,
                    [self.ci.ctrl.get(), cnd.id],
                    self.tys,
                );

                'b: {
                    let branch = match self.ci.nodes[if_node].ty {
                        ty::Id::LEFT_UNREACHABLE => else_,
                        ty::Id::RIGHT_UNREACHABLE => Some(then),
                        _ => break 'b,
                    };

                    self.ci.nodes.remove(if_node);

                    if let Some(branch) = branch {
                        return self.expr(branch);
                    } else {
                        return Some(Value::VOID);
                    }
                }

                let else_scope = self.ci.scope.dup(&mut self.ci.nodes);

                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Then, [if_node], self.tys),
                    &mut self.ci.nodes,
                );
                let lcntrl = self.expr(then).map_or(Nid::MAX, |_| self.ci.ctrl.get());

                let mut then_scope = mem::replace(&mut self.ci.scope, else_scope);
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Else, [if_node], self.tys),
                    &mut self.ci.nodes,
                );
                let rcntrl = if let Some(else_) = else_ {
                    self.expr(else_).map_or(Nid::MAX, |_| self.ci.ctrl.get())
                } else {
                    self.ci.ctrl.get()
                };

                if lcntrl == Nid::MAX && rcntrl == Nid::MAX {
                    then_scope.clear(&mut self.ci.nodes);
                    return None;
                } else if lcntrl == Nid::MAX {
                    then_scope.clear(&mut self.ci.nodes);
                    return Some(Value::VOID);
                } else if rcntrl == Nid::MAX {
                    self.ci.scope.clear(&mut self.ci.nodes);
                    self.ci.scope = then_scope;
                    self.ci.ctrl.set(lcntrl, &mut self.ci.nodes);
                    return Some(Value::VOID);
                }

                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [lcntrl, rcntrl], self.tys),
                    &mut self.ci.nodes,
                );

                self.ci.nodes.merge_scopes(
                    &mut self.ci.loops,
                    &self.ci.ctrl,
                    &mut self.ci.scope,
                    &mut then_scope,
                    self.tys,
                );
                then_scope.clear(&mut self.ci.nodes);

                Some(Value::VOID)
            }
            ref e => {
                self.report_unhandled_ast(e, "bruh");
                Value::NEVER
            }
        }
    }

    fn gen_inferred_const(
        &mut self,
        ctx: Ctx,
        fallback: ty::Id,
        value: impl Into<i64>,
        filter: impl Fn(ty::Id) -> bool,
    ) -> Option<Value> {
        Some(
            self.ci.nodes.new_const_lit(
                ctx.ty
                    .map(|ty| self.tys.inner_of(ty).unwrap_or(ty))
                    .filter(|&ty| filter(ty))
                    .unwrap_or(fallback),
                value,
            ),
        )
    }

    fn gen_call(&mut self, func: &Expr, args: &[Expr], inline: bool) -> Option<Value> {
        let ty = self.ty(func);
        let ty::Kind::Func(mut fu) = ty.expand() else {
            self.report(func.pos(), fa!("compiler cant (yet) call '{}'", self.ty_display(ty)));
            return Value::NEVER;
        };

        let Some(sig) = self.compute_signature(&mut fu, func.pos(), args) else {
            return Value::NEVER;
        };

        let Func { expr, file, is_inline, .. } = self.tys.ins.funcs[fu];
        let ast = &self.files[file.index()];
        let &Expr::Closure { args: cargs, body, .. } = expr.get(ast) else { unreachable!() };

        if args.len() != cargs.len() {
            self.report(
                func.pos(),
                fa!(
                    "expected {} function argumenr{}, got {}",
                    cargs.len(),
                    if cargs.len() == 1 { "" } else { "s" },
                    args.len()
                ),
            );
        }

        if inline && is_inline {
            self.report(
                func.pos(),
                "function is declared as inline so this @inline directive only reduces readability",
            );
        }

        let (mut tys, mut args, mut cargs) = (sig.args.args(), args.iter(), cargs.iter());
        if is_inline || inline {
            let var_base = self.ci.scope.vars.len();
            let aclass_base = self.ci.scope.aclasses.len();
            while let (Some(aty), Some(arg)) = (tys.next(self.tys), args.next()) {
                let carg = cargs.next().unwrap();
                let var = match aty {
                    Arg::Type(id) => Variable::new(carg.id, id, false, NEVER, &mut self.ci.nodes),
                    Arg::Value(ty) => {
                        let mut value = self.raw_expr_ctx(arg, Ctx::default().with_ty(ty))?;
                        self.strip_var(&mut value);
                        debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                        debug_assert_ne!(value.id, 0);
                        self.assert_ty(arg.pos(), &mut value, ty, fa!("argument {}", carg.name));
                        Variable::new(carg.id, ty, value.ptr, value.id, &mut self.ci.nodes)
                    }
                };
                self.ci.scope.vars.push(var);
            }

            let prev_var_base = mem::replace(&mut self.ci.inline_var_base, var_base);
            let prev_aclass_base = mem::replace(&mut self.ci.inline_aclass_base, aclass_base);
            let prev_inline_ret = self.ci.inline_ret.take();
            self.ci.inline_depth += 1;
            let prev_ret = self.ci.ret.replace(sig.ret);
            let prev_file = mem::replace(&mut self.ci.file, file);
            let prev_ctrl = self.ci.ctrl.get();

            if self.expr(body).is_some() {
                if sig.ret == ty::Id::VOID {
                    self.expr(&Expr::Return { pos: body.pos(), val: None });
                } else {
                    self.report(
                        body.pos(),
                        "expected all paths in the fucntion to return \
                        or the return type to be 'void'",
                    );
                }
            }

            self.ci.ret = prev_ret;
            self.ci.file = prev_file;
            self.ci.inline_depth -= 1;
            self.ci.inline_var_base = prev_var_base;
            self.ci.inline_aclass_base = prev_aclass_base;
            for var in self.ci.scope.vars.drain(var_base..) {
                var.remove(&mut self.ci.nodes);
            }
            for var in self.ci.scope.aclasses.drain(aclass_base..) {
                var.remove(&mut self.ci.nodes);
            }

            let (v, ctrl, mut scope, aclass) =
                mem::replace(&mut self.ci.inline_ret, prev_inline_ret)?;
            if is_inline
                && ctrl.get() != prev_ctrl
                && (!self.ci.nodes[ctrl.get()].kind.is_eca()
                    || self.ci.nodes[ctrl.get()].inputs[0] != prev_ctrl)
            {
                self.report(body.pos(), "function is makred inline but it contains controlflow");
            }

            scope.vars.drain(var_base..).for_each(|v| v.remove(&mut self.ci.nodes));
            scope.aclasses.drain(aclass_base..).for_each(|v| v.remove(&mut self.ci.nodes));
            self.ci.nodes.unlock(v.id);
            self.ci.scope.clear(&mut self.ci.nodes);
            self.ci.scope = scope;

            if let Some(aclass) = aclass {
                let (_, reg) = self.ci.nodes.aclass_index(v.id);
                self.ci.nodes[reg].aclass = self.ci.scope.aclasses.len() as _;
                self.ci.scope.aclasses.push(aclass);
            }

            mem::replace(&mut self.ci.ctrl, ctrl).remove(&mut self.ci.nodes);

            Some(v)
        } else {
            self.make_func_reachable(fu);
            let mut inps = Vc::from([NEVER]);
            let mut clobbered_aliases = BitSet::default();
            while let (Some(ty), Some(arg)) = (tys.next(self.tys), args.next()) {
                let carg = cargs.next().unwrap();
                let Arg::Value(ty) = ty else { continue };

                let mut value = self.raw_expr_ctx(arg, Ctx::default().with_ty(ty))?;
                self.strip_var(&mut value);
                debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                self.assert_ty(arg.pos(), &mut value, ty, fa!("argument {}", carg.name));
                self.strip_ptr(&mut value);
                self.add_clobbers(value, &mut clobbered_aliases);

                self.ci.nodes.lock(value.id);
                inps.push(value.id);
            }

            for &n in inps.iter().skip(1) {
                self.ci.nodes.unlock(n);
            }

            self.append_clobbers(&mut inps, &mut clobbered_aliases);

            let alt_value = match sig.ret.loc(self.tys) {
                Loc::Reg => None,
                Loc::Stack => {
                    let stck = self.new_stack(func.pos(), sig.ret);
                    clobbered_aliases.set(self.ci.nodes.aclass_index(stck).0 as _);
                    inps.push(stck);
                    Some(Value::ptr(stck).ty(sig.ret))
                }
            };

            inps[0] = self.ci.ctrl.get();
            self.ci.ctrl.set(
                self.ci.nodes.new_node_nop(sig.ret, Kind::Call { func: fu, args: sig.args }, inps),
                &mut self.ci.nodes,
            );

            self.add_clobber_stores(clobbered_aliases);

            alt_value.or(Some(Value::new(self.ci.ctrl.get()).ty(sig.ret)))
        }
    }

    fn gen_global(&mut self, global: ty::Global) -> Option<Value> {
        let gl = &self.tys.ins.globals[global];
        let value = self.ci.nodes.new_node_nop(gl.ty, Kind::Global { global }, [VOID]);
        self.ci.nodes[value].aclass = GLOBAL_ACLASS as _;
        Some(Value::ptr(value).ty(gl.ty))
    }

    fn gen_const(&mut self, cnst: ty::Const, ctx: Ctx) -> Option<Value> {
        let c = &self.tys.ins.consts[cnst];
        let prev = mem::replace(&mut self.ci.file, c.file);
        let f = &self.files[c.file.index()];
        let Expr::BinOp { left, right, .. } = c.ast.get(f) else { unreachable!() };

        let mut value = left
            .find_pattern_path(c.name, right, |expr, is_ct| {
                debug_assert!(is_ct);
                self.raw_expr_ctx(expr, ctx)
            })
            .unwrap_or_else(|_| unreachable!())?;
        self.strip_var(&mut value);
        self.ci.file = prev;
        Some(value)
    }

    fn add_clobbers(&mut self, value: Value, clobbered_aliases: &mut BitSet) {
        if let Some(base) = self.tys.base_of(value.ty) {
            clobbered_aliases.set(self.ci.nodes.aclass_index(value.id).0 as _);
            if base.has_pointers(self.tys) {
                clobbered_aliases.set(DEFAULT_ACLASS as _);
            }
        } else if value.ty.has_pointers(self.tys) {
            clobbered_aliases.set(DEFAULT_ACLASS as _);
        }
    }

    fn append_clobbers(&mut self, inps: &mut Vc, clobbered_aliases: &mut BitSet) {
        clobbered_aliases.set(GLOBAL_ACLASS as _);
        for clobbered in clobbered_aliases.iter() {
            let aclass = &mut self.ci.scope.aclasses[clobbered];
            self.ci.nodes.load_loop_aclass(clobbered, aclass, &mut self.ci.loops);
            inps.push(aclass.last_store.get());
        }
    }

    fn add_clobber_stores(&mut self, clobbered_aliases: BitSet) {
        for clobbered in clobbered_aliases.iter() {
            debug_assert_matches!(self.ci.nodes[self.ci.ctrl.get()].kind, Kind::Call { .. });
            self.ci.scope.aclasses[clobbered].clobber.set(self.ci.ctrl.get(), &mut self.ci.nodes);
        }
        self.ci.nodes[self.ci.ctrl.get()].clobbers = clobbered_aliases;
    }

    fn struct_op(
        &mut self,
        pos: Pos,
        op: TokenKind,
        s: ty::Struct,
        dst: Nid,
        lhs: Nid,
        rhs: Nid,
    ) -> bool {
        let mut offs = OffsetIter::new(s, self.tys);
        while let Some((ty, off)) = offs.next_ty(self.tys) {
            let lhs = self.offset(lhs, off);
            let rhs = self.offset(rhs, off);
            let dst = self.offset(dst, off);
            match ty.expand() {
                _ if ty.is_pointer() || ty.is_integer() || ty == ty::Id::BOOL => {
                    let lhs = self.load_mem(lhs, ty);
                    let rhs = self.load_mem(rhs, ty);
                    let res =
                        self.ci.nodes.new_node(ty, Kind::BinOp { op }, [VOID, lhs, rhs], self.tys);
                    self.store_mem(dst, ty, res);
                }
                ty::Kind::Struct(is) => {
                    if !self.struct_op(pos, op, is, dst, lhs, rhs) {
                        self.report(
                            pos,
                            fa!("... when appliing '{0} {op} {0}'", self.ty_display(s.into())),
                        );
                    }
                }
                _ => self.report(pos, fa!("'{0} {op} {0}' is not supported", self.ty_display(ty))),
            }
        }

        true
    }

    fn compute_signature(&mut self, func: &mut ty::Func, pos: Pos, args: &[Expr]) -> Option<Sig> {
        let Func { file, expr, sig, .. } = self.tys.ins.funcs[*func];
        let fast = self.files[file.index()].clone();
        let &Expr::Closure { args: cargs, ret, .. } = expr.get(&fast) else {
            unreachable!();
        };

        Some(if let Some(sig) = sig {
            sig
        } else {
            let arg_base = self.tys.tmp.args.len();

            let base = self.ci.scope.vars.len();
            for (arg, carg) in args.iter().zip(cargs) {
                let ty = self.ty_in(file, &carg.ty);

                self.tys.tmp.args.push(ty);
                let sym = parser::find_symbol(&fast.symbols, carg.id);
                let ty = if sym.flags & idfl::COMPTIME == 0 {
                    // FIXME: could fuck us
                    ty::Id::UNDECLARED
                } else {
                    if ty != ty::Id::TYPE {
                        self.report(
                            arg.pos(),
                            fa!(
                                "arbitrary comptime types are not supported yet \
                                (expected '{}' got '{}')",
                                self.ty_display(ty::Id::TYPE),
                                self.ty_display(ty)
                            ),
                        );
                        return None;
                    }
                    let ty = self.ty(arg);
                    self.tys.tmp.args.push(ty);
                    ty
                };

                self.ci.scope.vars.push(Variable::new(
                    carg.id,
                    ty,
                    false,
                    NEVER,
                    &mut self.ci.nodes,
                ));
            }

            let Some(args) = self.tys.pack_args(arg_base) else {
                self.report(pos, "function instance has too many arguments");
                return None;
            };
            let ret = self.ty_in(file, ret);

            self.ci.scope.vars.drain(base..).for_each(|v| v.remove(&mut self.ci.nodes));

            let sym = SymKey::FuncInst(*func, args);
            let ct = |ins: &mut crate::TypeIns| {
                let fuc = ins.funcs[*func];
                debug_assert!(fuc.comp_state.iter().all(|&s| s == CompState::default()));
                ins.funcs
                    .push(Func { base: Some(*func), sig: Some(Sig { args, ret }), ..fuc })
                    .into()
            };
            let ty::Kind::Func(f) =
                self.tys.syms.get_or_insert(sym, &mut self.tys.ins, ct).expand()
            else {
                unreachable!()
            };
            *func = f;

            Sig { args, ret }
        })
    }

    fn assign_pattern(&mut self, pat: &Expr, right: Value) {
        match *pat {
            Expr::Ident { id, .. } => {
                self.ci.scope.vars.push(Variable::new(
                    id,
                    right.ty,
                    right.ptr,
                    right.id,
                    &mut self.ci.nodes,
                ));
            }
            Expr::Ctor { pos, fields, .. } => {
                let ty::Kind::Struct(idx) = right.ty.expand() else {
                    self.report(pos, "can't use struct destruct on non struct value (TODO: shold work with modules)");
                    return;
                };

                for &CtorField { pos, name, ref value } in fields {
                    let Some((offset, ty)) = OffsetIter::offset_of(self.tys, idx, name) else {
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
        self.strip_ptr(&mut n);
        Some(n)
    }

    fn expr(&mut self, expr: &Expr) -> Option<Value> {
        self.expr_ctx(expr, Default::default())
    }

    fn strip_ptr(&mut self, target: &mut Value) {
        if mem::take(&mut target.ptr) {
            target.id = self.load_mem(target.id, target.ty);
        }
    }

    fn offset(&mut self, val: Nid, off: Offset) -> Nid {
        if off == 0 {
            return val;
        }

        let off = self.ci.nodes.new_const(ty::Id::INT, off);
        let aclass = self.ci.nodes.aclass_index(val).1;
        let inps = [VOID, val, off];
        let seted =
            self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Add }, inps, self.tys);
        self.ci.nodes.pass_aclass(aclass, seted);
        seted
    }

    fn strip_var(&mut self, n: &mut Value) {
        if mem::take(&mut n.var) {
            let id = (u16::MAX - n.id) as usize;
            n.ptr = self.ci.scope.vars[id].ptr;
            n.id = self.ci.scope.vars[id].value();
        }
    }

    fn jump_to(&mut self, pos: Pos, id: usize) -> Option<Value> {
        let Some(mut loob) = self.ci.loops.last_mut() else {
            self.report(pos, "break outside a loop");
            return None;
        };

        if loob.ctrl[id].is_live() {
            loob.ctrl[id].set(
                self.ci.nodes.new_node(
                    ty::Id::VOID,
                    Kind::Region,
                    [self.ci.ctrl.get(), loob.ctrl[id].get()],
                    self.tys,
                ),
                &mut self.ci.nodes,
            );
            let mut scope = mem::take(&mut loob.ctrl_scope[id]);
            let ctrl = mem::take(&mut loob.ctrl[id]);

            self.ci.nodes.merge_scopes(
                &mut self.ci.loops,
                &ctrl,
                &mut scope,
                &mut self.ci.scope,
                self.tys,
            );

            loob = self.ci.loops.last_mut().unwrap();
            loob.ctrl_scope[id] = scope;
            loob.ctrl[id] = ctrl;
            self.ci.ctrl.set(NEVER, &mut self.ci.nodes);
        } else {
            let term = StrongRef::new(NEVER, &mut self.ci.nodes);
            loob.ctrl[id] = mem::replace(&mut self.ci.ctrl, term);
            loob.ctrl_scope[id] = self.ci.scope.dup(&mut self.ci.nodes);
            loob.ctrl_scope[id]
                .vars
                .drain(loob.scope.vars.len()..)
                .for_each(|v| v.remove(&mut self.ci.nodes));
            loob.ctrl_scope[id]
                .aclasses
                .drain(loob.scope.aclasses.len()..)
                .for_each(|v| v.remove(&mut self.ci.nodes));
        }

        None
    }

    fn complete_call_graph(&mut self) -> bool {
        let prev_err_len = self.errors.borrow().len();
        while self.ci.task_base < self.tys.tasks.len()
            && let Some(task_slot) = self.tys.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.emit_func(task);
        }
        self.errors.borrow().len() == prev_err_len
    }

    fn emit_func(&mut self, FTask { file, id, ct }: FTask) {
        let func = &mut self.tys.ins.funcs[id];
        debug_assert_eq!(func.file, file);
        let cct = self.ct.active();
        debug_assert_eq!(cct, ct);
        func.comp_state[cct as usize] = CompState::Compiled;
        let sig = func.sig.expect("to emmit only concrete functions");
        let ast = &self.files[file.index()];
        let expr = func.expr.get(ast);

        self.pool.push_ci(file, Some(sig.ret), 0, &mut self.ci);
        let prev_err_len = self.errors.borrow().len();

        log::info!("{}", self.ast_display(expr));

        let &Expr::Closure { body, args, pos, .. } = expr else {
            unreachable!("{}", self.ast_display(expr))
        };
        self.ci.pos.push(pos);

        let mut tys = sig.args.args();
        let mut args = args.iter();
        while let Some(aty) = tys.next(self.tys) {
            let arg = args.next().unwrap();
            match aty {
                Arg::Type(ty) => {
                    self.ci.scope.vars.push(Variable::new(
                        arg.id,
                        ty,
                        false,
                        NEVER,
                        &mut self.ci.nodes,
                    ));
                }
                Arg::Value(ty) => {
                    let mut deps = Vc::from([VOID]);
                    if ty.loc(self.tys) == Loc::Stack && self.tys.size_of(ty) <= 16 {
                        deps.push(MEM);
                    }
                    // TODO: whe we not using the deps?
                    let value = self.ci.nodes.new_node_nop(ty, Kind::Arg, deps);
                    let ptr = ty.loc(self.tys) == Loc::Stack;
                    self.ci.scope.vars.push(Variable::new(
                        arg.id,
                        ty,
                        ptr,
                        value,
                        &mut self.ci.nodes,
                    ));
                    if ty.loc(self.tys) == Loc::Stack {
                        self.ci.nodes[value].aclass = self.ci.scope.aclasses.len() as _;
                        self.ci.scope.aclasses.push(AClass::new(&mut self.ci.nodes));
                    }
                }
            }
        }

        if self.expr(body).is_some() {
            if sig.ret == ty::Id::VOID {
                self.expr(&Expr::Return { pos: body.pos(), val: None });
            } else {
                self.report(
                    body.pos(),
                    fa!(
                        "expected all paths in the fucntion to return \
                        or the return type to be 'void' (return type is '{}')",
                        self.ty_display(sig.ret),
                    ),
                );
            }
        }

        self.ci.scope.vars.drain(..).for_each(|v| v.remove_ignore_arg(&mut self.ci.nodes));

        if self.finalize(prev_err_len) {
            let backend = if !cct { &mut *self.backend } else { &mut *self.ct_backend };
            backend.emit_body(id, &mut self.ci.nodes, self.tys, self.files);
        }

        self.ci.pos.pop();
        self.pool.pop_ci(&mut self.ci);
    }

    fn finalize(&mut self, prev_err_len: usize) -> bool {
        use {AssertKind as AK, CondOptRes as CR};

        self.ci.finalize(&mut self.pool.nid_stack, self.tys, self.files);

        //let mut to_remove = vec![];
        for (id, node) in self.ci.nodes.iter() {
            let Kind::Assert { kind, pos } = node.kind else { continue };

            let res = self.ci.nodes.try_match_cond(id);

            // TODO: highlight the pin position
            let msg = match (kind, res) {
                (AK::UnwrapCheck, CR::Known { value: false, .. }) => {
                    "unwrap is not needed since the value is (provably) never null, \
                    remove it, or replace with '@as(<expr_ty>, <opt_expr>)'"
                }
                (AK::UnwrapCheck, CR::Known { value: true, .. }) => {
                    "unwrap is incorrect since the value is (provably) always null, \
                    make sure your logic is correct"
                }
                (AK::NullCheck, CR::Known { value: true, .. }) => {
                    "the value is always null, some checks might need to be inverted"
                }
                (AK::NullCheck, CR::Unknown) => {
                    "can't prove the value is not 'null', \
                    use '@unwrap(<opt>)' if you believe compiler is stupid, \
                    or explicitly check for null and handle it \
                    ('if <opt> == null { /* handle */ } else { /* use opt */ }')"
                }
                _ => unreachable!(),
            };
            self.report(pos, msg);
        }

        for &node in self.ci.nodes[NEVER].inputs.iter() {
            if self.ci.nodes[node].kind == Kind::Return
                && self.ci.nodes[self.ci.nodes.aclass_index(self.ci.nodes[node].inputs[1]).1].kind
                    == Kind::Stck
            {
                self.report(
                    self.ci.nodes[node].pos,
                    "returning value with local provenance \
                    (pointer will be invalid after function returns)",
                );
                self.report(
                    self.ci.nodes[self.ci.nodes.aclass_index(self.ci.nodes[node].inputs[1]).1].pos,
                    "...the pointer points to stack allocation created here",
                );
            }
        }

        if self.errors.borrow().len() == prev_err_len {
            self.ci.nodes.check_final_integrity(self.ty_display(ty::Id::VOID));
            self.ci.nodes.graphviz(self.ty_display(ty::Id::VOID));
            self.ci.nodes.gcm(&mut self.pool.nid_stack, &mut self.pool.nid_set);
            self.ci.nodes.check_loop_depth_integrity(self.ty_display(ty::Id::VOID));
            self.ci.nodes.basic_blocks();
            self.ci.nodes.graphviz(self.ty_display(ty::Id::VOID));
        } else {
            //self.ci.nodes.graphviz_in_browser(self.ty_display(ty::Id::VOID));
        }

        self.errors.borrow().len() == prev_err_len
    }

    fn ty(&mut self, expr: &Expr) -> ty::Id {
        self.ty_in(self.ci.file, expr)
    }

    fn ty_in(&mut self, file: Module, expr: &Expr) -> ty::Id {
        self.parse_ty(file, expr, None, self.files)
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(self.tys, self.files, ty)
    }

    fn ast_display(&self, ast: &'a Expr<'a>) -> parser::Display<'a> {
        parser::Display::new(&self.file().file, ast)
    }

    #[must_use]
    #[track_caller]
    fn binop_ty(
        &mut self,
        pos: Pos,
        lhs: &mut Value,
        rhs: &mut Value,
        op: TokenKind,
    ) -> (ty::Id, Nid) {
        if let Some(upcasted) = lhs.ty.try_upcast(rhs.ty) {
            let to_correct = if lhs.ty != upcasted {
                Some((lhs, rhs))
            } else if rhs.ty != upcasted {
                Some((rhs, lhs))
            } else {
                None
            };

            if let Some((oper, other)) = to_correct {
                if self.tys.size_of(upcasted) > self.tys.size_of(oper.ty) {
                    self.extend(oper, upcasted);
                }
                if matches!(op, TokenKind::Add | TokenKind::Sub)
                    && let Some(elem) = self.tys.base_of(upcasted)
                {
                    let cnst = self.ci.nodes.new_const(ty::Id::INT, self.tys.size_of(elem));
                    oper.id = self.ci.nodes.new_node(
                        upcasted,
                        Kind::BinOp { op: TokenKind::Mul },
                        [VOID, oper.id, cnst],
                        self.tys,
                    );
                    return (upcasted, self.ci.nodes.aclass_index(other.id).1);
                }
            }

            (upcasted, VOID)
        } else {
            let ty = self.ty_display(lhs.ty);
            let expected = self.ty_display(rhs.ty);
            self.report(pos, fa!("'{ty} {op} {expected}' is not supported"));
            (ty::Id::NEVER, VOID)
        }
    }

    fn wrap_in_opt(&mut self, pos: Pos, val: &mut Value) {
        debug_assert!(!val.var);

        let was_ptr = val.ptr;
        let oty = self.tys.make_opt(val.ty);

        if let Some((uninit, ..)) = self.tys.nieche_of(val.ty) {
            self.strip_ptr(val);
            val.ty = oty;
            assert!(!uninit, "TODO");
            return;
        }

        let OptLayout { flag_ty, flag_offset, payload_offset } = self.tys.opt_layout(val.ty);

        match oty.loc(self.tys) {
            Loc::Reg => {
                self.strip_ptr(val);
                // registers have inverted offsets so that accessing the inner type is a noop
                let flag_offset = self.tys.size_of(oty) * 8 - flag_offset * 8 - 1;
                let fill = self.ci.nodes.new_const(oty, 1i64 << flag_offset);
                val.id = self.ci.nodes.new_node(
                    oty,
                    Kind::BinOp { op: TokenKind::Bor },
                    [VOID, val.id, fill],
                    self.tys,
                );
                val.ty = oty;
            }
            Loc::Stack => {
                self.strip_ptr(val);
                let stack = self.new_stack(pos, oty);
                let fill = self.ci.nodes.new_const(flag_ty, 1);
                self.store_mem(stack, flag_ty, fill);
                let off = self.offset(stack, payload_offset);
                self.store_mem(off, val.ty, val.id);
                val.id = stack;
                val.ptr = true;
                val.ty = oty;
            }
        }

        if !was_ptr {
            self.strip_ptr(val);
        }
    }

    fn implicit_unwrap(&mut self, pos: Pos, opt: &mut Value) {
        self.unwrap_low(pos, opt, AssertKind::NullCheck);
    }

    fn explicit_unwrap(&mut self, pos: Pos, opt: &mut Value) {
        self.unwrap_low(pos, opt, AssertKind::UnwrapCheck);
    }

    fn unwrap_low(&mut self, pos: Pos, opt: &mut Value, kind: AssertKind) {
        let Some(ty) = self.tys.inner_of(opt.ty) else { return };
        let null_check = self.gen_null_check(*opt, ty, TokenKind::Eq);

        let oty = mem::replace(&mut opt.ty, ty);
        self.unwrap_opt_unchecked(ty, oty, opt);

        // TODO: extract the if check int a fucntion
        let ass = self.ci.nodes.new_node_nop(oty, Kind::Assert { kind, pos }, [
            self.ci.ctrl.get(),
            null_check,
            opt.id,
        ]);
        self.ci.nodes.pass_aclass(self.ci.nodes.aclass_index(opt.id).1, ass);
        opt.id = ass;
    }

    fn unwrap_opt_unchecked(&mut self, ty: ty::Id, oty: ty::Id, opt: &mut Value) {
        if self.tys.nieche_of(ty).is_some() {
            return;
        }

        let OptLayout { payload_offset, .. } = self.tys.opt_layout(ty);

        match oty.loc(self.tys) {
            Loc::Reg => {}
            Loc::Stack => {
                opt.id = self.offset(opt.id, payload_offset);
            }
        }
    }

    fn gen_null_check(&mut self, mut cmped: Value, ty: ty::Id, op: TokenKind) -> Nid {
        let OptLayout { flag_ty, flag_offset, .. } = self.tys.opt_layout(ty);
        debug_assert!(cmped.ty.is_optional());

        match cmped.ty.loc(self.tys) {
            Loc::Reg => {
                self.strip_ptr(&mut cmped);
                let inps = [VOID, cmped.id, self.ci.nodes.new_const(cmped.ty, 0)];
                self.ci.nodes.new_node(ty::Id::BOOL, Kind::BinOp { op }, inps, self.tys)
            }
            Loc::Stack => {
                cmped.id = self.offset(cmped.id, flag_offset);
                cmped.ty = flag_ty;
                debug_assert!(cmped.ptr);
                self.strip_ptr(&mut cmped);
                let inps = [VOID, cmped.id, self.ci.nodes.new_const(flag_ty, 0)];
                self.ci.nodes.new_node(ty::Id::BOOL, Kind::BinOp { op }, inps, self.tys)
            }
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
        if let Some(upcasted) = src.ty.try_upcast(expected)
            && upcasted == expected
        {
            if src.ty.is_never() {
                return true;
            }

            if src.ty != upcasted {
                if let Some(inner) = self.tys.inner_of(upcasted) {
                    if inner != src.ty {
                        self.assert_ty(pos, src, inner, hint);
                    }
                    self.wrap_in_opt(pos, src);
                } else {
                    debug_assert!(
                        src.ty.is_integer() || src.ty == ty::Id::NEVER,
                        "{} {}",
                        self.ty_display(src.ty),
                        self.ty_display(upcasted)
                    );
                    debug_assert!(
                        upcasted.is_integer() || src.ty == ty::Id::NEVER,
                        "{} {}",
                        self.ty_display(src.ty),
                        self.ty_display(upcasted)
                    );
                    self.extend(src, upcasted);
                }
            }
            true
        } else {
            if let Some(inner) = self.tys.inner_of(src.ty)
                && inner.try_upcast(expected) == Some(expected)
            {
                self.implicit_unwrap(pos, src);
                return self.assert_ty(pos, src, expected, hint);
            }

            let ty = self.ty_display(src.ty);

            let expected = self.ty_display(expected);
            self.report(pos, fa!("expected {hint} to be of type {expected}, got {ty}"));
            false
        }
    }

    fn extend(&mut self, value: &mut Value, to: ty::Id) {
        self.strip_ptr(value);
        let inps = [VOID, value.id];
        *value =
            self.ci.nodes.new_node_lit(to, Kind::UnOp { op: TokenKind::Number }, inps, self.tys);
        value.ty = to;
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl core::fmt::Display) {
        let mut buf = self.errors.borrow_mut();
        write!(buf, "{}", self.file().report(pos, msg)).unwrap();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: impl Display) {
        log::info!("{ast:#?}");
        self.report(ast.pos(), fa!("compiler does not (yet) know how to handle ({hint})"));
    }

    fn file(&self) -> &'a parser::Ast {
        &self.files[self.ci.file.index()]
    }
}

impl TypeParser for Codegen<'_> {
    fn tys(&mut self) -> &mut Types {
        self.tys
    }

    fn eval_const(&mut self, file: Module, expr: &Expr, ret: ty::Id) -> u64 {
        self.ct.activate();
        let mut scope = mem::take(&mut self.ci.scope.vars);
        self.pool.push_ci(file, Some(ret), self.tys.tasks.len(), &mut self.ci);
        self.ci.scope.vars = scope;

        let prev_err_len = self.errors.borrow().len();

        self.expr(&Expr::Return { pos: expr.pos(), val: Some(expr) });

        scope = mem::take(&mut self.ci.scope.vars);

        let res =
            if self.finalize(prev_err_len) { self.emit_and_eval(file, ret, &mut []) } else { 1 };

        self.pool.pop_ci(&mut self.ci);
        self.ci.scope.vars = scope;

        self.ct.deactivate();
        res
    }

    fn infer_type(&mut self, expr: &Expr) -> ty::Id {
        self.pool.save_ci(&self.ci);
        let ty = self.expr(expr).map_or(ty::Id::NEVER, |v| v.ty);

        self.pool.restore_ci(&mut self.ci);
        ty
    }

    fn on_reuse(&mut self, existing: ty::Id) {
        let state_slot = self.ct.active() as usize;
        if let ty::Kind::Func(id) = existing.expand()
            && let func = &mut self.tys.ins.funcs[id]
            && let CompState::Queued(idx) = func.comp_state[state_slot]
            && idx < self.tys.tasks.len()
        {
            func.comp_state[state_slot] = CompState::Queued(self.tys.tasks.len());
            let task = self.tys.tasks[idx].take();
            self.tys.tasks.push(task);
        }
    }

    fn eval_global(&mut self, file: Module, name: Ident, expr: &Expr) -> ty::Id {
        self.ct.activate();

        let gid = self.tys.ins.globals.push(Global { file, name, ..Default::default() });

        self.pool.push_ci(file, None, self.tys.tasks.len(), &mut self.ci);
        let prev_err_len = self.errors.borrow().len();

        self.expr(&(Expr::Return { pos: expr.pos(), val: Some(expr) }));

        let ret = self.ci.ret.expect("for return type to be infered");
        if self.finalize(prev_err_len) {
            let mut mem = vec![0u8; self.tys.size_of(ret) as usize];
            self.emit_and_eval(file, ret, &mut mem);
            self.tys.ins.globals[gid].data = mem;
        }

        self.pool.pop_ci(&mut self.ci);
        self.tys.ins.globals[gid].ty = ret;

        self.ct.deactivate();
        gid.into()
    }

    fn report(&self, file: Module, pos: Pos, msg: impl Display) -> ty::Id {
        let mut buf = self.errors.borrow_mut();
        write!(buf, "{}", self.files[file.index()].report(pos, msg)).unwrap();
        ty::Id::NEVER
    }

    fn find_local_ty(&mut self, ident: Ident) -> Option<ty::Id> {
        self.ci.scope.vars.iter().rfind(|v| (v.id == ident && v.value() == NEVER)).map(|v| v.ty)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::{hbvm::HbvmBackend, CodegenCtx},
        crate::ty,
        alloc::{string::String, vec::Vec},
        core::fmt::Write,
    };

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        _ = log::set_logger(&crate::fs::Logger);
        log::set_max_level(log::LevelFilter::Info);
        //log::set_max_level(log::LevelFilter::Trace);

        let mut ctx = CodegenCtx::default();
        let (ref files, embeds) = crate::test_parse_files(ident, input, &mut ctx.parser);
        let mut backend = HbvmBackend::default();
        let mut codegen = super::Codegen::new(&mut backend, files, &mut ctx);
        codegen.push_embeds(embeds);

        codegen.generate(ty::Module::MAIN);

        {
            let errors = codegen.errors.borrow();
            if !errors.is_empty() {
                output.push_str(&errors);
                return;
            }
        }

        let mut out = Vec::new();
        codegen.assemble(&mut out);

        let err = codegen.disasm(output, &out);
        if let Err(e) = err {
            writeln!(output, "!!! asm is invalid: {e}").unwrap();
        } else {
            log::info!("================ running {ident} ==============");
            log::trace!("{output}");
            super::hbvm::test_run_vm(&out, output);
        }
    }

    crate::run_tests! { generate:
        // Tour Examples
        main_fn;
        arithmetic;
        floating_point_arithmetic;
        functions;
        comments;
        if_statements;
        variables;
        loops;
        pointers;
        nullable_types;
        structs;
        hex_octal_binary_literals;
        struct_operators;
        global_variables;
        constants;
        directives;
        c_strings;
        struct_patterns;
        arrays;
        inline;
        idk;
        generic_functions;
        die;

        // Incomplete Examples;
        //comptime_pointers;
        generic_types;
        fb_driver;

        // Purely Testing Examples;
        different_function_destinations;
        triggering_store_in_divergent_branch;
        wrong_dead_code_elimination;
        memory_swap;
        very_nested_loops;
        generic_type_mishap;
        storing_into_nullable_struct;
        scheduling_block_did_dirty;
        null_check_returning_small_global;
        null_check_in_the_loop;
        stack_provenance;
        advanced_floating_point_arithmetic;
        nullable_structure;
        needless_unwrap;
        inlining_issues;
        null_check_test;
        only_break_loop;
        reading_idk;
        nonexistent_ident_import;
        big_array_crash;
        returning_global_struct;
        small_struct_bitcast;
        small_struct_assignment;
        intcast_store;
        string_flip;
        signed_to_unsigned_upcast;
        wide_ret;
        comptime_min_reg_leak;
        different_types;
        struct_return_from_module_function;
        sort_something_viredly;
        struct_in_register;
        comptime_function_from_another_file;
        inline_test;
        inlined_generic_functions;
        some_generic_code;
        integer_inference_issues;
        writing_into_string;
        request_page;
        tests_ptr_to_ptr_copy;
        global_variable_wiredness;
        inline_return_stack;

        // Just Testing Optimizations;
        const_folding_with_arg;
        branch_assignments;
        exhaustive_loop_testing;
        pointer_opts;
        conditional_stores;
        loop_stores;
        dead_code_in_loop;
        infinite_loop_after_peephole;
        aliasing_overoptimization;
        global_aliasing_overptimization;
        overwrite_aliasing_overoptimization;
        more_if_opts;
        optional_from_eca;
        returning_optional_issues;
    }
}
