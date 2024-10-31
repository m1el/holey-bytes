use {
    self::{hbvm::Comptime, strong_ref::StrongRef},
    crate::{
        ctx_map::CtxEntry,
        debug,
        lexer::{self, TokenKind},
        parser::{
            self,
            idfl::{self},
            CtorField, Expr, FileId, Pos,
        },
        task,
        ty::{self, Arg, ArrayLen, Loc, Tuple},
        utils::{BitSet, Vc},
        FTask, Func, Global, Ident, Offset, OffsetIter, OptLayout, Reloc, Sig, StringRef, SymKey,
        TypeParser, TypedReloc, Types,
    },
    alloc::{string::String, vec::Vec},
    core::{
        assert_matches::debug_assert_matches,
        cell::RefCell,
        fmt::{self, Debug, Display, Write},
        format_args as fa, mem,
        ops::{self, Deref},
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
type AClassId = u16;

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
struct Nodes {
    values: Vec<Result<Node, (Nid, debug::Trace)>>,
    visited: BitSet,
    free: Nid,
    lookup: Lookup,
    complete: bool,
}

impl Default for Nodes {
    fn default() -> Self {
        Self {
            values: Default::default(),
            free: Nid::MAX,
            lookup: Default::default(),
            visited: Default::default(),
            complete: false,
        }
    }
}

impl Nodes {
    fn loop_depth(&mut self, target: Nid) -> LoopDepth {
        if self[target].loop_depth != 0 {
            return self[target].loop_depth;
        }

        self[target].loop_depth = match self[target].kind {
            Kind::Entry | Kind::Then | Kind::Else | Kind::Call { .. } | Kind::Return | Kind::If => {
                let dpth = self.loop_depth(self[target].inputs[0]);
                if self[target].loop_depth != 0 {
                    return self[target].loop_depth;
                }
                dpth
            }
            Kind::Region => {
                let l = self.loop_depth(self[target].inputs[0]);
                let r = self.loop_depth(self[target].inputs[1]);
                debug_assert_eq!(l, r);
                l
            }
            Kind::Loop => {
                let depth = Self::loop_depth(self, self[target].inputs[0]) + 1;
                self[target].loop_depth = depth;
                let mut cursor = self[target].inputs[1];
                while cursor != target {
                    self[cursor].loop_depth = depth;
                    let next = if self[cursor].kind == Kind::Region {
                        self.loop_depth(self[cursor].inputs[0]);
                        self[cursor].inputs[1]
                    } else {
                        self.idom(cursor)
                    };
                    debug_assert_ne!(next, VOID);
                    if matches!(self[cursor].kind, Kind::Then | Kind::Else) {
                        let other = *self[next]
                            .outputs
                            .iter()
                            .find(|&&n| self[n].kind != self[cursor].kind)
                            .unwrap();
                        if self[other].loop_depth == 0 {
                            self[other].loop_depth = depth - 1;
                        }
                    }
                    cursor = next;
                }
                depth
            }
            Kind::Start | Kind::End => 1,
            u => unreachable!("{u:?}"),
        };

        self[target].loop_depth
    }

    fn idepth(&mut self, target: Nid) -> IDomDepth {
        if target == VOID {
            return 0;
        }
        if self[target].depth == 0 {
            self[target].depth = match self[target].kind {
                Kind::End | Kind::Start => unreachable!("{:?}", self[target].kind),
                Kind::Region => {
                    self.idepth(self[target].inputs[0]).max(self.idepth(self[target].inputs[1]))
                }
                _ => self.idepth(self[target].inputs[0]),
            } + 1;
        }
        self[target].depth
    }

    fn fix_loops(&mut self) {
        'o: for l in self[LOOPS].outputs.clone() {
            let mut cursor = self[l].inputs[1];
            while cursor != l {
                if self[cursor].kind == Kind::If
                    && self[cursor]
                        .outputs
                        .clone()
                        .into_iter()
                        .any(|b| self.loop_depth(b) < self.loop_depth(cursor))
                {
                    continue 'o;
                }
                cursor = self.idom(cursor);
            }

            self[l].outputs.push(NEVER);
            self[NEVER].inputs.push(l);
        }
    }

    fn push_up_impl(&mut self, node: Nid) {
        if !self.visited.set(node) {
            return;
        }

        for i in 1..self[node].inputs.len() {
            let inp = self[node].inputs[i];
            if !self[inp].kind.is_pinned() {
                self.push_up_impl(inp);
            }
        }

        if self[node].kind.is_pinned() {
            return;
        }

        let mut deepest = VOID;
        for i in 0..self[node].inputs.len() {
            let inp = self[node].inputs[i];
            if self.idepth(inp) > self.idepth(deepest) {
                if matches!(self[inp].kind, Kind::Call { .. }) {
                    deepest = inp;
                } else {
                    deepest = self.idom(inp);
                }
            }
        }

        if deepest == self[node].inputs[0] {
            return;
        }

        let index = self[0].outputs.iter().position(|&p| p == node).unwrap();
        self[0].outputs.remove(index);
        self[node].inputs[0] = deepest;
        debug_assert!(
            !self[deepest].outputs.contains(&node)
                || matches!(self[deepest].kind, Kind::Call { .. }),
            "{node} {:?} {deepest} {:?}",
            self[node],
            self[deepest]
        );
        self[deepest].outputs.push(node);
    }

    fn collect_rpo(&mut self, node: Nid, rpo: &mut Vec<Nid>) {
        if !self.is_cfg(node) || !self.visited.set(node) {
            return;
        }

        for i in 0..self[node].outputs.len() {
            self.collect_rpo(self[node].outputs[i], rpo);
        }
        rpo.push(node);
    }

    fn push_up(&mut self, rpo: &mut Vec<Nid>) {
        debug_assert!(rpo.is_empty());
        self.collect_rpo(VOID, rpo);

        for &node in rpo.iter().rev() {
            self.loop_depth(node);
            for i in 0..self[node].inputs.len() {
                self.push_up_impl(self[node].inputs[i]);
            }

            if matches!(self[node].kind, Kind::Loop | Kind::Region) {
                for i in 0..self[node].outputs.len() {
                    let usage = self[node].outputs[i];
                    if self[usage].kind == Kind::Phi {
                        self.push_up_impl(usage);
                    }
                }
            }
        }

        debug_assert_eq!(
            self.iter()
                .map(|(n, _)| n)
                .filter(|&n| !self.visited.get(n)
                    && !matches!(self[n].kind, Kind::Arg | Kind::Mem | Kind::Loops))
                .collect::<Vec<_>>(),
            vec![],
            "{:?}",
            self.iter()
                .filter(|&(n, nod)| !self.visited.get(n)
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

    fn push_down(&mut self, node: Nid) {
        if !self.visited.set(node) {
            return;
        }

        for usage in self[node].outputs.clone() {
            if self.is_forward_edge(usage, node) && self[node].kind == Kind::Stre {
                self.push_down(usage);
            }
        }

        for usage in self[node].outputs.clone() {
            if self.is_forward_edge(usage, node) {
                self.push_down(usage);
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
            min = self.find_antideps(node, min);
        }

        if self[node].kind == Kind::Stre {
            self[node].antidep = self[node].inputs[0];
        }

        if self[min].kind.ends_basic_block() {
            min = self.idom(min);
        }

        self.check_dominance(node, min, true);

        let prev = self[node].inputs[0];
        debug_assert!(self.idepth(min) >= self.idepth(prev));
        let index = self[prev].outputs.iter().position(|&p| p == node).unwrap();
        self[prev].outputs.remove(index);
        self[node].inputs[0] = min;
        self[min].outputs.push(node);
    }

    fn find_antideps(&mut self, load: Nid, mut min: Nid) -> Nid {
        debug_assert!(self[load].kind == Kind::Load);

        let (aclass, _) = self.aclass_index(self[load].inputs[1]);

        let mut cursor = min;
        while cursor != self[load].inputs[0] {
            self[cursor].antidep = load;
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
                    while cursor != self[out].antidep {
                        if self[cursor].antidep == load {
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
                    while cursor != self[out].antidep {
                        if self[cursor].antidep == load {
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
        self[from].outputs.push(to);
        self[to].inputs.push(from);
    }

    fn use_block(&mut self, target: Nid, from: Nid) -> Nid {
        if self[from].kind != Kind::Phi {
            return self.idom(from);
        }

        let index = self[from].inputs.iter().position(|&n| n == target).unwrap();
        self[self[from].inputs[0]].inputs[index - 1]
    }

    fn idom(&mut self, target: Nid) -> Nid {
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

    fn common_dom(&mut self, mut a: Nid, mut b: Nid) -> Nid {
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
    ) {
        for (i, (to_value, from_value)) in to.vars.iter_mut().zip(from.vars.iter_mut()).enumerate()
        {
            debug_assert_eq!(to_value.ty, from_value.ty);
            if to_value.value() != from_value.value() {
                self.load_loop_var(i, from_value, loops);
                self.load_loop_var(i, to_value, loops);
                if to_value.value() != from_value.value() {
                    let inps = [ctrl.get(), from_value.value(), to_value.value()];
                    to_value.set_value_remove(self.new_node(from_value.ty, Kind::Phi, inps), self);
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
                        .set_remove(self.new_node(ty::Id::VOID, Kind::Phi, inps), self);
                }
            }
        }
    }

    fn graphviz_low(
        &self,
        tys: &Types,
        files: &[parser::Ast],
        out: &mut String,
    ) -> core::fmt::Result {
        use core::fmt::Write;

        writeln!(out)?;
        writeln!(out, "digraph G {{")?;
        writeln!(out, "rankdir=BT;")?;
        writeln!(out, "concentrate=true;")?;
        writeln!(out, "compound=true;")?;

        for (i, node) in self.iter() {
            let color = match () {
                _ if node.lock_rc == Nid::MAX => "orange",
                _ if node.lock_rc == Nid::MAX - 1 => "blue",
                _ if node.lock_rc != 0 => "red",
                _ if node.outputs.is_empty() => "purple",
                _ if node.is_mem() => "green",
                _ if self.is_cfg(i) => "yellow",
                _ => "white",
            };

            if node.ty != ty::Id::VOID {
                writeln!(
                    out,
                    " node{i}[label=\"{i} {} {} {} {}\" color={color}]",
                    node.kind,
                    ty::Display::new(tys, files, node.ty),
                    node.aclass,
                    node.mem,
                )?;
            } else {
                writeln!(
                    out,
                    " node{i}[label=\"{i} {} {} {}\" color={color}]",
                    node.kind, node.aclass, node.mem,
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

    fn graphviz(&self, tys: &Types, files: &[parser::Ast]) {
        let out = &mut String::new();
        _ = self.graphviz_low(tys, files, out);
        log::info!("{out}");
    }

    fn graphviz_in_browser(&self, _tys: &Types, _files: &[parser::Ast]) {
        #[cfg(all(debug_assertions, feature = "std"))]
        {
            let out = &mut String::new();
            _ = self.graphviz_low(_tys, _files, out);
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

    fn gcm(&mut self, rpo: &mut Vec<Nid>) {
        self.fix_loops();
        self.visited.clear(self.values.len());
        self.push_up(rpo);
        self.visited.clear(self.values.len());
        self.push_down(VOID);
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = Nid::MAX;
        self.complete = false;
    }

    fn new_node_nop(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Nid {
        let node =
            Node { ralloc_backref: u16::MAX, inputs: inps.into(), kind, ty, ..Default::default() };

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

    fn new_node(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Nid {
        let id = self.new_node_nop(ty, kind, inps);
        if let Some(opt) = self.peephole(id) {
            debug_assert_ne!(opt, id);
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
        self.new_node_lit(ty, Kind::CInt { value: value.into() }, [VOID])
    }

    fn new_node_lit(&mut self, ty: ty::Id, kind: Kind, inps: impl Into<Vc>) -> Value {
        Value::new(self.new_node(ty, kind, inps)).ty(ty)
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

    fn late_peephole(&mut self, target: Nid) -> Option<Nid> {
        if let Some(id) = self.peephole(target) {
            self.replace(target, id);
            return None;
        }
        None
    }

    fn iter_peeps(&mut self, mut fuel: usize, stack: &mut Vec<Nid>) {
        debug_assert!(stack.is_empty());

        self.iter()
            .filter_map(|(id, node)| node.kind.is_peeped().then_some(id))
            .collect_into(stack);
        stack.iter().for_each(|&s| self.lock(s));

        while fuel != 0
            && let Some(node) = stack.pop()
        {
            fuel -= 1;

            if self[node].outputs.is_empty() {
                self.push_adjacent_nodes(node, stack);
            }

            if self.unlock_remove(node) {
                continue;
            }

            if let Some(new) = self.peephole(node) {
                self.replace(node, new);
                self.push_adjacent_nodes(new, stack);
            }

            debug_assert_matches!(
                self.iter()
                    .find(|(i, n)| n.lock_rc != 0 && n.kind.is_peeped() && !stack.contains(i)),
                None
            );
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
            if self.values[i as usize].is_ok() && self[i].kind.is_peeped() && self[i].lock_rc == 0 {
                stack.push(i);
            }
        }

        self[of].peep_triggers = Vc::default();
        stack.iter().skip(prev_len).for_each(|&n| self.lock(n));
    }

    pub fn aclass_index(&self, region: Nid) -> (usize, Nid) {
        (self[region].aclass as _, self[region].mem)
    }

    fn peephole(&mut self, target: Nid) -> Option<Nid> {
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
                    mem::swap(&mut lhs, &mut rhs);
                    changed = true;
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
                        let new_rhs = self.new_const(ty, op.apply_binop(av, bv, is_float));
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
                    let new_rhs = self.new_const(ty, value + 1);
                    return Some(self.new_node(ty, K::BinOp { op: T::Mul }, [ctrl, rhs, new_rhs]));
                }

                if op == T::Sub
                    && self[lhs].kind == (K::BinOp { op: T::Add })
                    && let K::CInt { value: a } = self[rhs].kind
                    && let K::CInt { value: b } = self[self[lhs].inputs[2]].kind
                {
                    let new_rhs = self.new_const(ty, b - a);
                    return Some(self.new_node(ty, K::BinOp { op: T::Add }, [
                        ctrl,
                        self[lhs].inputs[1],
                        new_rhs,
                    ]));
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
                let &[_, oper] = self[target].inputs.as_slice() else { unreachable!() };
                let ty = self[target].ty;

                let is_float = self[oper].ty.is_float();

                if let K::CInt { value } = self[oper].kind {
                    return Some(self.new_const(ty, op.apply_unop(value, is_float)));
                }
            }
            K::If => {
                if self[target].ty == ty::Id::VOID {
                    let &[ctrl, cond] = self[target].inputs.as_slice() else { unreachable!() };
                    if let K::CInt { value } = self[cond].kind {
                        let ty = if value == 0 {
                            ty::Id::LEFT_UNREACHABLE
                        } else {
                            ty::Id::RIGHT_UNREACHABLE
                        };
                        return Some(self.new_node_nop(ty, K::If, [ctrl, cond]));
                    }

                    'b: {
                        let mut cursor = ctrl;
                        let ty = loop {
                            if cursor == ENTRY {
                                break 'b;
                            }

                            // TODO: do more inteligent checks on the condition
                            if self[cursor].kind == Kind::Then
                                && self[self[cursor].inputs[0]].inputs[1] == cond
                            {
                                break ty::Id::RIGHT_UNREACHABLE;
                            }
                            if self[cursor].kind == Kind::Else
                                && self[self[cursor].inputs[0]].inputs[1] == cond
                            {
                                break ty::Id::LEFT_UNREACHABLE;
                            }

                            cursor = self.idom(cursor);
                        };

                        return Some(self.new_node_nop(ty, K::If, [ctrl, cond]));
                    }
                }
            }
            K::Then => {
                if self[self[target].inputs[0]].ty == ty::Id::LEFT_UNREACHABLE {
                    return Some(NEVER);
                } else if self[self[target].inputs[0]].ty == ty::Id::RIGHT_UNREACHABLE {
                    return Some(self[self[target].inputs[0]].inputs[0]);
                }
            }
            K::Else => {
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
                    if self[n].kind != Kind::Stre || self[n].inputs.len() != 4 {
                        new_inps.push(n);
                        continue;
                    }

                    let mut cursor = n;
                    let class = self.aclass_index(self[cursor].inputs[2]);

                    if self[class.1].kind != Kind::Stck {
                        new_inps.push(n);
                        continue;
                    }

                    cursor = self[cursor].inputs[3];
                    while cursor != MEM {
                        if self.aclass_index(self[cursor].inputs[2]) != class
                            || self[cursor].inputs.len() != 4
                        {
                            new_inps.push(n);
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
            K::Stck => {
                if let &[mut a, mut b] = self[target].outputs.as_slice() {
                    if self[a].kind == Kind::Load {
                        mem::swap(&mut a, &mut b);
                    }

                    if matches!(self[a].kind, Kind::Call { .. })
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
                            self[ele].peep_triggers.push(target);
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
                        unidentifed.remove(index);
                        saved.push(contact_point);
                        first_store = cursor;
                        cursor = *self[cursor].inputs.get(3).unwrap_or(&MEM);

                        if unidentifed.is_empty() {
                            break;
                        }
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

                    if !unidentifed.is_empty() {
                        break 'eliminate;
                    }

                    // FIXME: when the loads and stores become parallel we will need to get saved
                    // differently
                    let mut prev_store = store;
                    for mut oper in saved.into_iter().rev() {
                        let mut region = region;
                        if let Kind::BinOp { op } = self[oper].kind {
                            debug_assert_eq!(self[oper].outputs.len(), 1);
                            debug_assert_eq!(self[self[oper].outputs[0]].kind, Kind::Stre);
                            region = self.new_node(self[oper].ty, Kind::BinOp { op }, [
                                VOID,
                                region,
                                self[oper].inputs[2],
                            ]);
                            oper = self[oper].outputs[0];
                        }

                        let mut inps = self[oper].inputs.clone();
                        debug_assert_eq!(inps.len(), 4);
                        inps[2] = region;
                        inps[3] = prev_store;
                        prev_store = self.new_node(self[oper].ty, Kind::Stre, inps);
                    }

                    return Some(prev_store);
                }

                if value != VOID
                    && self[target].inputs.len() == 4
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
                let &[_, region, store] = self[target].inputs.as_slice() else { unreachable!() };

                if self[store].kind == Kind::Stre
                    && self[store].inputs[2] == region
                    && self[store].ty == self[target].ty
                    && self[store]
                        .outputs
                        .iter()
                        .all(|&n| !matches!(self[n].kind, Kind::Call { .. }))
                {
                    return Some(self[store].inputs[1]);
                }

                let (index, reg) = self.aclass_index(region);
                if index != 0 && self[reg].kind == Kind::Stck {
                    let mut cursor = store;
                    while cursor != MEM
                        && self[cursor].kind == Kind::Stre
                        && self[cursor].inputs[1] != VOID
                        && self[cursor]
                            .outputs
                            .iter()
                            .all(|&n| !matches!(self[n].kind, Kind::Call { .. }))
                    {
                        if self[cursor].inputs[2] == region && self[cursor].ty == self[target].ty {
                            return Some(self[cursor].inputs[1]);
                        }
                        cursor = self[cursor].inputs[3];
                    }
                }
            }
            K::Loop => {
                if self[target].inputs[0] == NEVER {
                    return Some(NEVER);
                }

                if self[target].inputs[1] == NEVER {
                    self.lock(target);
                    for o in self[target].outputs.clone() {
                        if self[o].kind == Kind::Phi {
                            self.replace(o, self[o].inputs[1]);
                        }
                    }
                    self.unlock(target);
                    return Some(self[target].inputs[0]);
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
        debug_assert_ne!(target, with, "{:?}", self[target]);
        for out in self[target].outputs.clone() {
            let index = self[out].inputs.iter().position(|&p| p == target).unwrap();
            self.modify_input(out, index, with);
        }
    }

    fn modify_input(&mut self, target: Nid, inp_index: usize, with: Nid) -> Nid {
        self.remove_node_lookup(target);
        debug_assert_ne!(self[target].inputs[inp_index], with, "{:?}", self[target]);

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
        self[id].lock_rc -= 1;
        self.remove(id)
    }

    fn iter(&self) -> impl DoubleEndedIterator<Item = (Nid, &Node)> {
        self.values.iter().enumerate().filter_map(|(i, s)| Some((i as _, s.as_ref().ok()?)))
    }

    #[expect(clippy::format_in_format_args)]
    fn basic_blocks_instr(&mut self, out: &mut String, node: Nid) -> core::fmt::Result {
        if self[node].kind != Kind::Loop && self[node].kind != Kind::Region {
            write!(out, "  {node:>2}-c{:>2}: ", self[node].ralloc_backref)?;
        }
        match self[node].kind {
            Kind::Start => unreachable!(),
            Kind::End => return Ok(()),
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
            Kind::Call { func, args: _ } => {
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
            Kind::Loops => write!(out, " loops:      "),
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
        self.visited.clear(self.values.len());
        self.basic_blocks_low(&mut out, VOID).unwrap();
        log::info!("{out}");
    }

    fn is_cfg(&self, o: Nid) -> bool {
        self[o].kind.is_cfg()
    }

    fn check_final_integrity(&self, tys: &Types, files: &[parser::Ast]) {
        if !cfg!(debug_assertions) {
            return;
        }

        let mut failed = false;
        for (id, node) in self.iter() {
            if node.lock_rc != 0 {
                log::error!("{} {} {:?}", node.lock_rc, 0, node.kind);
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
        }

        if failed {
            self.graphviz_in_browser(tys, files);
            panic!()
        }
    }

    fn load_loop_var(&mut self, index: usize, var: &mut Variable, loops: &mut [Loop]) {
        if var.value() != VOID {
            return;
        }

        let [loops @ .., loob] = loops else { unreachable!() };
        let node = loob.node;
        let lvar = &mut loob.scope.vars[index];

        self.load_loop_var(index, lvar, loops);

        if !self[lvar.value()].is_lazy_phi(node) {
            let lvalue = lvar.value();
            let inps = [node, lvalue, VOID];
            lvar.set_value(self.new_node_nop(lvar.ty, Kind::Phi, inps), self);
            self[lvar.value()].aclass = self[lvalue].aclass;
            self[lvar.value()].mem = self[lvalue].mem;
        }
        var.set_value(lvar.value(), self);
    }

    fn load_loop_aclass(&mut self, index: usize, var: &mut AClass, loops: &mut [Loop]) {
        if var.last_store.get() != VOID {
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
        var.last_store.set(lvar.last_store.get(), self);
    }

    fn check_dominance(&mut self, nd: Nid, min: Nid, check_outputs: bool) {
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

    fn dominates(&mut self, dominator: Nid, mut dominated: Nid) -> bool {
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
    // [ctrl]
    Stck,
    // [ctrl, memory]
    Load,
    // [ctrl, value, memory]
    Stre,
}

impl Kind {
    fn is_pinned(&self) -> bool {
        self.is_cfg() || matches!(self, Self::Phi | Self::Arg | Self::Mem | Self::Loops)
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
    offset: Offset,
    ralloc_backref: RallocBRef,
    depth: IDomDepth,
    lock_rc: LockRc,
    loop_depth: LoopDepth,
    aclass: AClassId,
    mem: Nid,
    antidep: Nid,
}

impl Node {
    fn is_dangling(&self) -> bool {
        self.outputs.len() + self.lock_rc as usize == 0 && self.kind != Kind::Arg
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
}

type RallocBRef = u16;
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
struct ItemCtx {
    file: FileId,
    ret: Option<ty::Id>,
    task_base: usize,
    inline_var_base: usize,
    inline_depth: u16,
    inline_ret: Option<(Value, StrongRef, Scope)>,
    nodes: Nodes,
    ctrl: StrongRef,
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

        self.call_count = 0;

        self.file = file;
        self.ret = ret;
        self.task_base = task_base;

        self.nodes.clear();
        self.scope.vars.clear();

        let start = self.nodes.new_node(ty::Id::VOID, Kind::Start, []);
        debug_assert_eq!(start, VOID);
        let end = self.nodes.new_node(ty::Id::NEVER, Kind::End, []);
        debug_assert_eq!(end, NEVER);
        self.nodes.lock(end);
        self.ctrl =
            StrongRef::new(self.nodes.new_node(ty::Id::VOID, Kind::Entry, [VOID]), &mut self.nodes);
        debug_assert_eq!(self.ctrl.get(), ENTRY);
        let mem = self.nodes.new_node(ty::Id::VOID, Kind::Mem, [VOID]);
        debug_assert_eq!(mem, MEM);
        self.nodes.lock(mem);
        let loops = self.nodes.new_node(ty::Id::VOID, Kind::Loops, [VOID]);
        debug_assert_eq!(loops, LOOPS);
        self.nodes.lock(loops);
        self.scope.aclasses.push(AClass::new(&mut self.nodes)); // DEFAULT
        self.scope.aclasses.push(AClass::new(&mut self.nodes)); // GLOBAL
    }

    fn finalize(&mut self, stack: &mut Vec<Nid>, _tys: &Types, _files: &[parser::Ast]) {
        self.scope.clear(&mut self.nodes);
        mem::take(&mut self.ctrl).soft_remove(&mut self.nodes);

        self.nodes.iter_peeps(1000, stack);

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
    pub fn with_ty(self, ty: ty::Id) -> Self {
        Self { ty: Some(ty) }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
    used_cis: usize,
    ralloc: Regalloc,
    nid_stack: Vec<Nid>,
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
            mem::swap(slot, target);
        } else {
            self.cis.push(ItemCtx::default());
            mem::swap(self.cis.last_mut().unwrap(), target);
        }
        target.init(file, ret, task_base);
        self.used_cis += 1;
    }

    pub fn pop_ci(&mut self, target: &mut ItemCtx) {
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
pub struct CodegenCtx {
    pub parser: parser::Ctx,
    tys: Types,
    pool: Pool,
    ct: Comptime,
}

impl CodegenCtx {
    pub fn clear(&mut self) {
        self.parser.clear();
        self.tys.clear();
        self.pool.clear();
        self.ct.clear();
    }
}

pub struct Errors<'a>(&'a RefCell<String>);

impl Deref for Errors<'_> {
    type Target = RefCell<String>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Drop for Errors<'_> {
    fn drop(&mut self) {
        if debug::panicking() && !self.0.borrow().is_empty() {
            log::error!("{}", self.0.borrow());
        }
    }
}

pub struct Codegen<'a> {
    pub files: &'a [parser::Ast],
    pub errors: Errors<'a>,
    tys: &'a mut Types,
    ci: ItemCtx,
    pool: &'a mut Pool,
    ct: &'a mut Comptime,
}

impl<'a> Codegen<'a> {
    pub fn new(files: &'a [parser::Ast], ctx: &'a mut CodegenCtx) -> Self {
        Self {
            files,
            errors: Errors(&ctx.parser.errors),
            tys: &mut ctx.tys,
            ci: Default::default(),
            pool: &mut ctx.pool,
            ct: &mut ctx.ct,
        }
    }

    fn emit_and_eval(&mut self, file: FileId, ret: ty::Id, ret_loc: &mut [u8]) -> u64 {
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

        self.ci.emit_ct_body(self.tys, self.files, Sig { args: Tuple::empty(), ret }, self.pool);

        let func = Func {
            file,
            relocs: mem::take(&mut self.ci.relocs),
            code: mem::take(&mut self.ci.code),
            ..Default::default()
        };

        // TODO: return them back
        let fuc = self.tys.ins.funcs.len() as ty::Func;
        self.tys.ins.funcs.push(func);

        self.tys.dump_reachable(fuc, &mut self.ct.code);
        self.dump_ct_asm();

        self.ct.run(ret_loc, self.tys.ins.funcs[fuc as usize].offset)
    }

    fn dump_ct_asm(&self) {
        #[cfg(debug_assertions)]
        {
            let mut vc = String::new();
            if let Err(e) = self.tys.disasm(&self.ct.code, self.files, &mut vc, |_| {}) {
                panic!("{e} {}", vc);
            } else {
                log::trace!("{}", vc);
            }
        }
    }

    pub fn push_embeds(&mut self, embeds: Vec<Vec<u8>>) {
        self.tys.ins.globals = embeds
            .into_iter()
            .map(|data| Global {
                ty: self.tys.make_array(ty::Id::U8, data.len() as _),
                data,
                ..Default::default()
            })
            .collect();
    }

    fn new_stack(&mut self, ty: ty::Id) -> Nid {
        let stck = self.ci.nodes.new_node_nop(ty, Kind::Stck, [VOID, MEM]);
        self.ci.nodes[stck].aclass = self.ci.scope.aclasses.len() as _;
        self.ci.nodes[stck].mem = stck;
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
            // simply switch the class to the default one
            let aclass = &mut self.ci.scope.aclasses[value_index];
            self.ci.nodes.load_loop_aclass(value_index, aclass, &mut self.ci.loops);
            let last_store = aclass.last_store.get();
            let mut cursor = last_store;
            let mut first_store = cursor;
            while cursor != MEM {
                first_store = cursor;
                cursor = self.ci.nodes[cursor].inputs[3];
            }

            if last_store != MEM {
                let base_class = self.ci.scope.aclasses[0].last_store.get();
                if base_class != MEM {
                    self.ci.nodes.modify_input(first_store, 3, base_class);
                }
                self.ci.scope.aclasses[0].last_store.set(last_store, &mut self.ci.nodes);
            }
            self.ci.nodes[value_region].aclass = 0;
        }

        let (index, _) = self.ci.nodes.aclass_index(region);
        let aclass = &mut self.ci.scope.aclasses[index];
        self.ci.nodes.load_loop_aclass(index, aclass, &mut self.ci.loops);
        let vc = Vc::from([aclass.clobber.get(), value, region, aclass.last_store.get()]);
        mem::take(&mut aclass.last_store).soft_remove(&mut self.ci.nodes);
        let store = self.ci.nodes.new_node(ty, Kind::Stre, vc);
        aclass.last_store = StrongRef::new(store, &mut self.ci.nodes);
        store
    }

    fn load_mem(&mut self, region: Nid, ty: ty::Id) -> Nid {
        debug_assert_ne!(region, VOID);
        debug_assert_ne!({ self.ci.nodes[region].ty }, ty::Id::VOID, "{:?}", {
            self.ci.nodes[region].lock_rc = Nid::MAX;
            self.ci.nodes.graphviz_in_browser(self.tys, self.files);
        });
        debug_assert!(
            self.ci.nodes[region].kind != Kind::Load || self.ci.nodes[region].ty.is_pointer(),
            "{:?} {} {}",
            self.ci.nodes.graphviz_in_browser(self.tys, self.files),
            self.file().path,
            self.ty_display(self.ci.nodes[region].ty)
        );
        debug_assert!(self.ci.nodes[region].kind != Kind::Stre);
        let (index, _) = self.ci.nodes.aclass_index(region);
        let aclass = &mut self.ci.scope.aclasses[index];
        self.ci.nodes.load_loop_aclass(index, aclass, &mut self.ci.loops);
        let vc = [aclass.clobber.get(), region, aclass.last_store.get()];
        self.ci.nodes.new_node(ty, Kind::Load, vc)
    }

    pub fn generate(&mut self, entry: FileId) {
        self.find_type(0, entry, entry, Err("main"), self.files);
        if self.tys.ins.funcs.is_empty() {
            return;
        }
        self.make_func_reachable(0);
        self.complete_call_graph();
    }

    pub fn assemble_comptime(&mut self) -> Comptime {
        self.ct.code.clear();
        self.tys.reassemble(&mut self.ct.code);
        self.ct.reset();
        core::mem::take(self.ct)
    }

    pub fn assemble(&mut self, buf: &mut Vec<u8>) {
        self.tys.reassemble(buf);
    }

    pub fn disasm(&mut self, output: &mut String) -> Result<(), DisasmError> {
        let mut bin = Vec::new();
        self.assemble(&mut bin);
        self.tys.disasm(&bin, self.files, output, |_| {})
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let fuc = &mut self.tys.ins.funcs[func as usize];
        if fuc.offset == u32::MAX {
            fuc.offset = task::id(self.tys.tasks.len() as _);
            self.tys.tasks.push(Some(FTask { file: fuc.file, id: func }));
        }
    }

    fn raw_expr(&mut self, expr: &Expr) -> Option<Value> {
        self.raw_expr_ctx(expr, Ctx::default())
    }

    fn raw_expr_ctx(&mut self, expr: &Expr, mut ctx: Ctx) -> Option<Value> {
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
                        let stack = self.new_stack(oty);
                        let offset = self.offset(stack, flag_offset);
                        let value = self.ci.nodes.new_const(flag_ty, 0);
                        self.store_mem(offset, flag_ty, value);
                        Some(Value::ptr(stack).ty(oty))
                    }
                }
            }
            Expr::Idk { pos } => {
                inference!(ty, ctx, self, pos, "value", "@as(<ty>, idk)");

                if matches!(ty.expand(), ty::Kind::Struct(_) | ty::Kind::Slice(_)) {
                    Some(Value::ptr(self.new_stack(ty)).ty(ty))
                } else {
                    self.report(
                        pos,
                        fa!(
                            "type '{}' cannot be uninitialized, use a zero \
                            value instead ('null' in case of pointers)",
                            self.ty_display(ty)
                        ),
                    );
                    Value::NEVER
                }
            }
            Expr::Bool { value, .. } => Some(self.ci.nodes.new_const_lit(ty::Id::BOOL, value)),
            Expr::Number { value, .. } => Some(
                self.ci.nodes.new_const_lit(
                    ctx.ty
                        .map(|ty| self.tys.inner_of(ty).unwrap_or(ty))
                        .filter(|ty| ty.is_integer())
                        .unwrap_or(ty::Id::DEFAULT_INT),
                    value,
                ),
            ),
            Expr::Float { value, .. } => Some(
                self.ci.nodes.new_const_lit(
                    ctx.ty
                        .map(|ty| self.tys.inner_of(ty).unwrap_or(ty))
                        .filter(|ty| ty.is_float())
                        .unwrap_or(ty::Id::F32),
                    value as i64,
                ),
            ),
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
                    ty::Kind::Builtin(ty::NEVER) => Value::NEVER,
                    ty::Kind::Global(global) => {
                        let gl = &self.tys.ins.globals[global as usize];
                        let value = self.ci.nodes.new_node(gl.ty, Kind::Global { global }, [VOID]);
                        self.ci.nodes[value].aclass = GLOBAL_ACLASS as _;
                        Some(Value::ptr(value).ty(gl.ty))
                    }
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
                        let global = self.tys.ins.globals.len() as ty::Global;
                        self.tys.ins.globals.push(Global { data, ty, ..Default::default() });
                        vacant_entry
                            .insert(crate::ctx_map::Key { value: StringRef(global), hash }, ())
                            .0
                            .value
                            .0
                    }
                };
                let global = self.ci.nodes.new_node(ty, Kind::Global { global }, [VOID]);
                self.ci.nodes[global].aclass = GLOBAL_ACLASS as _;
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

                if self.ci.inline_depth == 0 {
                    debug_assert_ne!(self.ci.ctrl.get(), VOID);
                    let mut inps = Vc::from([self.ci.ctrl.get(), value.id]);
                    for (i, aclass) in self.ci.scope.aclasses.iter_mut().enumerate() {
                        self.ci.nodes.load_loop_aclass(i, aclass, &mut self.ci.loops);
                        inps.push(aclass.last_store.get());
                    }

                    self.ci.ctrl.set(
                        self.ci.nodes.new_node_nop(ty::Id::VOID, Kind::Return, inps),
                        &mut self.ci.nodes,
                    );

                    self.ci.nodes[NEVER].inputs.push(self.ci.ctrl.get());
                    self.ci.nodes[self.ci.ctrl.get()].outputs.push(NEVER);
                } else if let Some((pv, ctrl, scope)) = &mut self.ci.inline_ret {
                    ctrl.set(
                        self.ci
                            .nodes
                            .new_node(ty::Id::VOID, Kind::Region, [self.ci.ctrl.get(), ctrl.get()]),
                        &mut self.ci.nodes,
                    );
                    self.ci.nodes.merge_scopes(&mut self.ci.loops, ctrl, scope, &mut self.ci.scope);
                    self.ci.nodes.unlock(pv.id);
                    pv.id =
                        self.ci.nodes.new_node(value.ty, Kind::Phi, [ctrl.get(), value.id, pv.id]);
                    self.ci.nodes.lock(pv.id);
                    self.ci.ctrl.set(NEVER, &mut self.ci.nodes);
                } else {
                    self.ci.nodes.lock(value.id);
                    let mut scope = self.ci.scope.dup(&mut self.ci.nodes);
                    scope
                        .vars
                        .drain(self.ci.inline_var_base..)
                        .for_each(|v| v.remove(&mut self.ci.nodes));
                    let repl = StrongRef::new(NEVER, &mut self.ci.nodes);
                    self.ci.inline_ret =
                        Some((value, mem::replace(&mut self.ci.ctrl, repl), scope));
                }

                None
            }
            Expr::Field { target, name, pos } => {
                let mut vtarget = self.raw_expr(target)?;
                self.strip_var(&mut vtarget);
                let tty = vtarget.ty;

                if let ty::Kind::Module(m) = tty.expand() {
                    return match self
                        .find_type(pos, self.ci.file, m, Err(name), self.files)
                        .expand()
                    {
                        ty::Kind::Builtin(ty::NEVER) => Value::NEVER,
                        ty::Kind::Global(global) => {
                            let gl = &self.tys.ins.globals[global as usize];
                            let value =
                                self.ci.nodes.new_node(gl.ty, Kind::Global { global }, [VOID]);
                            self.ci.nodes[value].aclass = GLOBAL_ACLASS as _;
                            Some(Value::ptr(value).ty(gl.ty))
                        }
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
            Expr::UnOp { op: TokenKind::Band, val, .. } => {
                let ctx = Ctx { ty: ctx.ty.and_then(|ty| self.tys.base_of(ty)) };

                let mut val = self.raw_expr_ctx(val, ctx)?;
                self.strip_var(&mut val);

                if val.ptr {
                    val.ptr = false;
                    val.ty = self.tys.make_ptr(val.ty);
                    return Some(val);
                }

                let stack = self.new_stack(val.ty);
                self.store_mem(stack, val.ty, val.id);

                Some(Value::new(stack).ty(self.tys.make_ptr(val.ty)))
            }
            Expr::UnOp { op: TokenKind::Mul, val, pos } => {
                let ctx = Ctx { ty: ctx.ty.map(|ty| self.tys.make_ptr(ty)) };
                let mut val = self.expr_ctx(val, ctx)?;

                self.unwrap_opt(pos, &mut val);

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
                let val =
                    self.expr_ctx(val, Ctx::default().with_ty(ctx.ty.unwrap_or(ty::Id::INT)))?;
                if val.ty.is_integer() {
                    Some(self.ci.nodes.new_node_lit(val.ty, Kind::UnOp { op }, [VOID, val.id]))
                } else if val.ty.is_float() {
                    let value = self.ci.nodes.new_const(val.ty, (-1f64).to_bits() as i64);
                    Some(self.ci.nodes.new_node_lit(val.ty, Kind::BinOp { op: TokenKind::Mul }, [
                        VOID, val.id, value,
                    ]))
                } else {
                    self.report(pos, fa!("cant negate '{}'", self.ty_display(val.ty)));
                    Value::NEVER
                }
            }
            Expr::BinOp { left, op: TokenKind::Decl, right, .. } => {
                let mut right = self.expr(right)?;

                if right.ty.loc(self.tys) == Loc::Stack {
                    let stck = self.new_stack(right.ty);
                    self.store_mem(stck, right.ty, right.id);
                    right.id = stck;
                    right.ptr = true;
                }
                self.assign_pattern(left, right);
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
                    return Some(self.ci.nodes.new_const_lit(ty::Id::BOOL, 1));
                };

                Some(Value::new(self.gen_null_check(cmped, ty, op)).ty(ty::BOOL))
            }
            Expr::BinOp { left, pos, op, right }
                if !matches!(op, TokenKind::Assign | TokenKind::Decl) =>
            {
                let mut lhs = self.raw_expr_ctx(left, ctx)?;
                self.strip_var(&mut lhs);

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
                        let (ty, aclass, mem) = self.binop_ty(pos, &mut lhs, &mut rhs, op);
                        let inps = [VOID, lhs.id, rhs.id];
                        let bop =
                            self.ci.nodes.new_node_lit(ty.bin_ret(op), Kind::BinOp { op }, inps);
                        self.ci.nodes[bop.id].aclass = aclass as _;
                        self.ci.nodes[bop.id].mem = mem;
                        Some(bop)
                    }
                    ty::Kind::Struct(s) if op.is_homogenous() => {
                        self.ci.nodes.lock(lhs.id);
                        let rhs = self.raw_expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                        self.ci.nodes.unlock(lhs.id);
                        let mut rhs = rhs?;
                        self.strip_var(&mut rhs);
                        self.assert_ty(pos, &mut rhs, lhs.ty, "struct operand");
                        let dst = self.new_stack(lhs.ty);
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

                let elem = self.tys.ins.slices[s as usize].elem;
                let mut idx = self.expr_ctx(index, Ctx::default().with_ty(ty::Id::DEFAULT_INT))?;
                self.assert_ty(index.pos(), &mut idx, ty::Id::DEFAULT_INT, "subscript");
                let size = self.ci.nodes.new_const(ty::Id::INT, self.tys.size_of(elem));
                let inps = [VOID, idx.id, size];
                let offset =
                    self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Mul }, inps);
                let (aclass, mem) = self.ci.nodes.aclass_index(bs.id);
                let inps = [VOID, bs.id, offset];
                let ptr =
                    self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Add }, inps);
                self.ci.nodes[ptr].aclass = aclass as _;
                self.ci.nodes[ptr].mem = mem;

                Some(Value::ptr(ptr).ty(elem))
            }
            Expr::Embed { id, .. } => {
                let glob = &self.tys.ins.globals[id as usize];
                let g = self.ci.nodes.new_node(glob.ty, Kind::Global { global: id }, [VOID]);
                Some(Value::ptr(g).ty(glob.ty))
            }
            Expr::Directive { name: "sizeof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(
                    self.ci.nodes.new_const_lit(
                        ctx.ty
                            .map(|ty| self.tys.inner_of(ty).unwrap_or(ty))
                            .filter(|ty| ty.is_integer())
                            .unwrap_or(ty::Id::DEFAULT_INT),
                        self.tys.size_of(ty),
                    ),
                )
            }
            Expr::Directive { name: "alignof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(
                    self.ci.nodes.new_const_lit(
                        ctx.ty
                            .map(|ty| self.tys.inner_of(ty).unwrap_or(ty))
                            .filter(|ty| ty.is_integer())
                            .unwrap_or(ty::Id::DEFAULT_INT),
                        self.tys.align_of(ty),
                    ),
                )
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
                        let stack = self.new_stack(ty);
                        self.store_mem(stack, val.ty, val.id);
                        val.id = stack;
                        val.ptr = true;
                    }
                    _ => {}
                }

                val.ty = ty;
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

                if self.tys.size_of(val.ty) < self.tys.size_of(ty) {
                    Some(
                        self.ci
                            .nodes
                            .new_node_lit(ty, Kind::UnOp { op: TokenKind::Float }, [VOID, val.id]),
                    )
                } else {
                    Some(val.ty(ty))
                }
            }
            Expr::Directive { name: "fti", args: [expr], .. } => {
                let val = self.expr(expr)?;

                let ret_ty = match val.ty {
                    ty::Id::F64 => ty::Id::INT,
                    ty::Id::F32 => ty::Id::I32,
                    _ => {
                        self.report(
                            expr.pos(),
                            fa!("expected float ('{}' is not a float)", self.ty_display(val.ty)),
                        );
                        return Value::NEVER;
                    }
                };

                Some(
                    self.ci
                        .nodes
                        .new_node_lit(ret_ty, Kind::UnOp { op: TokenKind::Number }, [VOID, val.id]),
                )
            }
            Expr::Directive { name: "itf", args: [expr], .. } => {
                let mut val = self.expr(expr)?;

                let (ret_ty, expected) = match val.ty.simple_size().unwrap() {
                    8 => (ty::Id::F64, ty::Id::INT),
                    _ => (ty::Id::F32, ty::Id::I32),
                };

                self.assert_ty(expr.pos(), &mut val, expected, "converted integer");

                Some(
                    self.ci
                        .nodes
                        .new_node_lit(ret_ty, Kind::UnOp { op: TokenKind::Float }, [VOID, val.id]),
                )
            }
            Expr::Directive { name: "as", args: [ty, expr], .. } => {
                let ty = self.ty(ty);
                let ctx = Ctx::default().with_ty(ty);
                let mut val = self.raw_expr_ctx(expr, ctx)?;
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
                        let stck = self.new_stack(ty);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(ty))
                    }
                };

                inps[0] = self.ci.ctrl.get();
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty, Kind::Call { func: ty::ECA, args }, inps),
                    &mut self.ci.nodes,
                );

                self.add_clobber_stores(clobbered_aliases);

                alt_value.or(Some(Value::new(self.ci.ctrl.get()).ty(ty)))
            }
            Expr::Call { func, args, .. } => {
                self.ci.call_count += 1;
                let ty = self.ty(func);
                let ty::Kind::Func(mut fu) = ty.expand() else {
                    self.report(
                        func.pos(),
                        fa!("compiler cant (yet) call '{}'", self.ty_display(ty)),
                    );
                    return Value::NEVER;
                };

                let Some(sig) = self.compute_signature(&mut fu, func.pos(), args) else {
                    return Value::NEVER;
                };
                self.make_func_reachable(fu);

                let fuc = &self.tys.ins.funcs[fu as usize];
                let ast = &self.files[fuc.file as usize];
                let &Expr::Closure { args: cargs, .. } = fuc.expr.get(ast) else { unreachable!() };

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

                let mut inps = Vc::from([NEVER]);
                let mut tys = sig.args.args();
                let mut cargs = cargs.iter();
                let mut args = args.iter();
                let mut clobbered_aliases = BitSet::default();
                while let Some(ty) = tys.next(self.tys) {
                    let carg = cargs.next().unwrap();
                    let Some(arg) = args.next() else { break };
                    let Arg::Value(ty) = ty else { continue };

                    let mut value = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                    self.assert_ty(arg.pos(), &mut value, ty, fa!("argument {}", carg.name));
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
                        let stck = self.new_stack(sig.ret);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(sig.ret))
                    }
                };

                inps[0] = self.ci.ctrl.get();
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(sig.ret, Kind::Call { func: fu, args: sig.args }, inps),
                    &mut self.ci.nodes,
                );

                self.add_clobber_stores(clobbered_aliases);

                alt_value.or(Some(Value::new(self.ci.ctrl.get()).ty(sig.ret)))
            }
            Expr::Directive { name: "inline", args: [func, args @ ..], .. } => {
                let ty = self.ty(func);
                let ty::Kind::Func(mut fu) = ty.expand() else {
                    self.report(
                        func.pos(),
                        fa!(
                            "first argument to @inline should be a function,
                                        but here its '{}'",
                            self.ty_display(ty)
                        ),
                    );
                    return Value::NEVER;
                };

                let Some(sig) = self.compute_signature(&mut fu, func.pos(), args) else {
                    return Value::NEVER;
                };

                let Func { expr, file, .. } = self.tys.ins.funcs[fu as usize];

                let ast = &self.files[file as usize];
                let &Expr::Closure { args: cargs, body, .. } = expr.get(ast) else {
                    unreachable!()
                };

                if args.len() != cargs.len() {
                    self.report(
                        func.pos(),
                        fa!(
                            "expected {} inline function argumenr{}, got {}",
                            cargs.len(),
                            if cargs.len() == 1 { "" } else { "s" },
                            args.len()
                        ),
                    );
                }

                let mut tys = sig.args.args();
                let mut args = args.iter();
                let mut cargs = cargs.iter();
                let var_base = self.ci.scope.vars.len();
                while let Some(aty) = tys.next(self.tys) {
                    let carg = cargs.next().unwrap();
                    let Some(arg) = args.next() else { break };
                    match aty {
                        Arg::Type(id) => {
                            self.ci.scope.vars.push(Variable::new(
                                carg.id,
                                id,
                                false,
                                NEVER,
                                &mut self.ci.nodes,
                            ));
                        }
                        Arg::Value(ty) => {
                            let mut value = self.raw_expr_ctx(arg, Ctx::default().with_ty(ty))?;
                            self.strip_var(&mut value);
                            debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                            debug_assert_ne!(value.id, 0);
                            self.assert_ty(
                                arg.pos(),
                                &mut value,
                                ty,
                                fa!("argument {}", carg.name),
                            );

                            self.ci.scope.vars.push(Variable::new(
                                carg.id,
                                ty,
                                value.ptr,
                                value.id,
                                &mut self.ci.nodes,
                            ));
                        }
                    }
                }

                let prev_var_base =
                    mem::replace(&mut self.ci.inline_var_base, self.ci.scope.vars.len());
                let prev_ret = self.ci.ret.replace(sig.ret);
                let prev_inline_ret = self.ci.inline_ret.take();
                let prev_file = mem::replace(&mut self.ci.file, file);
                self.ci.inline_depth += 1;

                if self.expr(body).is_some() && sig.ret != ty::Id::VOID {
                    self.report(
                        body.pos(),
                        "expected all paths in the fucntion to return \
                                    or the return type to be 'void'",
                    );
                }

                self.ci.ret = prev_ret;
                self.ci.file = prev_file;
                self.ci.inline_depth -= 1;
                self.ci.inline_var_base = prev_var_base;
                for var in self.ci.scope.vars.drain(var_base..) {
                    var.remove(&mut self.ci.nodes);
                }

                mem::replace(&mut self.ci.inline_ret, prev_inline_ret).map(|(v, ctrl, scope)| {
                    self.ci.nodes.unlock(v.id);
                    self.ci.scope.clear(&mut self.ci.nodes);
                    self.ci.scope = scope;
                    self.ci.scope.vars.drain(var_base..).for_each(|v| v.remove(&mut self.ci.nodes));
                    mem::replace(&mut self.ci.ctrl, ctrl).remove(&mut self.ci.nodes);
                    v
                })
            }
            Expr::Tupl { pos, ty, fields, .. } => {
                ctx.ty = ty.map(|ty| self.ty(ty)).or(ctx.ty);
                inference!(sty, ctx, self, pos, "struct or slice", "<struct_ty>.(...)");

                match sty.expand() {
                    ty::Kind::Struct(s) => {
                        let mem = self.new_stack(sty);
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

                        let mem = self.new_stack(aty);

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
                ctx.ty = ty.map(|ty| self.ty(ty)).or(ctx.ty);
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
                let mem = self.new_stack(sty);
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
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Loop, [
                        self.ci.ctrl.get(),
                        self.ci.ctrl.get(),
                        LOOPS,
                    ]),
                    &mut self.ci.nodes,
                );
                self.ci.loops.push(Loop {
                    node: self.ci.ctrl.get(),
                    ctrl: [StrongRef::DEFAULT; 2],
                    ctrl_scope: core::array::from_fn(|_| Default::default()),
                    scope: self.ci.scope.dup(&mut self.ci.nodes),
                });

                for var in self.ci.scope.vars.iter_mut() {
                    var.set_value(VOID, &mut self.ci.nodes);
                }

                for aclass in self.ci.scope.aclasses.iter_mut() {
                    aclass.last_store.set(VOID, &mut self.ci.nodes);
                }

                self.expr(body);

                let Loop { ctrl: [con, ..], ctrl_scope: [cons, ..], .. } =
                    self.ci.loops.last_mut().unwrap();
                let mut cons = mem::take(cons);

                if let Some(con) = mem::take(con).unwrap(&mut self.ci.nodes) {
                    self.ci.ctrl.set(
                        self.ci
                            .nodes
                            .new_node(ty::Id::VOID, Kind::Region, [con, self.ci.ctrl.get()]),
                        &mut self.ci.nodes,
                    );
                    self.ci.nodes.merge_scopes(
                        &mut self.ci.loops,
                        &self.ci.ctrl,
                        &mut self.ci.scope,
                        &mut cons,
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
                    }

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
                        if loop_var.value() != scope_var.value() {
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

                    debug_assert!(!self.ci.nodes[dest_class.last_store.get()].is_lazy_phi(node));
                }

                scope.clear(&mut self.ci.nodes);
                bres.clear(&mut self.ci.nodes);

                self.ci.nodes.unlock(node);
                let rpl = self.ci.nodes.late_peephole(node).unwrap_or(node);
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

                let if_node =
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::If, [self.ci.ctrl.get(), cnd.id]);

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
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Then, [if_node]),
                    &mut self.ci.nodes,
                );
                let lcntrl = self.expr(then).map_or(Nid::MAX, |_| self.ci.ctrl.get());

                let mut then_scope = mem::replace(&mut self.ci.scope, else_scope);
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Else, [if_node]),
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
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [lcntrl, rcntrl]),
                    &mut self.ci.nodes,
                );

                self.ci.nodes.merge_scopes(
                    &mut self.ci.loops,
                    &self.ci.ctrl,
                    &mut self.ci.scope,
                    &mut then_scope,
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
                    let res = self.ci.nodes.new_node(ty, Kind::BinOp { op }, [VOID, lhs, rhs]);
                    self.store_mem(dst, ty, res);
                }
                ty::Kind::Struct(is) => {
                    if !self.struct_op(pos, op, is, dst, lhs, rhs) {
                        self.report(
                            pos,
                            fa!(
                                "... when appliing '{0} {op} {0}'",
                                self.ty_display(ty::Kind::Struct(s).compress())
                            ),
                        );
                    }
                }
                _ => self.report(pos, fa!("'{0} {op} {0}' is not supported", self.ty_display(ty))),
            }
        }

        true
    }

    fn compute_signature(&mut self, func: &mut ty::Func, pos: Pos, args: &[Expr]) -> Option<Sig> {
        let fuc = &self.tys.ins.funcs[*func as usize];
        let fast = self.files[fuc.file as usize].clone();
        let &Expr::Closure { args: cargs, ret, .. } = fuc.expr.get(&fast) else {
            unreachable!();
        };

        Some(if let Some(sig) = fuc.sig {
            sig
        } else {
            let arg_base = self.tys.tmp.args.len();

            let base = self.ci.scope.vars.len();
            for (arg, carg) in args.iter().zip(cargs) {
                let ty = self.ty(&carg.ty);
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
            let ret = self.ty(ret);

            self.ci.scope.vars.drain(base..).for_each(|v| v.remove(&mut self.ci.nodes));

            let sym = SymKey::FuncInst(*func, args);
            let ct = |ins: &mut crate::TypeIns| {
                let func_id = ins.funcs.len();
                let fuc = &ins.funcs[*func as usize];
                ins.funcs.push(Func {
                    file: fuc.file,
                    name: fuc.name,
                    base: Some(*func),
                    sig: Some(Sig { args, ret }),
                    expr: fuc.expr,
                    ..Default::default()
                });

                ty::Kind::Func(func_id as _).compress()
            };
            *func = self.tys.syms.get_or_insert(sym, &mut self.tys.ins, ct).expand().inner();

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
        let (aclass, mem) = self.ci.nodes.aclass_index(val);
        let inps = [VOID, val, off];
        let seted = self.ci.nodes.new_node(ty::Id::INT, Kind::BinOp { op: TokenKind::Add }, inps);
        self.ci.nodes[seted].aclass = aclass as _;
        self.ci.nodes[seted].mem = mem;
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
                self.ci.nodes.new_node(ty::Id::VOID, Kind::Region, [
                    self.ci.ctrl.get(),
                    loob.ctrl[id].get(),
                ]),
                &mut self.ci.nodes,
            );
            let mut scope = mem::take(&mut loob.ctrl_scope[id]);
            let ctrl = mem::take(&mut loob.ctrl[id]);

            self.ci.nodes.merge_scopes(&mut self.ci.loops, &ctrl, &mut scope, &mut self.ci.scope);

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

    fn emit_func(&mut self, FTask { file, id }: FTask) {
        let func = &mut self.tys.ins.funcs[id as usize];
        debug_assert_eq!(func.file, file);
        func.offset = u32::MAX - 1;
        let sig = func.sig.expect("to emmit only concrete functions");
        let ast = &self.files[file as usize];
        let expr = func.expr.get(ast);

        self.pool.push_ci(file, Some(sig.ret), 0, &mut self.ci);
        let prev_err_len = self.errors.borrow().len();

        let &Expr::Closure { body, args, .. } = expr else {
            unreachable!("{}", self.ast_display(expr))
        };

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

        self.ci.finalize(&mut self.pool.nid_stack, self.tys, self.files);

        if self.errors.borrow().len() == prev_err_len {
            self.ci.emit_body(self.tys, self.files, sig, self.pool);
            self.tys.ins.funcs[id as usize].code.append(&mut self.ci.code);
            self.tys.ins.funcs[id as usize].relocs.append(&mut self.ci.relocs);
        }

        self.pool.pop_ci(&mut self.ci);
    }

    fn ty(&mut self, expr: &Expr) -> ty::Id {
        self.parse_ty(self.ci.file, expr, None, self.files)
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
    ) -> (ty::Id, usize, Nid) {
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
                    oper.id =
                        self.ci.nodes.new_node(upcasted, Kind::BinOp { op: TokenKind::Mul }, [
                            VOID, oper.id, cnst,
                        ]);
                    return (
                        upcasted,
                        self.ci.nodes[other.id].aclass as _,
                        self.ci.nodes[other.id].mem,
                    );
                }
            }

            (upcasted, DEFAULT_ACLASS, VOID)
        } else {
            let ty = self.ty_display(lhs.ty);
            let expected = self.ty_display(rhs.ty);
            self.report(pos, fa!("'{ty} {op} {expected}' is not supported"));
            (ty::Id::NEVER, DEFAULT_ACLASS, VOID)
        }
    }

    fn wrap_in_opt(&mut self, val: &mut Value) {
        debug_assert!(!val.var);

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
                let flag_offset = self.tys.size_of(oty) - flag_offset - 1;
                let fill = self.ci.nodes.new_const(oty, 1i64 << (flag_offset * 8 - 1));
                val.id = self
                    .ci
                    .nodes
                    .new_node(oty, Kind::BinOp { op: TokenKind::Bor }, [VOID, val.id, fill]);
                val.ty = oty;
            }
            Loc::Stack if val.ty.loc(self.tys) == Loc::Reg => {
                self.strip_ptr(val);
                let stack = self.new_stack(oty);
                let fill = self.ci.nodes.new_const(flag_ty, 1);
                self.store_mem(stack, flag_ty, fill);
                let off = self.offset(stack, payload_offset);
                self.store_mem(off, val.ty, val.id);
                val.id = stack;
                val.ptr = true;
                val.ty = oty;
            }
            _ => todo!(),
        }
    }

    fn unwrap_opt(&mut self, pos: Pos, opt: &mut Value) {
        let Some(ty) = self.tys.inner_of(opt.ty) else { return };
        let null_check = self.gen_null_check(*opt, ty, TokenKind::Eq);

        // TODO: extract the if check int a fucntion
        let ctrl = self.ci.nodes.new_node(ty::Id::VOID, Kind::If, [self.ci.ctrl.get(), null_check]);
        let ctrl_ty = self.ci.nodes[ctrl].ty;
        self.ci.nodes.remove(ctrl);
        let oty = mem::replace(&mut opt.ty, ty);
        match ctrl_ty {
            ty::Id::LEFT_UNREACHABLE => {
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
            ty::Id::RIGHT_UNREACHABLE => {
                self.report(pos, "the value is always null, some checks might need to be inverted");
            }
            _ => {
                self.report(
                    pos,
                    "can't prove the value is not 'null', \
                    there is not nice syntax for bypassing this, sorry",
                );
            }
        }
    }

    fn gen_null_check(&mut self, mut cmped: Value, ty: ty::Id, op: TokenKind) -> Nid {
        let OptLayout { flag_ty, flag_offset, .. } = self.tys.opt_layout(ty);

        match cmped.ty.loc(self.tys) {
            Loc::Reg => {
                self.strip_ptr(&mut cmped);
                let inps = [VOID, cmped.id, self.ci.nodes.new_const(cmped.ty, 0)];
                self.ci.nodes.new_node(ty::Id::BOOL, Kind::BinOp { op }, inps)
            }
            Loc::Stack => {
                cmped.id = self.offset(cmped.id, flag_offset);
                cmped.ty = flag_ty;
                self.strip_ptr(&mut cmped);
                let inps = [VOID, cmped.id, self.ci.nodes.new_const(ty, 0)];
                self.ci.nodes.new_node(ty::Id::BOOL, Kind::BinOp { op }, inps)
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
                    self.wrap_in_opt(src);
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
                self.unwrap_opt(pos, src);
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
        let mask = self.ci.nodes.new_const(to, (1i64 << (self.tys.size_of(value.ty) * 8)) - 1);
        let inps = [VOID, value.id, mask];
        *value = self.ci.nodes.new_node_lit(to, Kind::BinOp { op: TokenKind::Band }, inps);
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
        &self.files[self.ci.file as usize]
    }
}

impl TypeParser for Codegen<'_> {
    fn tys(&mut self) -> &mut Types {
        self.tys
    }

    fn eval_const(&mut self, file: FileId, expr: &Expr, ret: ty::Id) -> u64 {
        let mut scope = mem::take(&mut self.ci.scope.vars);
        self.pool.push_ci(file, Some(ret), self.tys.tasks.len(), &mut self.ci);
        self.ci.scope.vars = scope;

        let prev_err_len = self.errors.borrow().len();

        self.expr(&Expr::Return { pos: expr.pos(), val: Some(expr) });

        scope = mem::take(&mut self.ci.scope.vars);
        self.ci.finalize(&mut self.pool.nid_stack, self.tys, self.files);

        let res = if self.errors.borrow().len() == prev_err_len {
            self.emit_and_eval(file, ret, &mut [])
        } else {
            1
        };

        self.pool.pop_ci(&mut self.ci);
        self.ci.scope.vars = scope;

        res
    }

    fn infer_type(&mut self, expr: &Expr) -> ty::Id {
        self.pool.save_ci(&self.ci);
        let ty = self.expr(expr).map_or(ty::Id::NEVER, |v| v.ty);

        self.pool.restore_ci(&mut self.ci);
        ty
    }

    fn on_reuse(&mut self, existing: ty::Id) {
        if let ty::Kind::Func(id) = existing.expand()
            && let func = &mut self.tys.ins.funcs[id as usize]
            && let Err(idx) = task::unpack(func.offset)
            && idx < self.tys.tasks.len()
        {
            func.offset = task::id(self.tys.tasks.len());
            let task = self.tys.tasks[idx].take();
            self.tys.tasks.push(task);
        }
    }

    fn eval_global(&mut self, file: FileId, name: Ident, expr: &Expr) -> ty::Id {
        let gid = self.tys.ins.globals.len() as ty::Global;
        self.tys.ins.globals.push(Global { file, name, ..Default::default() });

        let ty = ty::Kind::Global(gid);
        self.pool.push_ci(file, None, self.tys.tasks.len(), &mut self.ci);
        let prev_err_len = self.errors.borrow().len();

        self.expr(&(Expr::Return { pos: expr.pos(), val: Some(expr) }));

        self.ci.finalize(&mut self.pool.nid_stack, self.tys, self.files);

        let ret = self.ci.ret.expect("for return type to be infered");
        if self.errors.borrow().len() == prev_err_len {
            let mut mem = vec![0u8; self.tys.size_of(ret) as usize];
            self.emit_and_eval(file, ret, &mut mem);
            self.tys.ins.globals[gid as usize].data = mem;
        }

        self.pool.pop_ci(&mut self.ci);
        self.tys.ins.globals[gid as usize].ty = ret;

        ty.compress()
    }

    fn report(&self, file: FileId, pos: Pos, msg: impl Display) -> ty::Id {
        let mut buf = self.errors.borrow_mut();
        write!(buf, "{}", self.files[file as usize].report(pos, msg)).unwrap();
        ty::Id::NEVER
    }

    fn find_local_ty(&mut self, ident: Ident) -> Option<ty::Id> {
        self.ci.scope.vars.iter().rfind(|v| (v.id == ident && v.value() == NEVER)).map(|v| v.ty)
    }
}

// FIXME: make this more efficient (allocated with arena)

#[cfg(test)]
mod tests {
    use {
        super::CodegenCtx,
        alloc::{string::String, vec::Vec},
        core::fmt::Write,
    };

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        _ = log::set_logger(&crate::fs::Logger);
        log::set_max_level(log::LevelFilter::Info);
        //log::set_max_level(log::LevelFilter::Trace);

        let mut ctx = CodegenCtx::default();
        let (ref files, embeds) = crate::test_parse_files(ident, input, &mut ctx.parser);
        let mut codegen = super::Codegen::new(files, &mut ctx);
        codegen.push_embeds(embeds);

        codegen.generate(0);

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

        super::hbvm::test_run_vm(&out, output);
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
        directives;
        c_strings;
        struct_patterns;
        arrays;
        inline;
        idk;
        generic_functions;

        // Incomplete Examples;
        //comptime_pointers;
        generic_types;
        fb_driver;

        // Purely Testing Examples;
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
    }
}
