use {
    self::strong_ref::StrongRef,
    crate::{
        ctx_map::CtxEntry,
        debug,
        ident::Ident,
        instrs,
        lexer::{self, TokenKind},
        parser::{
            self,
            idfl::{self},
            CtorField, Expr, FileId, Pos,
        },
        reg, task,
        ty::{self, Arg, ArrayLen, Loc, Tuple},
        vc::{BitSet, Vc},
        Comptime, FTask, Func, Global, HashMap, Offset, OffsetIter, PLoc, Reloc, Sig, StringRef,
        SymKey, TypeParser, TypedReloc, Types,
    },
    alloc::{borrow::ToOwned, string::String, vec::Vec},
    core::{
        assert_matches::debug_assert_matches,
        cell::RefCell,
        fmt::{self, Debug, Display, Write},
        format_args as fa, mem,
        ops::{self, Deref},
    },
    hashbrown::hash_map,
    hbbytecode::DisasmError,
    regalloc2::VReg,
};

const VOID: Nid = 0;
const NEVER: Nid = 1;
const ENTRY: Nid = 2;
const MEM: Nid = 3;

type Nid = u16;

type Lookup = crate::ctx_map::CtxMap<Nid>;

impl crate::ctx_map::CtxEntry for Nid {
    type Ctx = [Result<Node, (Nid, debug::Trace)>];
    type Key<'a> = (Kind, &'a [Nid], ty::Id);

    fn key<'a>(&self, ctx: &'a Self::Ctx) -> Self::Key<'a> {
        ctx[*self as usize].as_ref().unwrap().key()
    }
}

#[derive(Clone)]
struct Nodes {
    values: Vec<Result<Node, (Nid, debug::Trace)>>,
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
    fn merge_scopes(
        &mut self,
        loops: &mut [Loop],
        ctrl: &StrongRef,
        to: &mut Scope,
        from: &mut Scope,
    ) {
        for (i, (to_value, from_value)) in to.iter_mut().zip(from.iter_mut()).enumerate() {
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

        to.loads.drain(..).for_each(|l| _ = l.remove(self));
        from.loads.drain(..).for_each(|l| _ = l.remove(self));
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
                    " node{i}[label=\"{i} {} {}\" color={color}]",
                    node.kind,
                    ty::Display::new(tys, files, node.ty)
                )?;
            } else {
                writeln!(out, " node{i}[label=\"{i} {}\" color={color}]", node.kind,)?;
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

    #[allow(dead_code)]
    fn graphviz(&self, tys: &Types, files: &[parser::Ast]) {
        let out = &mut String::new();
        _ = self.graphviz_low(tys, files, out);
        log::info!("{out}");
    }

    fn graphviz_in_browser(&self, tys: &Types, files: &[parser::Ast]) {
        #[cfg(all(debug_assertions, feature = "std"))]
        {
            let out = &mut String::new();
            _ = self.graphviz_low(tys, files, out);
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

    fn gcm(&mut self) {
        self.visited.clear(self.values.len());
        push_up(self);
        // TODO: handle infinte loops
        self.visited.clear(self.values.len());
        push_down(self, VOID);
    }

    fn remove_low(&mut self, id: Nid) -> Node {
        if cfg!(debug_assertions) {
            let value =
                mem::replace(&mut self.values[id as usize], Err((self.free, debug::trace())))
                    .unwrap();
            self.free = id;
            value
        } else {
            mem::replace(&mut self.values[id as usize], Err((Nid::MAX, debug::trace()))).unwrap()
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

        debug_assert!(!matches!(self[target].kind, Kind::Call { .. }), "{:?}", self[target]);

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

    fn late_peephole(&mut self, target: Nid) -> Option<Nid> {
        if let Some(id) = self.peephole(target) {
            self.replace(target, id);
            return Some(id);
        }
        None
    }

    fn iter_peeps(&mut self, mut fuel: usize) {
        let mut in_stack = BitSet::default();
        in_stack.clear(self.values.len());
        let mut stack =
            self.iter().map(|(id, ..)| id).inspect(|&id| _ = in_stack.set(id)).collect::<Vec<_>>();

        while fuel != 0
            && let Some(node) = stack.pop()
        {
            fuel -= 1;
            in_stack.unset(node);
            let new = self.late_peephole(node);
            if let Some(new) = new {
                for &i in self[new].outputs.iter().chain(self[new].inputs.iter()) {
                    if in_stack.set(i) {
                        stack.push(i)
                    }
                }
            }
        }
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

                if op == T::Sub
                    && self[lhs].kind == (K::BinOp { op: T::Add })
                    && let K::CInt { value: a } = self[rhs].kind
                    && let K::CInt { value: b } = self[self[lhs].inputs[2]].kind
                {
                    let new_rhs = self.new_node_nop(ty, K::CInt { value: b - a }, [ctrl]);
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
                    && self[self[target].inputs[1]].kind != Kind::Load
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
            K::Extend => {
                if self[target].ty.simple_size() == self[self[target].inputs[1]].ty.simple_size() {
                    return Some(self[target].inputs[1]);
                }
            }
            K::Loop => {
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
            if !matches!(node.kind, Kind::End | Kind::Mem | Kind::Arg) && node.outputs.is_empty() {
                log::error!("outputs are empry {id} {:?}", node.kind);
                failed = true;
            }
        }

        if failed {
            self.graphviz_in_browser(tys, files);
            panic!()
        }
    }

    fn load_loop_var(&mut self, index: usize, value: &mut Variable, loops: &mut [Loop]) {
        self.load_loop_value(&mut |l| l.scope.iter_mut().nth(index).unwrap(), value, loops);
    }

    fn load_loop_store(&mut self, value: &mut Variable, loops: &mut [Loop]) {
        self.load_loop_value(&mut |l| &mut l.scope.store, value, loops);
    }

    fn load_loop_value(
        &mut self,
        get_lvalue: &mut impl FnMut(&mut Loop) -> &mut Variable,
        var: &mut Variable,
        loops: &mut [Loop],
    ) {
        if var.value() != VOID {
            return;
        }

        let [loops @ .., loob] = loops else { unreachable!() };
        let node = loob.node;
        let lvar = get_lvalue(loob);

        self.load_loop_value(get_lvalue, lvar, loops);

        if !self[lvar.value()].is_lazy_phi(node) {
            let inps = [node, lvar.value(), VOID];
            lvar.set_value(self.new_node_nop(lvar.ty, Kind::Phi, inps), self);
        }
        var.set_value(lvar.value(), self);
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

    #[allow(dead_code)]
    fn eliminate_stack_temporaries(&mut self) {
        'o: for stack in self[MEM].outputs.clone() {
            if self.values[stack as usize].is_err() || self[stack].kind != Kind::Stck {
                continue;
            }
            let mut full_read_into = None;
            let mut unidentifed = Vc::default();
            for &o in self[stack].outputs.iter() {
                match self[o].kind {
                    Kind::Load
                        if self[o].ty == self[stack].ty
                            && self[o].outputs.iter().all(|&n| self[n].kind == Kind::Stre)
                            && let mut full_stores = self[o].outputs.iter().filter(|&&n| {
                                self[n].kind == Kind::Stre && self[n].inputs[1] == o
                            })
                            && let Some(&n) = full_stores.next()
                            && full_stores.next().is_none() =>
                    {
                        if full_read_into.replace(n).is_some() {
                            continue 'o;
                        }
                    }
                    _ => unidentifed.push(o),
                }
            }

            let Some(dst) = full_read_into else { continue };

            let mut saved = Vc::default();
            let mut cursor = dst;
            cursor = *self[cursor].inputs.get(3).unwrap_or(&MEM);
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
                let Some(index) = unidentifed.iter().position(|&n| n == contact_point) else {
                    continue 'o;
                };
                unidentifed.remove(index);
                saved.push(contact_point);
                cursor = *self[cursor].inputs.get(3).unwrap_or(&MEM);

                if unidentifed.is_empty() {
                    break;
                }
            }

            if !unidentifed.is_empty() {
                continue;
            }

            // FIXME: when the loads and stores become parallel we will need to get saved
            // differently
            let region = self[dst].inputs[2];
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

                self.modify_input(oper, 2, region);
            }

            self.replace(dst, *self[dst].inputs.get(3).unwrap_or(&MEM));
            if self.values[stack as usize].is_ok() {
                self.lock(stack);
            }
            if self.values[dst as usize].is_ok() {
                self.lock(dst);
            }
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
        args: ty::Tuple,
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
        self.is_cfg() || matches!(self, Self::Phi | Self::Arg | Self::Mem)
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

    fn is_lazy_phi(&self, loob: Nid) -> bool {
        self.kind == Kind::Phi && self.inputs[2] == 0 && self.inputs[0] == loob
    }

    fn is_not_gvnd(&self) -> bool {
        (self.kind == Kind::Phi && self.inputs[2] == 0)
            || matches!(self.kind, Kind::Arg | Kind::Stck)
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
pub struct Scope {
    vars: Vec<Variable>,
    loads: Vec<StrongRef>,
    store: Variable,
}

impl Scope {
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Variable> {
        core::iter::once(&mut self.store).chain(self.vars.iter_mut())
    }

    fn dup(&self, nodes: &mut Nodes) -> Self {
        Self {
            vars: self.vars.iter().map(|v| v.dup(nodes)).collect(),
            loads: self.loads.iter().map(|l| l.dup(nodes)).collect(),
            store: self.store.dup(nodes),
        }
    }

    fn clear(&mut self, nodes: &mut Nodes) {
        self.vars.drain(..).for_each(|n| n.remove(nodes));
        self.loads.drain(..).for_each(|l| _ = l.remove(nodes));
        mem::take(&mut self.store).remove(nodes);
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
        self.scope.store = Variable::new(0, ty::Id::VOID, false, MEM, &mut self.nodes);
    }

    fn finalize(&mut self) {
        self.scope.clear(&mut self.nodes);
        self.nodes.unlock(NEVER);
        mem::take(&mut self.ctrl).soft_remove(&mut self.nodes);
        self.nodes.unlock(MEM);
        self.nodes.eliminate_stack_temporaries();
        self.nodes.iter_peeps(1000);
    }

    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        crate::emit(&mut self.code, instr);
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
        let mut args = fuc.nodes[VOID].outputs[2..].iter();
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
            self.emit(instrs::addi64(rg, reg::STACK_PTR, fuc.nodes[arg].offset as _));
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
                                op.cond_op(fuc.nodes[fuc.nodes[cnd].inputs[1]].ty.is_signed())
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
                    Kind::CInt { value } => self.emit(match tys.size_of(node.ty) {
                        1 => instrs::li8(atr(allocs[0]), value as _),
                        2 => instrs::li16(atr(allocs[0]), value as _),
                        4 => instrs::li32(atr(allocs[0]), value as _),
                        _ => instrs::li64(atr(allocs[0]), value as _),
                    }),
                    Kind::Extend => {
                        let base = fuc.nodes[node.inputs[1]].ty;
                        let dest = node.ty;

                        self.emit(extend(base, dest, 1, 0))
                    }
                    Kind::UnOp { op } => {
                        let op = op.unop().expect("TODO: unary operator not supported");
                        let &[dst, oper] = allocs else { unreachable!() };
                        self.emit(op(atr(dst), atr(oper)));
                    }
                    Kind::BinOp { .. } if node.lock_rc != 0 => {}
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
                        } else if let Some(against) = op.cmp_against() {
                            let &[_, lh, rh] = node.inputs.as_slice() else { unreachable!() };
                            self.emit(extend(fuc.nodes[lh].ty, fuc.nodes[lh].ty.extend(), 0, 0));
                            self.emit(extend(fuc.nodes[rh].ty, fuc.nodes[rh].ty.extend(), 1, 1));

                            let signed = fuc.nodes[lh].ty.is_signed();
                            let op_fn = if signed { instrs::cmps } else { instrs::cmpu };
                            let &[dst, lhs, rhs] = allocs else { unreachable!() };
                            self.emit(op_fn(atr(dst), atr(lhs), atr(rhs)));
                            self.emit(instrs::cmpui(atr(dst), atr(dst), against));
                            if matches!(op, TokenKind::Eq | TokenKind::Lt | TokenKind::Gt) {
                                self.emit(instrs::not(atr(dst), atr(dst)));
                            }
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
                            let stck = fuc.nodes[*node.inputs.last().unwrap()].offset;
                            self.emit(instrs::st(r, reg::STACK_PTR, stck as _, size));
                        }
                        if let Some(PLoc::Reg(r, size)) = ret
                            && node.ty.loc(tys) == Loc::Stack
                        {
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
                        if node.ty.loc(tys) != Loc::Stack {
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

    fn emit_body(
        &mut self,
        tys: &mut Types,
        files: &[parser::Ast],
        sig: Sig,
        ralloc: &mut Regalloc,
    ) {
        self.nodes.check_final_integrity(tys, files);
        self.nodes.graphviz(tys, files);
        self.nodes.gcm();
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

        let saved = self.emit_body_code(sig, tys, files, ralloc);

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
        if !self.complete_call_graph() {
            return 1;
        }

        self.ci.emit_body(
            self.tys,
            self.files,
            Sig { args: Tuple::empty(), ret },
            &mut self.pool.ralloc,
        );
        self.ci.code.truncate(self.ci.code.len() - instrs::jala(0, 0, 0).0);
        self.ci.emit(instrs::tx());

        let func = Func {
            file,
            name: 0,
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

    fn store_mem(&mut self, region: Nid, ty: ty::Id, value: Nid) -> Nid {
        if value == NEVER {
            return NEVER;
        }

        debug_assert!(
            self.ci.nodes[region].kind != Kind::Load || self.ci.nodes[region].ty.is_pointer()
        );
        debug_assert!(self.ci.nodes[region].kind != Kind::Stre);

        self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
        let mut vc = Vc::from([VOID, value, region, self.ci.scope.store.value()]);
        for load in self.ci.scope.loads.drain(..) {
            if load.get() == value {
                load.soft_remove(&mut self.ci.nodes);
                continue;
            }
            if let Some(load) = load.remove(&mut self.ci.nodes) {
                vc.push(load);
            }
        }
        let store = self.ci.nodes.new_node_nop(ty, Kind::Stre, vc);
        self.ci.scope.store.set_value(store, &mut self.ci.nodes);
        let opted = self.ci.nodes.late_peephole(store).unwrap_or(store);
        self.ci.scope.store.set_value_remove(opted, &mut self.ci.nodes);
        opted
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
            self.cfile().path,
            self.ty_display(self.ci.nodes[region].ty)
        );
        debug_assert!(self.ci.nodes[region].kind != Kind::Stre);
        self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
        let vc = [VOID, region, self.ci.scope.store.value()];
        let load = self.ci.nodes.new_node(ty, Kind::Load, vc);
        self.ci.scope.loads.push(StrongRef::new(load, &mut self.ci.nodes));
        load
    }

    pub fn generate(&mut self, entry: FileId) {
        self.find_type(0, entry, Err("main"), self.files);
        self.make_func_reachable(0);
        self.complete_call_graph();
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
            Expr::Bool { value, .. } => Some(self.ci.nodes.new_node_lit(
                ty::Id::BOOL,
                Kind::CInt { value: value as i64 },
                [VOID],
            )),
            Expr::Number { value, .. } => Some(self.ci.nodes.new_node_lit(
                ctx.ty.filter(|ty| ty.is_integer()).unwrap_or(ty::Id::DEFAULT_INT),
                Kind::CInt { value },
                [VOID],
            )),
            Expr::Ident { id, .. }
                if let Some(index) = self.ci.scope.vars.iter().rposition(|v| v.id == id) =>
            {
                let var = &mut self.ci.scope.vars[index];
                self.ci.nodes.load_loop_var(index + 1, var, &mut self.ci.loops);

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
                    self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
                    inps.push(self.ci.scope.store.value());

                    self.ci.ctrl.set(
                        self.ci.nodes.new_node(ty::Id::VOID, Kind::Return, inps),
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
                    return match self.find_type(pos, m, Err(name), self.files).expand() {
                        ty::Kind::Builtin(ty::NEVER) => Value::NEVER,
                        ty::Kind::Global(global) => {
                            let gl = &self.tys.ins.globals[global as usize];
                            let value =
                                self.ci.nodes.new_node(gl.ty, Kind::Global { global }, [VOID]);
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

                let stack = self.ci.nodes.new_node_nop(val.ty, Kind::Stck, [VOID, MEM]);
                self.store_mem(stack, val.ty, val.id);

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
                let val =
                    self.expr_ctx(val, Ctx::default().with_ty(ctx.ty.unwrap_or(ty::Id::INT)))?;
                if !val.ty.is_integer() {
                    self.report(pos, fa!("cant negate '{}'", self.ty_display(val.ty)));
                }
                Some(self.ci.nodes.new_node_lit(val.ty, Kind::UnOp { op }, [VOID, val.id]))
            }
            Expr::BinOp { left, op: TokenKind::Decl, right, .. } => {
                let mut right = self.expr(right)?;
                if right.ty.loc(self.tys) == Loc::Stack {
                    let stck = self.ci.nodes.new_node_nop(right.ty, Kind::Stck, [VOID, MEM]);
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
            Expr::BinOp { left, pos, op, right }
                if !matches!(op, TokenKind::Assign | TokenKind::Decl) =>
            {
                let mut lhs = self.raw_expr_ctx(left, ctx)?;
                self.strip_var(&mut lhs);

                match lhs.ty.expand() {
                    _ if lhs.ty.is_pointer() || lhs.ty.is_integer() || lhs.ty == ty::Id::BOOL => {
                        if mem::take(&mut lhs.ptr) {
                            lhs.id = self.load_mem(lhs.id, lhs.ty);
                        }
                        self.ci.nodes.lock(lhs.id);
                        let rhs = self.expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                        self.ci.nodes.unlock(lhs.id);
                        let mut rhs = rhs?;
                        self.strip_var(&mut rhs);
                        let ty = self.binop_ty(pos, &mut lhs, &mut rhs, op);
                        let inps = [VOID, lhs.id, rhs.id];
                        Some(self.ci.nodes.new_node_lit(
                            ty::bin_ret(ty, op),
                            Kind::BinOp { op },
                            inps,
                        ))
                    }
                    ty::Kind::Struct(s) if op.is_homogenous() => {
                        self.ci.nodes.lock(lhs.id);
                        let rhs = self.raw_expr_ctx(right, Ctx::default().with_ty(lhs.ty));
                        self.ci.nodes.unlock(lhs.id);
                        let mut rhs = rhs?;
                        self.strip_var(&mut rhs);
                        self.assert_ty(pos, &mut rhs, lhs.ty, "struct operand");
                        let dst = self.ci.nodes.new_node(lhs.ty, Kind::Stck, [VOID, MEM]);
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
            Expr::Embed { id, .. } => {
                let glob = &self.tys.ins.globals[id as usize];
                let g = self.ci.nodes.new_node(glob.ty, Kind::Global { global: id }, [VOID]);
                Some(Value::ptr(g).ty(glob.ty))
            }
            Expr::Directive { name: "sizeof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(self.ci.nodes.new_node_lit(
                    ctx.ty.filter(|ty| ty.is_integer()).unwrap_or(ty::Id::DEFAULT_INT),
                    Kind::CInt { value: self.tys.size_of(ty) as _ },
                    [VOID],
                ))
            }
            Expr::Directive { name: "alignof", args: [ty], .. } => {
                let ty = self.ty(ty);
                Some(self.ci.nodes.new_node_lit(
                    ctx.ty.filter(|ty| ty.is_integer()).unwrap_or(ty::Id::DEFAULT_INT),
                    Kind::CInt { value: self.tys.align_of(ty) as _ },
                    [VOID],
                ))
            }
            Expr::Directive { name: "bitcast", args: [val], pos } => {
                let mut val = self.raw_expr(val)?;
                self.strip_var(&mut val);

                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "resulting type cannot be inferred from context, \
                        consider using `@as(<ty>, @bitcast(<expr>))` to hint the type",
                    );
                    return Value::NEVER;
                };

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
                        let stack = self.ci.nodes.new_node_nop(ty, Kind::Stck, [VOID, MEM]);
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

                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "resulting integer cannot be inferred from context, \
                        consider using `@as(<int_ty>, @intcast(<expr>))` to hint the type",
                    );
                    return Value::NEVER;
                };

                if !ty.is_integer() {
                    self.report(
                        expr.pos(),
                        fa!(
                            "intcast is inferred to output '{}', which is not an integer",
                            self.ty_display(ty)
                        ),
                    );
                }

                if self.tys.size_of(val.ty) <= self.tys.size_of(ty) {
                    val.ty = ty;
                    return Some(val);
                }

                let value = (1i64 << (self.tys.size_of(ty) * 8)) - 1;
                let mask = self.ci.nodes.new_node_nop(val.ty, Kind::CInt { value }, [VOID]);
                let inps = [VOID, val.id, mask];
                Some(self.ci.nodes.new_node_lit(ty, Kind::BinOp { op: TokenKind::Band }, inps))
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
                let Some(ty) = ctx.ty else {
                    self.report(
                        pos,
                        "return type cannot be inferred from context, \
                        consider using `@as(<return_ty>, @eca(<expr>...))` to hint the type",
                    );
                    return Value::NEVER;
                };

                let mut inps = Vc::from([NEVER]);
                let arg_base = self.tys.tmp.args.len();
                let mut has_ptr_arg = false;
                for arg in args {
                    let value = self.expr(arg)?;
                    has_ptr_arg |= value.ty.has_pointers(self.tys);
                    self.tys.tmp.args.push(value.ty);
                    debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                    self.ci.nodes.lock(value.id);
                    inps.push(value.id);
                }

                let args = self.tys.pack_args(arg_base).expect("TODO");

                for &n in inps.iter().skip(1) {
                    self.ci.nodes.unlock(n);
                }

                if has_ptr_arg {
                    inps.push(self.ci.scope.store.value());
                    self.ci.scope.loads.retain_mut(|load| {
                        if inps.contains(&load.get()) {
                            return true;
                        }

                        if let Some(load) = mem::take(load).remove(&mut self.ci.nodes) {
                            inps.push(load);
                        }

                        false
                    });
                }

                let alt_value = match ty.loc(self.tys) {
                    Loc::Reg => None,
                    Loc::Stack => {
                        let stck = self.ci.nodes.new_node_nop(ty, Kind::Stck, [VOID, MEM]);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(ty))
                    }
                };

                inps[0] = self.ci.ctrl.get();
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty, Kind::Call { func: ty::ECA, args }, inps),
                    &mut self.ci.nodes,
                );

                if has_ptr_arg {
                    self.store_mem(VOID, ty::Id::VOID, VOID);
                }

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
                let mut has_ptr_arg = false;
                while let Some(ty) = tys.next(self.tys) {
                    let carg = cargs.next().unwrap();
                    let Some(arg) = args.next() else { break };
                    let Arg::Value(ty) = ty else { continue };
                    has_ptr_arg |= ty.has_pointers(self.tys);

                    let mut value = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    debug_assert_ne!(self.ci.nodes[value.id].kind, Kind::Stre);
                    self.assert_ty(arg.pos(), &mut value, ty, fa!("argument {}", carg.name));

                    self.ci.nodes.lock(value.id);
                    inps.push(value.id);
                }

                for &n in inps.iter().skip(1) {
                    self.ci.nodes.unlock(n);
                }

                if has_ptr_arg {
                    inps.push(self.ci.scope.store.value());
                    self.ci.scope.loads.retain_mut(|load| {
                        if inps.contains(&load.get()) {
                            return true;
                        }

                        if let Some(load) = mem::take(load).remove(&mut self.ci.nodes) {
                            inps.push(load);
                        }

                        false
                    });
                }

                let alt_value = match sig.ret.loc(self.tys) {
                    Loc::Reg => None,
                    Loc::Stack => {
                        let stck = self.ci.nodes.new_node_nop(sig.ret, Kind::Stck, [VOID, MEM]);
                        inps.push(stck);
                        Some(Value::ptr(stck).ty(sig.ret))
                    }
                };

                inps[0] = self.ci.ctrl.get();
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(sig.ret, Kind::Call { func: fu, args: sig.args }, inps),
                    &mut self.ci.nodes,
                );

                if has_ptr_arg {
                    self.store_mem(VOID, ty::Id::VOID, VOID);
                }

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

                if self.expr(body).is_some() && sig.ret == ty::Id::VOID {
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

                        let mem = self.ci.nodes.new_node(aty, Kind::Stck, [VOID, MEM]);

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
                let value = self.ty(expr).repr() as i64;
                Some(self.ci.nodes.new_node_lit(ty::Id::TYPE, Kind::CInt { value }, [VOID]))
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
                let mut offs = OffsetIter::new(s, self.tys)
                    .into_iter(self.tys)
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

                ret
            }
            Expr::Loop { body, .. } => {
                self.ci.ctrl.set(
                    self.ci.nodes.new_node(ty::Id::VOID, Kind::Loop, [self.ci.ctrl.get(); 2]),
                    &mut self.ci.nodes,
                );
                self.ci.loops.push(Loop {
                    node: self.ci.ctrl.get(),
                    ctrl: [StrongRef::DEFAULT; 2],
                    ctrl_scope: core::array::from_fn(|_| Default::default()),
                    scope: self.ci.scope.dup(&mut self.ci.nodes),
                });

                for var in &mut self.ci.scope.iter_mut() {
                    var.set_value(VOID, &mut self.ci.nodes);
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
                    for (loop_var, scope_var) in self.ci.scope.iter_mut().zip(scope.iter_mut()) {
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
                    scope.clear(&mut self.ci.nodes);
                    self.ci.ctrl.set(NEVER, &mut self.ci.nodes);
                    return None;
                };

                self.ci.ctrl.set(bre, &mut self.ci.nodes);

                mem::swap(&mut self.ci.scope, &mut bres);

                debug_assert_eq!(self.ci.scope.vars.len(), scope.vars.len());
                debug_assert_eq!(self.ci.scope.vars.len(), bres.vars.len());

                self.ci.nodes.lock(node);

                for ((dest_var, scope_var), loop_var) in
                    self.ci.scope.iter_mut().zip(scope.iter_mut()).zip(bres.iter_mut())
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

                scope.clear(&mut self.ci.nodes);
                bres.clear(&mut self.ci.nodes);
                self.ci.scope.loads.drain(..).for_each(|l| _ = l.remove(&mut self.ci.nodes));

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
                    let branch = match self.tof(if_node).expand().inner() {
                        ty::LEFT_UNREACHABLE => else_,
                        ty::RIGHT_UNREACHABLE => Some(then),
                        _ => break 'b,
                    };

                    self.ci.nodes.remove(if_node);

                    if let Some(branch) = branch {
                        return self.expr(branch);
                    } else {
                        return Some(Value::VOID);
                    }
                }

                self.ci.nodes.load_loop_store(&mut self.ci.scope.store, &mut self.ci.loops);
                let orig_store = self.ci.scope.store.dup(&mut self.ci.nodes);
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

                orig_store.remove(&mut self.ci.nodes);

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
                Some(Value::VOID)
            }
        }
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
        if mem::take(&mut n.ptr) {
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
        if mem::take(&mut n.var) {
            let id = (u16::MAX - n.id) as usize;
            n.ptr = self.ci.scope.vars[id].ptr;
            n.id = self.ci.scope.vars[id].value();
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
        }

        None
    }

    #[inline(always)]
    fn tof(&self, id: Nid) -> ty::Id {
        self.ci.nodes[id].ty
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
                }
            }
        }

        if self.expr(body).is_some() && sig.ret == ty::Id::VOID {
            self.report(
                body.pos(),
                "expected all paths in the fucntion to return \
                or the return type to be 'void'",
            );
        }

        self.ci.scope.vars.drain(..).for_each(|v| v.remove_ignore_arg(&mut self.ci.nodes));

        self.ci.finalize();

        if self.errors.borrow().len() == prev_err_len {
            self.ci.emit_body(self.tys, self.files, sig, &mut self.pool.ralloc);
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
        parser::Display::new(&self.cfile().file, ast)
    }

    #[must_use]
    #[track_caller]
    fn binop_ty(&mut self, pos: Pos, lhs: &mut Value, rhs: &mut Value, op: TokenKind) -> ty::Id {
        if let Some(upcasted) = lhs.ty.try_upcast(rhs.ty, ty::TyCheck::BinOp) {
            let to_correct = if lhs.ty != upcasted {
                Some(lhs)
            } else if rhs.ty != upcasted {
                Some(rhs)
            } else {
                None
            };

            if let Some(oper) = to_correct {
                oper.ty = upcasted;
                if mem::take(&mut oper.ptr) {
                    oper.id = self.load_mem(oper.id, oper.ty);
                }
                oper.id = self.ci.nodes.new_node(upcasted, Kind::Extend, [VOID, oper.id]);
                if matches!(op, TokenKind::Add | TokenKind::Sub)
                    && let Some(elem) = self.tys.base_of(upcasted)
                {
                    let value = self.tys.size_of(elem) as i64;
                    let cnst =
                        self.ci.nodes.new_node_nop(ty::Id::INT, Kind::CInt { value }, [VOID]);
                    oper.id =
                        self.ci.nodes.new_node(upcasted, Kind::BinOp { op: TokenKind::Mul }, [
                            VOID, oper.id, cnst,
                        ]);
                }
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
        if let Some(upcasted) = src.ty.try_upcast(expected, ty::TyCheck::Assign)
            && upcasted == expected
        {
            if src.ty != upcasted {
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
                src.ty = upcasted;
                if mem::take(&mut src.ptr) {
                    src.id = self.load_mem(src.id, src.ty);
                }
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
    fn report(&self, pos: Pos, msg: impl core::fmt::Display) {
        let mut buf = self.errors.borrow_mut();
        write!(buf, "{}", self.cfile().report(pos, msg)).unwrap();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: impl Display) {
        log::info!("{ast:#?}");
        self.report(ast.pos(), fa!("compiler does not (yet) know how to handle ({hint})"));
    }

    fn cfile(&self) -> &'a parser::Ast {
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
        self.ci.finalize();

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

        self.ci.finalize();

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

    fn report(&self, pos: Pos, msg: impl Display) -> ty::Id {
        self.report(pos, msg);
        ty::Id::NEVER
    }

    fn find_local_ty(&mut self, ident: Ident) -> Option<ty::Id> {
        self.ci.scope.vars.iter().rfind(|v| (v.id == ident && v.value() == NEVER)).map(|v| v.ty)
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

    fn drg(&mut self, nid: Nid) -> regalloc2::Operand {
        regalloc2::Operand::reg_def(self.rg(nid))
    }

    fn rg(&self, nid: Nid) -> VReg {
        debug_assert!(
            !self.nodes.is_cfg(nid) || matches!(self.nodes[nid].kind, Kind::Call { .. }),
            "{:?}",
            self.nodes[nid]
        );
        debug_assert_eq!(self.nodes[nid].lock_rc, 0, "{:?}", self.nodes[nid]);
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
            Kind::End => {}
            Kind::If => {
                self.nodes[nid].ralloc_backref = self.nodes[prev].ralloc_backref;

                let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Kind::BinOp { op } = self.nodes[cond].kind
                    && let Some((_, swapped)) = op.cond_op(node.ty.is_signed())
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

                let (ret, mut parama) = self.tys.parama(self.sig.ret);
                let mut typs = self.sig.args.args();
                #[allow(clippy::unnecessary_to_owned)]
                let mut args = self.nodes[VOID].outputs[2..].to_owned().into_iter();
                while let Some(ty) = typs.next_value(self.tys) {
                    let arg = args.next().unwrap();
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
                if op.cond_op(node.ty.is_signed()).is_some()
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
            Kind::Stck
                if node.ty.loc(self.tys) == Loc::Reg && node.outputs.iter().all(|&n| {
                    matches!(self.nodes[n].kind, Kind::Stre | Kind::Load)
                        || matches!(self.nodes[n].kind, Kind::BinOp { op: TokenKind::Add }
                    if self.nodes.is_const(self.nodes[n].inputs[2])
                        && self.nodes[n]
                            .outputs
                            .iter()
                            .all(|&n| matches!(self.nodes[n].kind, Kind::Stre | Kind::Load)))
                }) => {}
            Kind::Stck if self.tys.size_of(node.ty) == 0 => self.nodes.lock(nid),
            Kind::Stck => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Idk => {
                let ops = vec![self.drg(nid)];
                self.add_instr(nid, ops);
            }
            Kind::Phi | Kind::Arg | Kind::Mem => {}
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
            Kind::Stre if node.inputs[2] == VOID => self.nodes.lock(nid),
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

        for i in 1..nodes[node].inputs.len() {
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

    debug_assert_eq!(
        nodes
            .iter()
            .map(|(n, _)| n)
            .filter(|&n| !nodes.visited.get(n) && !matches!(nodes[n].kind, Kind::Arg | Kind::Mem))
            .collect::<Vec<_>>(),
        vec![],
        "{:?}",
        nodes
            .iter()
            .filter(|&(n, nod)| !nodes.visited.get(n) && !matches!(nod.kind, Kind::Arg | Kind::Mem))
            .collect::<Vec<_>>()
    );
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
        debug_assert_ne!(idepth(nodes, is), idepth(nodes, then), "{is} {then}");
        loop_depth(is, nodes) < loop_depth(then, nodes)
            || idepth(nodes, is) > idepth(nodes, then)
            || nodes[then].kind == Kind::If
    }

    if !nodes.visited.set(node) {
        return;
    }

    for usage in nodes[node].outputs.clone() {
        if is_forward_edge(usage, node, nodes) && nodes[node].kind == Kind::Stre {
            push_down(nodes, usage);
        }
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
    while cursor != nodes[node].inputs[0] {
        cursor = idom(nodes, cursor);
        if better(nodes, cursor, min) {
            min = cursor;
        }
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
        super::{Codegen, CodegenCtx},
        crate::{
            lexer::TokenKind,
            parser::{self},
        },
        alloc::{string::String, vec::Vec},
        core::{fmt::Write, hash::BuildHasher, ops::Range},
    };

    #[derive(Default)]
    struct Rand(pub u64);

    impl Rand {
        pub fn next(&mut self) -> u64 {
            self.0 = crate::FnvBuildHasher::default().hash_one(self.0);
            self.0
        }

        pub fn range(&mut self, min: u64, max: u64) -> u64 {
            self.next() % (max - min) + min
        }
    }

    #[derive(Default)]
    struct FuncGen {
        rand: Rand,
        buf: String,
    }

    impl FuncGen {
        fn gen(&mut self, seed: u64) -> &str {
            self.rand = Rand(seed);
            self.buf.clear();
            self.buf.push_str("main := fn(): void { return ");
            self.expr().unwrap();
            self.buf.push('}');
            &self.buf
        }

        fn expr(&mut self) -> core::fmt::Result {
            match self.rand.range(0, 100) {
                0..80 => {
                    write!(self.buf, "{}", self.rand.next())
                }
                80..100 => {
                    self.expr()?;
                    let ops = [
                        TokenKind::Add,
                        TokenKind::Sub,
                        TokenKind::Mul,
                        TokenKind::Div,
                        TokenKind::Shl,
                        TokenKind::Eq,
                        TokenKind::Ne,
                        TokenKind::Lt,
                        TokenKind::Gt,
                        TokenKind::Le,
                        TokenKind::Ge,
                        TokenKind::Band,
                        TokenKind::Bor,
                        TokenKind::Xor,
                        TokenKind::Mod,
                        TokenKind::Shr,
                    ];
                    let op = ops[self.rand.range(0, ops.len() as u64) as usize];
                    write!(self.buf, " {op} ")?;
                    self.expr()
                }
                _ => unreachable!(),
            }
        }
    }

    fn fuzz(seed_range: Range<u64>) {
        let mut gen = FuncGen::default();
        let mut ctx = CodegenCtx::default();
        for i in seed_range {
            ctx.clear();
            let src = gen.gen(i);
            let parsed = parser::Ast::new("fuzz", src, &mut ctx.parser, &mut parser::no_loader);

            let mut cdg = Codegen::new(core::slice::from_ref(&parsed), &mut ctx);
            cdg.generate(0);
        }
    }

    #[test]
    #[ignore]
    fn fuzz_test() {
        _ = log::set_logger(&crate::fs::Logger);
        log::set_max_level(log::LevelFilter::Info);
        fuzz(0..10000);
    }

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
        structs_in_registers;
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
    }
}
