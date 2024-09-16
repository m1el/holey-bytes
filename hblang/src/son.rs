#![allow(dead_code)]
use {
    crate::{
        ident::{self, Ident},
        instrs::{self, jal},
        lexer::{self, TokenKind},
        log,
        parser::{
            self,
            idfl::{self},
            CommentOr, Expr, ExprRef, FileId, Pos, StructField,
        },
        task, ty, Field, Func, Reloc, Sig, Struct, SymKey, TypedReloc, Types,
    },
    core::fmt,
    std::{
        cell::RefCell,
        collections::hash_map,
        fmt::{Debug, Display, Write},
        hash::{Hash as _, Hasher},
        mem::{self, MaybeUninit},
        ops::{self, Deref, DerefMut, Not},
        ptr::Unique,
    },
};

macro_rules! node_loc {
    ($self:expr, $value:expr) => {
        $self.ci.colors[$self.ci.nodes[$value].color as usize - 1].loc
        //$self.ci.colors[dbg!(&$self.ci.nodes[$value]).color as usize - 1].loc
    };
}

struct Drom(&'static str);

impl Drop for Drom {
    fn drop(&mut self) {
        log::inf!("{}", self.0);
    }
}

const VC_SIZE: usize = 16;
const INLINE_ELEMS: usize = VC_SIZE / 2 - 1;
const VOID: Nid = 0;
const NEVER: Nid = 1;

union Vc {
    inline: InlineVc,
    alloced: AllocedVc,
}

impl Default for Vc {
    fn default() -> Self {
        Vc { inline: InlineVc { elems: MaybeUninit::uninit(), cap: 0 } }
    }
}

impl Debug for Vc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl Vc {
    fn is_inline(&self) -> bool {
        unsafe { self.inline.cap <= INLINE_ELEMS as Nid }
    }

    fn layout(&self) -> Option<std::alloc::Layout> {
        unsafe {
            self.is_inline()
                .not()
                .then(|| std::alloc::Layout::array::<Nid>(self.alloced.cap as _).unwrap_unchecked())
        }
    }

    fn len(&self) -> usize {
        unsafe {
            if self.is_inline() {
                self.inline.cap as _
            } else {
                self.alloced.len as _
            }
        }
    }

    fn find(&self, needle: Nid) -> usize {
        self.iter().position(|&n| n == needle).unwrap()
    }

    fn len_mut(&mut self) -> &mut Nid {
        unsafe {
            if self.is_inline() {
                &mut self.inline.cap
            } else {
                &mut self.alloced.len
            }
        }
    }

    fn as_ptr(&self) -> *const Nid {
        unsafe {
            match self.is_inline() {
                true => self.inline.elems.as_ptr().cast(),
                false => self.alloced.base.as_ptr(),
            }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut Nid {
        unsafe {
            match self.is_inline() {
                true => self.inline.elems.as_mut_ptr().cast(),
                false => self.alloced.base.as_ptr(),
            }
        }
    }

    fn as_slice(&self) -> &[Nid] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    fn as_slice_mut(&mut self) -> &mut [Nid] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    fn push(&mut self, value: Nid) {
        if let Some(layout) = self.layout()
            && unsafe { self.alloced.len == self.alloced.cap }
        {
            unsafe {
                self.alloced.cap *= 2;
                self.alloced.base = Unique::new_unchecked(
                    std::alloc::realloc(
                        self.alloced.base.as_ptr().cast(),
                        layout,
                        self.alloced.cap as usize * std::mem::size_of::<Nid>(),
                    )
                    .cast(),
                );
            }
        } else if self.len() == INLINE_ELEMS {
            unsafe {
                let mut allcd =
                    Self::alloc((self.inline.cap + 1).next_power_of_two() as _, self.len());
                std::ptr::copy_nonoverlapping(self.as_ptr(), allcd.as_mut_ptr(), self.len());
                *self = allcd;
            }
        }

        unsafe {
            *self.len_mut() += 1;
            self.as_mut_ptr().add(self.len() - 1).write(value);
        }
    }

    unsafe fn alloc(cap: usize, len: usize) -> Self {
        debug_assert!(cap > INLINE_ELEMS);
        let layout = unsafe { std::alloc::Layout::array::<Nid>(cap).unwrap_unchecked() };
        let alloc = unsafe { std::alloc::alloc(layout) };
        unsafe {
            Vc {
                alloced: AllocedVc {
                    base: Unique::new_unchecked(alloc.cast()),
                    len: len as _,
                    cap: cap as _,
                },
            }
        }
    }

    fn swap_remove(&mut self, index: usize) {
        let len = self.len() - 1;
        self.as_slice_mut().swap(index, len);
        *self.len_mut() -= 1;
    }

    fn remove(&mut self, index: usize) {
        self.as_slice_mut().copy_within(index + 1.., index);
        *self.len_mut() -= 1;
    }
}

impl Drop for Vc {
    fn drop(&mut self) {
        if let Some(layout) = self.layout() {
            unsafe {
                std::alloc::dealloc(self.alloced.base.as_ptr().cast(), layout);
            }
        }
    }
}

impl Clone for Vc {
    fn clone(&self) -> Self {
        self.as_slice().into()
    }
}

impl IntoIterator for Vc {
    type IntoIter = VcIntoIter;
    type Item = Nid;

    fn into_iter(self) -> Self::IntoIter {
        VcIntoIter { start: 0, end: self.len(), vc: self }
    }
}

struct VcIntoIter {
    start: usize,
    end: usize,
    vc: Vc,
}

impl Iterator for VcIntoIter {
    type Item = Nid;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        let ret = unsafe { std::ptr::read(self.vc.as_slice().get_unchecked(self.start)) };
        self.start += 1;
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for VcIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        self.end -= 1;
        Some(unsafe { std::ptr::read(self.vc.as_slice().get_unchecked(self.end)) })
    }
}

impl ExactSizeIterator for VcIntoIter {}

impl<const SIZE: usize> From<[Nid; SIZE]> for Vc {
    fn from(value: [Nid; SIZE]) -> Self {
        value.as_slice().into()
    }
}

impl<'a> From<&'a [Nid]> for Vc {
    fn from(value: &'a [Nid]) -> Self {
        if value.len() <= INLINE_ELEMS {
            let mut dflt = Self::default();
            unsafe {
                std::ptr::copy_nonoverlapping(value.as_ptr(), dflt.as_mut_ptr(), value.len())
            };
            dflt.inline.cap = value.len() as _;
            dflt
        } else {
            let mut allcd = unsafe { Self::alloc(value.len(), value.len()) };
            unsafe {
                std::ptr::copy_nonoverlapping(value.as_ptr(), allcd.as_mut_ptr(), value.len())
            };
            allcd
        }
    }
}

impl Deref for Vc {
    type Target = [Nid];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Vc {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct InlineVc {
    cap: Nid,
    elems: MaybeUninit<[Nid; INLINE_ELEMS]>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct AllocedVc {
    cap: Nid,
    len: Nid,
    base: Unique<Nid>,
}

#[derive(Default)]
struct BitSet {
    data: Vec<usize>,
}

impl BitSet {
    const ELEM_SIZE: usize = std::mem::size_of::<usize>() * 8;

    pub fn clear(&mut self, bit_size: usize) {
        let new_len = (bit_size + Self::ELEM_SIZE - 1) / Self::ELEM_SIZE;
        self.data.clear();
        self.data.resize(new_len, 0);
    }

    #[track_caller]
    pub fn set(&mut self, idx: Nid) -> bool {
        let idx = idx as usize;
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        let prev = self.data[data_idx] & (1 << sub_idx);
        self.data[data_idx] |= 1 << sub_idx;
        prev == 0
    }

    fn unset(&mut self, idx: Nid) {
        let idx = idx as usize;
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        self.data[data_idx] &= !(1 << sub_idx);
    }

    fn get(&self, idx: Nid) -> bool {
        let idx = idx as usize;
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        self.data[data_idx] & (1 << sub_idx) == 1
    }
}

type Nid = u16;

pub mod reg {
    pub const STACK_PTR: Reg = 254;
    pub const ZERO: Reg = 0;
    pub const RET: Reg = 1;
    pub const RET_ADDR: Reg = 31;

    pub type Reg = u8;

    #[derive(Default)]
    struct AllocMeta {
        rc: u16,
        depth: u16,
        #[cfg(debug_assertions)]
        allocated_at: Option<std::backtrace::Backtrace>,
    }

    struct Metas([AllocMeta; 256]);

    impl Default for Metas {
        fn default() -> Self {
            Metas(std::array::from_fn(|_| AllocMeta::default()))
        }
    }

    #[derive(Default)]
    pub struct Alloc {
        meta: Metas,
        free: Vec<Reg>,
        max_used: Reg,
    }

    impl Alloc {
        pub fn init(&mut self) {
            self.free.clear();
            self.free.extend((32..=253).rev());
            self.max_used = RET_ADDR;
        }

        pub fn allocate(&mut self, depth: u16) -> Reg {
            let reg = self.free.pop().expect("TODO: we need to spill");
            self.max_used = self.max_used.max(reg);
            self.meta.0[reg as usize] = AllocMeta {
                depth,
                rc: 1,
                #[cfg(debug_assertions)]
                allocated_at: Some(std::backtrace::Backtrace::capture()),
            };
            reg
        }

        pub fn dup(&mut self, reg: Reg) {
            self.meta.0[reg as usize].rc += 1;
        }

        pub fn free(&mut self, reg: Reg) {
            if self.meta.0[reg as usize].rc == 1 {
                self.free.push(reg);
                self.meta.0[reg as usize] = Default::default();
            } else {
                self.meta.0[reg as usize].rc -= 1;
            }
        }

        pub fn pushed_size(&self) -> usize {
            ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
        }

        pub fn mark_leaked(&mut self, reg: u8) {
            self.meta.0[reg as usize].rc = u16::MAX;
        }
    }
}

struct LookupEntry {
    nid: Nid,
    hash: u64,
}

#[derive(Default)]
struct IdentityHash(u64);

impl std::hash::Hasher for IdentityHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unimplemented!()
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

impl std::hash::Hash for LookupEntry {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

type Lookup = std::collections::hash_map::HashMap<
    LookupEntry,
    (),
    std::hash::BuildHasherDefault<IdentityHash>,
>;

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
    fn remove_low(&mut self, id: Nid) -> Node {
        let value = mem::replace(&mut self.values[id as usize], Err(self.free)).unwrap();
        self.free = id;
        value
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = Nid::MAX;
    }

    fn new_node_nop(&mut self, ty: impl Into<ty::Id>, kind: Kind, inps: impl Into<Vc>) -> Nid {
        let ty = ty.into();

        let node = Node { inputs: inps.into(), kind, ty, ..Default::default() };

        let mut lookup_meta = None;
        if !node.is_lazy_phi() {
            let (raw_entry, hash) = Self::find_node(&mut self.lookup, &self.values, &node);

            let entry = match raw_entry {
                hash_map::RawEntryMut::Occupied(mut o) => return o.get_key_value().0.nid,
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
            entry.insert(LookupEntry { nid: free, hash }, ());
        }
        free
    }

    fn find_node<'a>(
        lookup: &'a mut Lookup,
        values: &[Result<Node, Nid>],
        node: &Node,
    ) -> (
        hash_map::RawEntryMut<'a, LookupEntry, (), std::hash::BuildHasherDefault<IdentityHash>>,
        u64,
    ) {
        let mut hasher = crate::FnvHasher::default();
        node.key().hash(&mut hasher);
        let hash = hasher.finish();
        let entry = lookup
            .raw_entry_mut()
            .from_hash(hash, |n| values[n.nid as usize].as_ref().unwrap().key() == node.key());
        (entry, hash)
    }

    fn remove_node_lookup(&mut self, target: Nid) {
        if !self[target].is_lazy_phi() {
            match Self::find_node(
                &mut self.lookup,
                &self.values,
                self.values[target as usize].as_ref().unwrap(),
            )
            .0
            {
                hash_map::RawEntryMut::Occupied(o) => o.remove(),
                hash_map::RawEntryMut::Vacant(_) => unreachable!(),
            };
        }
    }

    fn new_node(&mut self, ty: impl Into<ty::Id>, kind: Kind, inps: impl Into<Vc>) -> Nid {
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
        match self[target].kind {
            Kind::Start => {}
            Kind::End => {}
            Kind::BinOp { op } => return self.peephole_binop(target, op),
            Kind::UnOp { op } => return self.peephole_unop(target, op),
            Kind::Return => {}
            Kind::Tuple { .. } => {}
            Kind::CInt { .. } => {}
            Kind::Call { .. } => {}
            Kind::If => return self.peephole_if(target),
            Kind::Region => {}
            Kind::Phi => return self.peephole_phi(target),
            Kind::Loop => {}
        }
        None
    }

    fn peephole_phi(&mut self, target: Nid) -> Option<Nid> {
        if self[target].inputs[1] == self[target].inputs[2] {
            return Some(self[target].inputs[1]);
        }

        None
    }

    fn peephole_if(&mut self, target: Nid) -> Option<Nid> {
        let cond = self[target].inputs[1];
        if let Kind::CInt { value } = self[cond].kind {
            let ty = if value == 0 { ty::LEFT_UNREACHABLE } else { ty::RIGHT_UNREACHABLE };
            return Some(self.new_node_nop(ty, Kind::If, [self[target].inputs[0], cond]));
        }

        None
    }

    fn peephole_unop(&mut self, target: Nid, op: TokenKind) -> Option<Nid> {
        let &[ctrl, oper] = self[target].inputs.as_slice() else { unreachable!() };
        let ty = self[target].ty;

        if let Kind::CInt { value } = self[oper].kind {
            return Some(self.new_node(ty, Kind::CInt { value: op.apply_unop(value) }, [ctrl]));
        }

        None
    }

    fn peephole_binop(&mut self, target: Nid, op: TokenKind) -> Option<Nid> {
        use {Kind as K, TokenKind as T};
        let &[ctrl, mut lhs, mut rhs] = self[target].inputs.as_slice() else { unreachable!() };
        let ty = self[target].ty;

        if let (&K::CInt { value: a }, &K::CInt { value: b }) = (&self[lhs].kind, &self[rhs].kind) {
            return Some(self.new_node(ty, K::CInt { value: op.apply_binop(a, b) }, [ctrl]));
        }

        if lhs == rhs {
            match op {
                T::Sub => return Some(self.new_node(ty, K::CInt { value: 0 }, [ctrl])),
                T::Add => {
                    let rhs = self.new_node_nop(ty, K::CInt { value: 2 }, [ctrl]);
                    return Some(self.new_node(ty, K::BinOp { op: T::Mul }, [ctrl, lhs, rhs]));
                }
                _ => {}
            }
        }

        // this is more general the pushing constants to left to help deduplicate expressions more
        let mut changed = false;
        if op.is_comutative() && self[lhs].key() < self[rhs].key() {
            std::mem::swap(&mut lhs, &mut rhs);
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
                    self.new_node_nop(ty, K::CInt { value: op.apply_binop(av, bv) }, [ctrl]);
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

        changed.then(|| self.new_node(ty, self[target].kind, [ctrl, lhs, rhs]))
    }

    fn is_const(&self, id: Nid) -> bool {
        matches!(self[id].kind, Kind::CInt { .. })
    }

    fn replace(&mut self, target: Nid, with: Nid) {
        let mut back_press = 0;
        for i in 0..self[target].outputs.len() {
            let out = self[target].outputs[i - back_press];
            let index = self[out].inputs.iter().position(|&p| p == target).unwrap();
            let prev_len = self[target].outputs.len();
            self.modify_input(out, index, with);
            back_press += (self[target].outputs.len() != prev_len) as usize;
        }

        self.remove(target);
    }

    fn modify_input(&mut self, target: Nid, inp_index: usize, with: Nid) -> Nid {
        self.remove_node_lookup(target);
        debug_assert_ne!(self[target].inputs[inp_index], with);

        let prev = self[target].inputs[inp_index];
        self[target].inputs[inp_index] = with;
        let (entry, hash) = Self::find_node(
            &mut self.lookup,
            &self.values,
            self.values[target as usize].as_ref().unwrap(),
        );
        match entry {
            hash_map::RawEntryMut::Occupied(mut other) => {
                let rpl = other.get_key_value().0.nid;
                self[target].inputs[inp_index] = prev;
                self.replace(target, rpl);
                rpl
            }
            hash_map::RawEntryMut::Vacant(slot) => {
                slot.insert(LookupEntry { nid: target, hash }, ());
                let index = self[prev].outputs.iter().position(|&o| o == target).unwrap();
                self[prev].outputs.swap_remove(index);
                self[with].outputs.push(target);

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

    fn graphviz_low(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;

        for (i, node) in self.iter() {
            let color = if self.is_cfg(i) { "yellow" } else { "white" };
            writeln!(out, "node{i}[label=\"{}\" color={color}]", node.kind)?;
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

    #[allow(clippy::format_in_format_args)]
    fn basic_blocks_instr(&mut self, out: &mut String, node: Nid) -> std::fmt::Result {
        if self[node].kind != Kind::Loop && self[node].kind != Kind::Region {
            write!(out, "  {node:>2}-c{:>2}: ", self[node].color)?;
        }
        match self[node].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::If => write!(out, "  if:      "),
            Kind::Region | Kind::Loop => writeln!(out, "      goto: {node}"),
            Kind::Return => write!(out, " ret:      "),
            Kind::CInt { value } => write!(out, "cint: #{value:<4}"),
            Kind::Phi => write!(out, " phi:      "),
            Kind::Tuple { index } => write!(out, " arg: {index:<5}"),
            Kind::BinOp { op } | Kind::UnOp { op } => {
                write!(out, "{:>4}:      ", op.name())
            }
            Kind::Call { func } => {
                write!(out, "call: {func} {}  ", self[node].depth)
            }
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

    fn basic_blocks_low(&mut self, out: &mut String, mut node: Nid) -> std::fmt::Result {
        let iter = |nodes: &Nodes, node| nodes[node].outputs.clone().into_iter().rev();
        while self.visited.set(node) {
            match self[node].kind {
                Kind::Start => {
                    writeln!(out, "start: {}", self[node].depth)?;
                    let mut cfg_index = Nid::MAX;
                    for o in iter(self, node) {
                        self.basic_blocks_instr(out, o)?;
                        if self[o].kind == (Kind::Tuple { index: 0 }) {
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
                Kind::Tuple { .. } => {
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
                            && (self[node].outputs[0] != o || std::mem::take(&mut print_ret))
                        {
                            self.basic_blocks_instr(out, o)?;
                        }
                        if self.is_cfg(o) {
                            cfg_index = o;
                        }
                    }
                    node = cfg_index;
                }
                Kind::CInt { .. } | Kind::Phi | Kind::BinOp { .. } | Kind::UnOp { .. } => {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    fn basic_blocks(&mut self) {
        let mut out = String::new();
        self.visited.clear(self.values.len());
        self.basic_blocks_low(&mut out, VOID).unwrap();
        log::inf!("{out}");
    }

    fn graphviz(&self) {
        let out = &mut String::new();
        _ = self.graphviz_low(out);
        log::inf!("{out}");
    }

    fn is_cfg(&self, o: Nid) -> bool {
        self[o].kind.is_cfg()
    }

    fn check_final_integrity(&self) {
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

    fn late_peephole(&mut self, target: Nid) -> Nid {
        if let Some(id) = self.peephole(target) {
            self.replace(target, id);
            return id;
        }
        target
    }

    fn load_loop_value(&mut self, index: usize, value: &mut Nid, loops: &mut [Loop]) {
        if *value != 0 {
            return;
        }

        let [loob, loops @ ..] = loops else { unreachable!() };
        let lvalue = &mut loob.scope[index].value;

        self.load_loop_value(index, lvalue, loops);

        if !self[*lvalue].is_lazy_phi() {
            self.unlock(*value);
            let inps = [loob.node, *lvalue, VOID];
            self.unlock(inps[1]);
            let ty = self[inps[1]].ty;
            let phi = self.new_node_nop(ty, Kind::Phi, inps);
            self[phi].lock_rc += 2;
            *value = phi;
            *lvalue = phi;
        } else {
            self.lock(*lvalue);
            self.unlock(*value);
            *value = *lvalue;
        }
    }

    fn check_dominance(&mut self, nd: Nid, min: Nid, check_outputs: bool) {
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
    End,
    If,
    Region,
    Loop,
    Return,
    CInt {
        value: i64,
    },
    Phi,
    Tuple {
        index: u32,
    },
    UnOp {
        op: lexer::TokenKind,
    },
    BinOp {
        op: lexer::TokenKind,
    },
    Call {
        func: ty::Func,
    },
}

impl Kind {
    fn is_pinned(&self) -> bool {
        self.is_cfg() || matches!(self, Kind::Phi)
    }

    fn is_cfg(&self) -> bool {
        matches!(
            self,
            Kind::Start
                | Kind::End
                | Kind::Return
                | Kind::Tuple { .. }
                | Kind::Call { .. }
                | Kind::If
                | Kind::Region
                | Kind::Loop
        )
    }

    fn ends_basic_block(&self) -> bool {
        matches!(self, Kind::Return | Kind::If | Kind::End)
    }

    fn starts_basic_block(&self) -> bool {
        matches!(self, Kind::Start | Kind::End | Kind::Tuple { .. } | Kind::Region | Kind::Loop)
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::CInt { value } => write!(f, "#{value}"),
            Kind::Tuple { index } => write!(f, "tupl[{index}]"),
            Kind::BinOp { op } => write!(f, "{op}"),
            Kind::Call { func, .. } => write!(f, "call {func}"),
            slf => write!(f, "{slf:?}"),
        }
    }
}

#[derive(Debug, Default, Clone)]
//#[repr(align(64))]
struct Node {
    inputs: Vc,
    outputs: Vc,
    kind: Kind,
    color: Color,
    depth: IDomDepth,
    lock_rc: LockRc,
    ty: ty::Id,
    loop_depth: LoopDepth,
    offset: Offset,
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
}

type Offset = u32;
type Size = u32;
type Color = u16;
type LoopDepth = u16;
type CallCount = u16;
type LockRc = u16;
type IDomDepth = u16;

struct Loop {
    node: Nid,
    ctrl: [Nid; 2],
    ctrl_scope: [Vec<Variable>; 2],
    scope: Vec<Variable>,
}

#[derive(Clone, Copy)]
struct Variable {
    id: Ident,
    value: Nid,
}

struct ColorMeta {
    rc: u32,
    depth: LoopDepth,
    loc: Loc,
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    id: ty::Id,
    ret: Option<ty::Id>,

    task_base: usize,

    nodes: Nodes,
    ctrl: Nid,

    loop_depth: LoopDepth,
    colors: Vec<ColorMeta>,
    call_count: u16,
    filled: Vec<Nid>,
    delayed_frees: Vec<Color>,

    loops: Vec<Loop>,
    vars: Vec<Variable>,
    regs: reg::Alloc,
    ret_relocs: Vec<Reloc>,
    relocs: Vec<TypedReloc>,
    jump_relocs: Vec<(Nid, Reloc)>,
    code: Vec<u8>,
}

impl ItemCtx {
    fn next_color(&mut self) -> Color {
        self.colors.push(ColorMeta { rc: 0, depth: self.loop_depth, loc: Default::default() });
        self.colors.len() as _ // leave out 0 (sentinel)
    }

    fn set_next_color(&mut self, node: Nid) -> Color {
        let color = self.next_color();
        self.set_color(node, color);
        color
    }

    fn set_color(&mut self, node: Nid, color: Color) {
        if self.nodes[node].color == color {
            return;
        }
        if self.nodes[node].color != 0 {
            self.colors[self.nodes[node].color as usize - 1].rc -= 1;
        }
        self.nodes[node].color = color;
        self.colors[color as usize - 1].rc += 1;
    }

    fn recolor(&mut self, node: Nid, from: Color, to: Color) {
        if from == to {
            return;
        }

        if self.nodes[node].color != from {
            return;
        }

        self.set_color(node, to);

        for i in 0..self.nodes[node].inputs.len() {
            self.recolor(self.nodes[node].inputs[i], from, to);
        }
    }

    fn check_color_integrity(&self) {
        let node_count = self
            .nodes
            .values
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    Ok(Node {
                        kind: Kind::BinOp { .. }
                            | Kind::Call { .. }
                            | Kind::Phi
                            | Kind::CInt { .. },
                        color: 1..,
                        ..
                    })
                ) || matches!(
                    v,
                    Ok(Node { kind: Kind::Tuple { index: 1.. },
                        color: 1..,
                    inputs, .. }) if inputs.first() == Some(&VOID)
                )
            })
            .count();
        let color_count = self.colors.iter().map(|c| c.rc).sum::<u32>();
        debug_assert_eq!(node_count, color_count as usize);
    }

    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        crate::emit(&mut self.code, instr);
    }

    fn free_loc(&mut self, loc: impl Into<Option<Loc>>) {
        if let Some(loc) = loc.into() {
            self.regs.free(loc.reg);
        }
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Loc {
    reg: reg::Reg,
}

#[derive(Default, Debug, Clone, Copy)]
struct GenCtx {
    loc: Option<Loc>,
}

impl GenCtx {
    pub fn with_loc(self, loc: impl Into<Loc>) -> Self {
        Self { loc: Some(loc.into()) }
    }
}

#[derive(Default)]
struct Pool {
    cis: Vec<ItemCtx>,
}

#[derive(Default)]
pub struct Codegen {
    pub files: Vec<parser::Ast>,
    tasks: Vec<Option<FTask>>,

    tys: Types,
    ci: ItemCtx,
    pool: Pool,
    errors: RefCell<String>,
}

impl Codegen {
    pub fn generate(&mut self) {
        self.find_or_declare(0, 0, None, "main");
        self.make_func_reachable(0);
        self.complete_call_graph_low();
    }

    fn make_func_reachable(&mut self, func: ty::Func) {
        let fuc = &mut self.tys.funcs[func as usize];
        if fuc.offset == u32::MAX {
            fuc.offset = task::id(self.tasks.len() as _);
            self.tasks.push(Some(FTask { file: fuc.file, id: func }));
        }
    }

    fn expr(&mut self, expr: &Expr) -> Option<Nid> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn build_struct(&mut self, fields: &[CommentOr<StructField>]) -> ty::Struct {
        let fields = fields
            .iter()
            .filter_map(CommentOr::or)
            .map(|sf| Field { name: sf.name.into(), ty: self.ty(&sf.ty) })
            .collect();
        self.tys.structs.push(Struct { fields });
        self.tys.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, ctx: Ctx) -> Option<Nid> {
        let msg = "i know nothing about this name gal which is vired \
                            because we parsed succesfully";
        match *expr {
            Expr::Comment { .. } => Some(VOID),
            Expr::Ident { pos, id, .. } => {
                let Some(index) = self.ci.vars.iter().position(|v| v.id == id) else {
                    self.report(pos, msg);
                    return Some(NEVER);
                };

                self.ci.nodes.load_loop_value(
                    index,
                    &mut self.ci.vars[index].value,
                    &mut self.ci.loops,
                );

                Some(self.ci.vars[index].value)
            }
            Expr::BinOp { left: &Expr::Ident { id, .. }, op: TokenKind::Decl, right } => {
                let value = self.expr(right)?;
                self.ci.nodes.lock(value);
                self.ci.vars.push(Variable { id, value });
                Some(VOID)
            }
            Expr::BinOp { left: &Expr::Ident { id, pos, .. }, op: TokenKind::Assign, right } => {
                let value = self.expr(right)?;
                self.ci.nodes.lock(value);

                let Some(var) = self.ci.vars.iter_mut().find(|v| v.id == id) else {
                    self.report(pos, msg);
                    return Some(NEVER);
                };

                let prev = std::mem::replace(&mut var.value, value);
                self.ci.nodes.unlock_remove(prev);
                Some(VOID)
            }
            Expr::BinOp { left, op, right } => {
                let lhs = self.expr_ctx(left, ctx)?;
                self.ci.nodes.lock(lhs);
                let rhs = self.expr_ctx(right, Ctx::default().with_ty(self.tof(lhs)));
                self.ci.nodes.unlock(lhs);
                let rhs = rhs?;
                let ty = self.assert_ty(
                    left.pos(),
                    self.tof(rhs),
                    self.tof(lhs),
                    false,
                    "right operand",
                );
                let inps = [VOID, lhs, rhs];
                Some(self.ci.nodes.new_node(ty::bin_ret(ty, op), Kind::BinOp { op }, inps))
            }
            Expr::UnOp { pos, op, val } => {
                let val = self.expr_ctx(val, ctx)?;
                if !self.tof(val).is_integer() {
                    self.report(
                        pos,
                        format_args!("cant negate '{}'", self.ty_display(self.tof(val))),
                    );
                }
                Some(self.ci.nodes.new_node(self.tof(val), Kind::UnOp { op }, [VOID, val]))
            }
            Expr::If { cond, then, else_, .. } => {
                let cond = self.expr_ctx(cond, Ctx::default().with_ty(ty::BOOL))?;

                let if_node = self.ci.nodes.new_node(ty::VOID, Kind::If, [self.ci.ctrl, cond]);

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
                        return Some(VOID);
                    }
                }

                let mut else_scope = self.ci.vars.clone();
                for &el in &self.ci.vars {
                    self.ci.nodes.lock(el.value);
                }

                self.ci.ctrl =
                    self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [if_node]);
                let lcntrl = self.expr(then).map_or(Nid::MAX, |_| self.ci.ctrl);

                let mut then_scope = std::mem::replace(&mut self.ci.vars, else_scope);
                self.ci.ctrl =
                    self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 1 }, [if_node]);
                let rcntrl = if let Some(else_) = else_ {
                    self.expr(else_).map_or(Nid::MAX, |_| self.ci.ctrl)
                } else {
                    self.ci.ctrl
                };

                if lcntrl == Nid::MAX && rcntrl == Nid::MAX {
                    for then_var in then_scope {
                        self.ci.nodes.unlock_remove(then_var.value);
                    }
                    return None;
                } else if lcntrl == Nid::MAX {
                    for then_var in then_scope {
                        self.ci.nodes.unlock_remove(then_var.value);
                    }
                    return Some(VOID);
                } else if rcntrl == Nid::MAX {
                    for else_var in &self.ci.vars {
                        self.ci.nodes.unlock_remove(else_var.value);
                    }
                    self.ci.vars = then_scope;
                    self.ci.ctrl = lcntrl;
                    return Some(VOID);
                }

                self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Region, [lcntrl, rcntrl]);

                else_scope = std::mem::take(&mut self.ci.vars);

                Self::merge_scopes(
                    &mut self.ci.nodes,
                    &mut self.ci.loops,
                    self.ci.ctrl,
                    &mut else_scope,
                    &mut then_scope,
                    true,
                );

                self.ci.vars = else_scope;

                Some(VOID)
            }
            Expr::Loop { body, .. } => {
                self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Loop, [self.ci.ctrl; 2]);
                self.ci.loops.push(Loop {
                    node: self.ci.ctrl,
                    ctrl: [Nid::MAX; 2],
                    ctrl_scope: std::array::from_fn(|_| vec![]),
                    scope: self.ci.vars.clone(),
                });

                for var in &mut self.ci.vars {
                    var.value = VOID;
                }
                self.ci.nodes[VOID].lock_rc += self.ci.vars.len() as LockRc;

                self.expr(body);

                let Loop { node, ctrl: [mut con, bre], ctrl_scope: [mut cons, mut bres], scope } =
                    self.ci.loops.pop().unwrap();

                if con != Nid::MAX {
                    con = self.ci.nodes.new_node(ty::VOID, Kind::Region, [con, self.ci.ctrl]);
                    Self::merge_scopes(
                        &mut self.ci.nodes,
                        &mut self.ci.loops,
                        con,
                        &mut self.ci.vars,
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

                std::mem::swap(&mut self.ci.vars, &mut bres);

                for ((dest_var, mut scope_var), loop_var) in
                    self.ci.vars.iter_mut().zip(scope).zip(bres)
                {
                    self.ci.nodes.unlock(loop_var.value);

                    if loop_var.value != VOID {
                        self.ci.nodes.unlock(scope_var.value);
                        if loop_var.value != scope_var.value {
                            scope_var.value =
                                self.ci.nodes.modify_input(scope_var.value, 2, loop_var.value);
                            self.ci.nodes.lock(scope_var.value);
                        } else {
                            let phi = &self.ci.nodes[scope_var.value];
                            debug_assert_eq!(phi.kind, Kind::Phi);
                            debug_assert_eq!(phi.inputs[2], VOID);
                            let prev = phi.inputs[1];
                            self.ci.nodes.replace(scope_var.value, prev);
                            scope_var.value = prev;
                            self.ci.nodes.lock(prev);
                        }
                    }

                    if dest_var.value == VOID {
                        self.ci.nodes.unlock(dest_var.value);
                        dest_var.value = scope_var.value;
                        self.ci.nodes.lock(dest_var.value);
                    }

                    self.ci.nodes.unlock_remove(scope_var.value);
                }

                self.ci.nodes.unlock(self.ci.ctrl);

                Some(VOID)
            }
            Expr::Break { pos } => self.jump_to(pos, 1),
            Expr::Continue { pos } => self.jump_to(pos, 0),
            Expr::Call { func: &Expr::Ident { pos, id, name, .. }, args, .. } => {
                self.ci.call_count += 1;
                let func = self.find_or_declare(pos, self.ci.file, Some(id), name);
                let ty::Kind::Func(func) = func else {
                    self.report(
                        pos,
                        format_args!(
                            "compiler cant (yet) call '{}'",
                            self.ty_display(func.compress())
                        ),
                    );
                    return Some(NEVER);
                };

                self.make_func_reachable(func);

                let fuc = &self.tys.funcs[func as usize];
                let sig = fuc.sig.expect("TODO: generic functions");
                let ast = self.files[fuc.file as usize].clone();
                let Expr::BinOp { right: &Expr::Closure { args: cargs, .. }, .. } =
                    fuc.expr.get(&ast).unwrap()
                else {
                    unreachable!()
                };

                self.assert_report(
                    args.len() == cargs.len(),
                    pos,
                    format_args!(
                        "expected {} function argumenr{}, got {}",
                        cargs.len(),
                        if cargs.len() == 1 { "" } else { "s" },
                        args.len()
                    ),
                );

                let mut inps = Vc::from([self.ci.ctrl]);
                for ((arg, carg), tyx) in args.iter().zip(cargs).zip(sig.args.range()) {
                    let ty = self.tys.args[tyx];
                    if self.tys.size_of(ty) == 0 {
                        continue;
                    }
                    let value = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    _ = self.assert_ty(
                        arg.pos(),
                        self.tof(value),
                        ty,
                        true,
                        format_args!("argument {}", carg.name),
                    );
                    inps.push(value);
                }
                self.ci.ctrl = self.ci.nodes.new_node(sig.ret, Kind::Call { func }, inps);

                Some(self.ci.ctrl)
            }
            Expr::Return { pos, val } => {
                let value = if let Some(val) = val {
                    self.expr_ctx(val, Ctx { ty: self.ci.ret })?
                } else {
                    VOID
                };

                let inps = [self.ci.ctrl, value];

                let out = &mut String::new();
                self.report_log_to(pos, "returning here", out);
                self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Return, inps);

                self.ci.nodes[NEVER].inputs.push(self.ci.ctrl);
                self.ci.nodes[self.ci.ctrl].outputs.push(NEVER);

                let expected = *self.ci.ret.get_or_insert(self.tof(value));
                _ = self.assert_ty(pos, self.tof(value), expected, true, "return value");

                None
            }
            Expr::Block { stmts, .. } => {
                let base = self.ci.vars.len();

                let mut ret = Some(VOID);
                for stmt in stmts {
                    ret = ret.and(self.expr(stmt));
                    if let Some(id) = ret {
                        _ = self.assert_ty(
                            stmt.pos(),
                            self.tof(id),
                            ty::VOID.into(),
                            true,
                            "statement",
                        );
                    } else {
                        break;
                    }
                }

                self.ci.nodes.lock(self.ci.ctrl);
                for var in self.ci.vars.drain(base..) {
                    self.ci.nodes.unlock_remove(var.value);
                }
                self.ci.nodes.unlock(self.ci.ctrl);

                ret
            }
            Expr::Number { value, .. } => Some(self.ci.nodes.new_node(
                ctx.ty.filter(|ty| ty.is_integer() || ty.is_pointer()).unwrap_or(ty::INT.into()),
                Kind::CInt { value },
                [VOID],
            )),
            ref e => {
                self.report_unhandled_ast(e, "bruh");
                Some(NEVER)
            }
        }
    }

    fn jump_to(&mut self, pos: Pos, id: usize) -> Option<Nid> {
        let Some(mut loob) = self.ci.loops.last_mut() else {
            self.report(pos, "break outside a loop");
            return None;
        };

        if loob.ctrl[id] == Nid::MAX {
            loob.ctrl[id] = self.ci.ctrl;
            loob.ctrl_scope[id] = self.ci.vars[..loob.scope.len()].to_owned();
            for v in &loob.ctrl_scope[id] {
                self.ci.nodes.lock(v.value)
            }
        } else {
            let reg = self.ci.nodes.new_node(ty::VOID, Kind::Region, [self.ci.ctrl, loob.ctrl[id]]);
            let mut scope = std::mem::take(&mut loob.ctrl_scope[id]);

            Self::merge_scopes(
                &mut self.ci.nodes,
                &mut self.ci.loops,
                reg,
                &mut scope,
                &mut self.ci.vars,
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
        to: &mut [Variable],
        from: &mut [Variable],
        drop_from: bool,
    ) {
        for (i, (else_var, then_var)) in to.iter_mut().zip(from).enumerate() {
            if else_var.value != then_var.value {
                nodes.load_loop_value(i, &mut then_var.value, loops);
                nodes.load_loop_value(i, &mut else_var.value, loops);
                if else_var.value != then_var.value {
                    let ty = nodes[else_var.value].ty;
                    debug_assert_eq!(ty, nodes[then_var.value].ty, "TODO: typecheck properly");

                    let inps = [ctrl, then_var.value, else_var.value];
                    nodes.unlock(else_var.value);
                    else_var.value = nodes.new_node(ty, Kind::Phi, inps);
                    nodes.lock(else_var.value);
                }
            }

            if drop_from {
                nodes.unlock_remove(then_var.value);
            }
        }
    }

    #[inline(always)]
    fn tof(&self, id: Nid) -> ty::Id {
        self.ci.nodes[id].ty
    }

    fn complete_call_graph(&mut self) {
        self.complete_call_graph_low();
    }

    fn complete_call_graph_low(&mut self) {
        while self.ci.task_base < self.tasks.len()
            && let Some(task_slot) = self.tasks.pop()
        {
            let Some(task) = task_slot else { continue };
            self.emit_func(task);
        }
    }

    fn emit_func(&mut self, FTask { file, id }: FTask) {
        let func = &mut self.tys.funcs[id as usize];
        func.offset = u32::MAX - 1;
        debug_assert!(func.file == file);
        let sig = func.sig.unwrap();
        let ast = self.files[file as usize].clone();
        let expr = func.expr.get(&ast).unwrap();

        let repl = ItemCtx {
            file,
            id: ty::Kind::Func(id).compress(),
            ret: Some(sig.ret),
            ..self.pool.cis.pop().unwrap_or_default()
        };
        let prev_ci = std::mem::replace(&mut self.ci, repl);

        let start = self.ci.nodes.new_node(ty::VOID, Kind::Start, []);
        debug_assert_eq!(start, VOID);
        let end = self.ci.nodes.new_node(ty::NEVER, Kind::End, []);
        debug_assert_eq!(end, NEVER);
        self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [VOID]);

        let Expr::BinOp {
            left: Expr::Ident { .. },
            op: TokenKind::Decl,
            right: &Expr::Closure { body, args, .. },
        } = expr
        else {
            unreachable!("{expr}")
        };

        let mut sig_args = sig.args.range();
        for (arg, index) in args.iter().zip(1u32..) {
            let ty = self.tys.args[sig_args.next().unwrap()];
            let value = self.ci.nodes.new_node(ty, Kind::Tuple { index }, [VOID]);
            self.ci.nodes.lock(value);
            let sym = parser::find_symbol(&ast.symbols, arg.id);
            assert!(sym.flags & idfl::COMPTIME == 0, "TODO");
            self.ci.vars.push(Variable { id: arg.id, value });
        }

        let orig_vars = self.ci.vars.clone();

        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        for var in self.ci.vars.drain(..) {
            self.ci.nodes.unlock(var.value);
        }

        if self.errors.borrow().is_empty() {
            self.gcm();

            #[cfg(debug_assertions)]
            {
                self.ci.nodes.check_final_integrity();
            }

            '_open_function: {
                self.ci.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
                self.ci.emit(instrs::st(reg::RET_ADDR, reg::STACK_PTR, 0, 0));
            }

            self.ci.regs.init();

            self.ci.nodes.basic_blocks();

            self.ci.nodes.visited.clear(self.ci.nodes.values.len());
            self.color_node(NEVER);

            let call_count = self.ci.call_count;
            let mut parama = self.tys.parama(sig.ret);
            '_color_args: {
                for (var, ti) in orig_vars.iter().zip(sig.args.range()) {
                    let ty = self.tys.args[ti];
                    let mut col = self.ci.nodes[var.value].color;
                    if col == 0 {
                        col = self.ci.set_next_color(var.value);
                    }
                    match self.tys.size_of(ty) {
                        0 => continue,
                        1..=8 => {
                            let loc = Loc { reg: parama.next() };
                            let slot = &mut self.ci.colors[col as usize - 1].loc;

                            if *slot != Loc::default() {
                                self.emit_pass_low(loc, var.value);
                            } else if check_no_calls(var.value, &mut self.ci.nodes) {
                                *slot = loc;
                            } else {
                                *slot = Loc { reg: self.ci.regs.allocate(0) };
                                self.emit_pass_low(loc, var.value);
                            }
                        }
                        _ => todo!(),
                    }
                }
            }

            //self.ci.nodes.graphviz();

            #[cfg(debug_assertions)]
            {
                self.ci.check_color_integrity();
            }

            self.ci.vars = orig_vars;
            self.ci.call_count = call_count;
            self.ci.nodes.visited.clear(self.ci.nodes.values.len());
            self.emit_node(VOID, VOID);
            self.ci.vars.clear();

            if let Some(last_ret) = self.ci.ret_relocs.last()
                && last_ret.offset as usize == self.ci.code.len() - 5
            {
                self.ci.code.truncate(self.ci.code.len() - 5);
                self.ci.ret_relocs.pop();
            }

            // FIXME: maybe do this incrementally
            for (nd, rel) in self.ci.jump_relocs.drain(..) {
                let offset = self.ci.nodes[nd].offset;
                rel.apply_jump(&mut self.ci.code, offset, 0);
            }

            let end = self.ci.code.len();
            for ret_rel in self.ci.ret_relocs.drain(..) {
                ret_rel.apply_jump(&mut self.ci.code, end as _, 0);
            }

            '_close_function: {
                let pushed = self.ci.regs.pushed_size() as i64;
                let stack = 0;

                write_reloc(&mut self.ci.code, 3, -(pushed + stack), 8);
                write_reloc(&mut self.ci.code, 3 + 8 + 3, stack, 8);
                write_reloc(&mut self.ci.code, 3 + 8 + 3 + 8, pushed, 2);

                self.ci.emit(instrs::ld(reg::RET_ADDR, reg::STACK_PTR, stack as _, pushed as _));
                self.ci.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, (pushed + stack) as _));
                self.ci.emit(instrs::jala(reg::ZERO, reg::RET_ADDR, 0));
            }
        }

        self.tys.funcs[id as usize].code.append(&mut self.ci.code);
        self.tys.funcs[id as usize].relocs.append(&mut self.ci.relocs);
        self.ci.nodes.clear();
        self.ci.colors.clear();
        self.ci.filled.clear();
        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
    }

    fn color_node(&mut self, ctrl: Nid) -> Option<Color> {
        if !self.ci.nodes.visited.set(ctrl) {
            return None;
        }

        let node = self.ci.nodes[ctrl].clone();
        match node.kind {
            Kind::Start => None,
            Kind::End => {
                for &i in node.inputs.iter() {
                    self.color_node(i);
                }
                None
            }
            Kind::If => {
                let &[pred, _] = node.inputs.as_slice() else { unreachable!() };
                self.color_node(pred);
                None
            }
            Kind::Region => {
                for o in node.outputs {
                    self.color_node(o);
                }
                self.color_node(node.inputs[0]);
                self.color_node(node.inputs[1]);
                None
            }
            Kind::Loop => {
                self.color_node(node.inputs[1]);
                for o in node.outputs {
                    self.color_node(o);
                }
                self.color_node(node.inputs[0]);
                None
            }
            Kind::Return => {
                if node.inputs[1] != VOID {
                    let col = self.ci.set_next_color(node.inputs[1]);
                    if check_no_calls(node.inputs[1], &mut self.ci.nodes) {
                        self.ci.colors[col as usize - 1].loc = Loc { reg: 1 };
                    }
                }

                self.color_node(node.inputs[0]);
                None
            }
            Kind::CInt { .. } => {
                if node.color == 0 && node.lock_rc as usize != node.outputs.len() {
                    self.ci.set_next_color(ctrl);
                }

                None
            }
            Kind::Phi => {
                if node.color == 0 {
                    self.ci.set_next_color(ctrl);
                }

                fn is_tangled_phy(
                    target: Nid,
                    phi: Nid,
                    min_idepth: IDomDepth,
                    nodes: &mut Nodes,
                ) -> bool {
                    for i in nodes[target].inputs.clone() {
                        if nodes.is_cfg(i) {
                            continue;
                        }

                        if idepth(nodes, i) < min_idepth {
                            continue;
                        }

                        if i == phi {
                            continue;
                        }

                        if nodes[i].kind == Kind::Phi && nodes[i].inputs[0] == nodes[phi].inputs[0]
                        {
                            return true;
                        }

                        if is_tangled_phy(i, phi, min_idepth, nodes) {
                            return true;
                        }
                    }

                    false
                }

                let is_region = self.ci.nodes[node.inputs[0]].kind == Kind::Region;
                let is_independent_phy = is_region
                    || !is_tangled_phy(
                        ctrl,
                        ctrl,
                        idepth(&mut self.ci.nodes, node.inputs[0]),
                        &mut self.ci.nodes,
                    );
                let &[_, l, r] = node.inputs.as_slice() else { unreachable!() };
                if is_independent_phy && is_last_branch_use(l, ctrl, &mut self.ci.nodes) {
                    self.ci.set_color(l, self.ci.nodes[ctrl].color);
                }
                if is_independent_phy && is_last_branch_use(r, ctrl, &mut self.ci.nodes) {
                    self.ci.set_color(r, self.ci.nodes[ctrl].color);
                }

                None
            }
            Kind::Tuple { index } => {
                if (self.ci.nodes[node.inputs[0]].kind == Kind::Start && index == 0)
                    || (self.ci.nodes[node.inputs[0]].kind == Kind::If && index < 2)
                {
                    for o in node.outputs {
                        self.color_node(o);
                    }
                    self.color_node(node.inputs[0]);
                }

                None
            }
            Kind::BinOp { .. } => {
                let can_optimize = matches!(node.kind, Kind::BinOp { op } if op.cond_op(false).is_some())
                    && matches!(node.outputs.as_slice(), &[stmt] if self.ci.nodes[stmt].kind == Kind::If);

                if node.color == 0 && !can_optimize {
                    self.ci.set_next_color(ctrl);
                }

                let &[_, lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                if !self.ci.nodes.is_const(rhs) || can_optimize {
                    self.color_node(rhs);
                } else {
                    self.ci.nodes[rhs].lock_rc += 1;
                }

                self.color_node(lhs);

                None
            }
            Kind::UnOp { .. } => {
                if node.color == 0 {
                    self.ci.set_next_color(ctrl);
                }
                self.color_node(node.inputs[1]);
                None
            }
            Kind::Call { func } => {
                if node.color == 0 {
                    self.ci.set_next_color(ctrl);
                }

                let func = self.tys.funcs[func as usize].sig.unwrap();
                let mut parama = self.tys.parama(func.ret);
                for (&i, ti) in node.inputs[1..].iter().zip(func.args.range()) {
                    let ty = self.tys.args[ti];
                    match self.tys.size_of(ty) {
                        0 => continue,
                        1..=8 => {
                            let col = self.ci.set_next_color(i);
                            let loc = Loc { reg: parama.next() };
                            if check_no_calls(i, &mut self.ci.nodes) {
                                self.ci.colors[col as usize - 1].loc = loc;
                            }
                        }
                        _ => todo!(),
                    }
                }

                for o in node.outputs {
                    self.color_node(o);
                }

                self.color_node(node.inputs[0]);
                None
            }
        }
    }

    fn emit_node(&mut self, ctrl: Nid, prev: Nid) -> Option<Nid> {
        if matches!(self.ci.nodes[ctrl].kind, Kind::Region | Kind::Loop) {
            let node = self.ci.nodes[ctrl].clone();

            let idx = 1 + node.inputs.iter().position(|&i| i == prev).unwrap();

            for ph in node.outputs {
                if self.ci.nodes[ph].kind != Kind::Phi {
                    continue;
                }

                self.emit_pass(self.ci.nodes[ph].inputs[idx], ph);
            }

            match (self.ci.nodes[ctrl].kind, self.ci.nodes.visited.set(ctrl)) {
                (Kind::Region, true) | (Kind::Loop, false) => {
                    self.ci.jump_relocs.push((ctrl, Reloc::new(self.ci.code.len(), 1, 4)));
                    self.ci.emit(instrs::jmp(0));
                    return None;
                }
                _ => {}
            }
        } else if !self.ci.nodes.visited.set(ctrl) {
            return None;
        }

        let node = self.ci.nodes[ctrl].clone();
        match node.kind {
            Kind::Start => self.emit_node(node.outputs[0], VOID),
            Kind::End => None,
            Kind::If => {
                let &[_, cond] = node.inputs.as_slice() else { unreachable!() };
                let &[mut then, mut else_] = node.outputs.as_slice() else { unreachable!() };

                if let Kind::BinOp { op } = self.ci.nodes[cond].kind
                    && let Some((op, swapped)) = op.cond_op(node.ty.is_signed())
                {
                    self.ci.nodes[cond].offset = self.ci.code.len() as _;
                    if swapped {
                        std::mem::swap(&mut then, &mut else_);
                    }
                    let &[_, lhs, rhs] = self.ci.nodes[cond].inputs.as_slice() else {
                        unreachable!()
                    };
                    self.ci.emit(op(node_loc!(self, lhs).reg, node_loc!(self, rhs).reg, 0));
                } else {
                    todo!()
                }

                self.emit_node(then, ctrl);
                let jump = self.ci.code.len() as i64 - self.ci.nodes[cond].offset as i64;
                self.emit_node(else_, ctrl);
                write_reloc(&mut self.ci.code, self.ci.nodes[cond].offset as usize + 3, jump, 2);

                None
            }
            Kind::Region | Kind::Loop => {
                self.ci.nodes[ctrl].offset = self.ci.code.len() as _;
                for o in node.outputs.into_iter().rev() {
                    self.emit_node(o, ctrl);
                }
                None
            }
            Kind::Return => {
                if node.inputs[1] != VOID {
                    let loc = Loc { reg: 1 };
                    if node_loc!(self, node.inputs[1]) != loc {
                        self.ci.emit(instrs::cp(loc.reg, node_loc!(self, node.inputs[1]).reg));
                    }
                }
                self.ci.ret_relocs.push(Reloc::new(self.ci.code.len(), 1, 4));
                self.ci.emit(instrs::jmp(0));
                self.emit_node(node.outputs[0], ctrl);
                None
            }
            Kind::CInt { value } => {
                if node.color != 0 {
                    if node_loc!(self, ctrl) == Loc::default() {
                        node_loc!(self, ctrl) = Loc { reg: self.ci.regs.allocate(0) };
                    }
                    // TODO: respect size
                    self.ci.emit(instrs::li64(node_loc!(self, ctrl).reg, value as _));
                }
                None
            }
            Kind::Phi => None,
            Kind::Tuple { index } => {
                if (self.ci.nodes[node.inputs[0]].kind == Kind::Start && index == 0)
                    || (self.ci.nodes[node.inputs[0]].kind == Kind::If && index < 2)
                {
                    for o in node.outputs.into_iter().rev() {
                        self.emit_node(o, ctrl);
                    }
                } else {
                    todo!();
                }

                None
            }
            Kind::BinOp { op } => {
                if node.color != 0 {
                    let &[_, lhs, rhs] = node.inputs.as_slice() else { unreachable!() };

                    self.lazy_init(ctrl);
                    if self.ci.nodes[rhs].color == 0
                        && let Kind::CInt { value } = self.ci.nodes[rhs].kind
                        && let Some(op) =
                            op.imm_binop(node.ty.is_signed(), self.tys.size_of(node.ty))
                    {
                        self.ci.emit(op(
                            node_loc!(self, ctrl).reg,
                            node_loc!(self, lhs).reg,
                            value as _,
                        ));
                    } else if let Some(op) =
                        op.binop(node.ty.is_signed(), self.tys.size_of(node.ty))
                    {
                        self.ci.emit(op(
                            node_loc!(self, ctrl).reg,
                            node_loc!(self, lhs).reg,
                            node_loc!(self, rhs).reg,
                        ));
                    } else {
                        todo!()
                    }
                }
                None
            }
            Kind::UnOp { op } => {
                if node.color != 0 {
                    let op = op.unop().expect("TODO: unsupported unary operator");
                    self.lazy_init(ctrl);
                    self.ci
                        .emit(op(node_loc!(self, ctrl).reg, node_loc!(self, node.inputs[1]).reg));
                }
                None
            }
            Kind::Call { func } => {
                let fuc = self.tys.funcs[func as usize].sig.unwrap();
                let mut parama = self.tys.parama(fuc.ret);
                for (&i, ti) in node.inputs[1..].iter().zip(fuc.args.range()) {
                    let ty = self.tys.args[ti];
                    match self.tys.size_of(ty) {
                        0 => continue,
                        1..=8 => {
                            let loc = Loc { reg: parama.next() };
                            if node_loc!(self, i) != loc {
                                self.ci.emit(instrs::cp(loc.reg, node_loc!(self, i).reg));
                            }
                        }
                        _ => todo!(),
                    }
                }

                let reloc = Reloc::new(self.ci.code.len(), 3, 4);
                self.ci.relocs.push(TypedReloc { target: ty::Kind::Func(func).compress(), reloc });
                self.ci.emit(jal(reg::RET_ADDR, reg::ZERO, 0));
                self.emit_pass_low(Loc { reg: 1 }, ctrl);
                for o in node.outputs.into_iter().rev() {
                    if self.ci.nodes[o].inputs[0] == ctrl {
                        self.emit_node(o, ctrl);
                    }
                }
                None
            }
        }
    }

    // TODO: sometimes its better to do this in bulk
    fn ty(&mut self, expr: &Expr) -> ty::Id {
        match *expr {
            Expr::Ident { id, .. } if ident::is_null(id) => id.into(),
            ref e => {
                self.report_unhandled_ast(e, "type");
                ty::NEVER.into()
            }
        }
    }

    fn find_or_declare(
        &mut self,
        pos: Pos,
        file: FileId,
        name: Option<Ident>,
        lit_name: &str,
    ) -> ty::Kind {
        log::trc!("find_or_declare: {lit_name} {file}");

        let f = self.files[file as usize].clone();
        let Some((expr, ident)) = f.find_decl(name.ok_or(lit_name)) else {
            match name.ok_or(lit_name) {
                Ok(name) => {
                    let name = self.cfile().ident_str(name);
                    self.report(pos, format_args!("idk indentifier: {name}"))
                }
                Err("main") => self.report(
                    pos,
                    format_args!(
                        "missing main function in '{}', compiler can't \
                        emmit libraries since such concept is not defined",
                        f.path
                    ),
                ),
                Err(name) => self.report(pos, format_args!("idk indentifier: {name}")),
            }
            return ty::Kind::Builtin(ty::NEVER);
        };

        if let Some(existing) = self.tys.syms.get(&SymKey { file, ident }) {
            if let ty::Kind::Func(id) = existing.expand()
                && let func = &mut self.tys.funcs[id as usize]
                && let Err(idx) = task::unpack(func.offset)
                && idx < self.tasks.len()
            {
                func.offset = task::id(self.tasks.len());
                let task = self.tasks[idx].take();
                self.tasks.push(task);
            }
            return existing.expand();
        }

        let prev_file = std::mem::replace(&mut self.ci.file, file);
        let sym = match expr {
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: &Expr::Closure { pos, args, ret, .. },
            } => {
                let func = Func {
                    file,
                    sig: '_b: {
                        let arg_base = self.tys.args.len();
                        for arg in args {
                            let sym = parser::find_symbol(&f.symbols, arg.id);
                            assert!(sym.flags & idfl::COMPTIME == 0, "TODO");
                            let ty = self.ty(&arg.ty);
                            self.tys.args.push(ty);
                        }

                        let Some(args) = self.pack_args(arg_base) else {
                            self.fatal_report(
                                pos,
                                "you cant be serious, using more the 31 arguments in a function",
                            );
                        };
                        let ret = self.ty(ret);

                        Some(Sig { args, ret })
                    },
                    expr: {
                        let refr = ExprRef::new(expr);
                        debug_assert!(refr.get(&f).is_some());
                        refr
                    },
                    ..Default::default()
                };

                let id = self.tys.funcs.len() as _;
                self.tys.funcs.push(func);

                ty::Kind::Func(id)
            }
            Expr::BinOp {
                left: &Expr::Ident { .. },
                op: TokenKind::Decl,
                right: Expr::Struct { fields, .. },
            } => ty::Kind::Struct(self.build_struct(fields)),
            Expr::BinOp { .. } => {
                todo!()
            }
            e => unimplemented!("{e:#?}"),
        };
        self.ci.file = prev_file;
        self.tys.syms.insert(SymKey { ident, file }, sym.compress());
        sym
    }

    fn ty_display(&self, ty: ty::Id) -> ty::Display {
        ty::Display::new(&self.tys, &self.files, ty)
    }

    #[must_use]
    #[track_caller]
    fn assert_ty(
        &self,
        pos: Pos,
        ty: ty::Id,
        expected: ty::Id,
        preserve_expected: bool,
        hint: impl fmt::Display,
    ) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected)
            && (!preserve_expected || res == expected)
        {
            res
        } else {
            let ty = self.ty_display(ty);
            let expected = self.ty_display(expected);
            self.report(pos, format_args!("expected {hint} to be of type {expected}, got {ty}"));
            ty::NEVER.into()
        }
    }

    fn report_log(&self, pos: Pos, msg: impl std::fmt::Display) {
        let mut buf = self.errors.borrow_mut();
        self.report_log_to(pos, msg, &mut *buf);
    }

    fn report_log_to(&self, pos: Pos, msg: impl std::fmt::Display, out: &mut impl std::fmt::Write) {
        self.cfile().report_to(pos, msg, out);
    }

    #[track_caller]
    fn assert_report(&self, cond: bool, pos: Pos, msg: impl std::fmt::Display) {
        if !cond {
            self.report(pos, msg);
        }
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl std::fmt::Display) {
        self.report_log(pos, msg);
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: &str) {
        self.report(
            ast.pos(),
            format_args!(
                "compiler does not (yet) know how to handle ({hint}):\n\
                {ast:}\n\
                info for weak people:\n\
                {ast:#?}"
            ),
        );
    }

    fn cfile(&self) -> &parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn pack_args(&mut self, arg_base: usize) -> Option<ty::Tuple> {
        let needle = &self.tys.args[arg_base..];
        if needle.is_empty() {
            return Some(ty::Tuple::empty());
        }
        let len = needle.len();
        // FIXME: maybe later when this becomes a bottleneck we use more
        // efficient search (SIMD?, indexing?)
        let sp = self.tys.args.windows(needle.len()).position(|val| val == needle).unwrap();
        self.tys.args.truncate((sp + needle.len()).max(arg_base));
        ty::Tuple::new(sp, len)
    }

    fn emit_pass(&mut self, src: Nid, dst: Nid) {
        if self.ci.nodes[src].color == self.ci.nodes[dst].color {
            return;
        }
        self.emit_pass_low(node_loc!(self, src), dst);
    }

    fn emit_pass_low(&mut self, src: Loc, dst: Nid) {
        let loc = &mut node_loc!(self, dst);
        if src != *loc {
            if *loc == Loc::default() {
                let reg = self.ci.regs.allocate(0);
                *loc = Loc { reg };
            }
            let inst = instrs::cp(loc.reg, src.reg);
            self.ci.emit(inst);
        }
    }

    fn lazy_init(&mut self, expr: Nid) -> bool {
        let loc = &mut node_loc!(self, expr);
        if *loc == Loc::default() {
            let reg = self.ci.regs.allocate(0);
            *loc = Loc { reg };
            return true;
        }
        false
    }

    fn fatal_report(&self, pos: Pos, msg: impl Display) -> ! {
        self.report(pos, msg);
        eprintln!("{}", self.errors.borrow());
        std::process::exit(1);
    }

    fn gcm(&mut self) {
        self.ci.nodes.visited.clear(self.ci.nodes.values.len());
        push_up(&mut self.ci.nodes, NEVER);
        // TODO: handle infinte loops
        self.ci.nodes.visited.clear(self.ci.nodes.values.len());
        push_down(&mut self.ci.nodes, VOID);
    }
}

fn loop_depth(target: Nid, nodes: &mut Nodes) -> LoopDepth {
    if nodes[target].loop_depth != 0 {
        return nodes[target].loop_depth;
    }

    nodes[target].loop_depth = match nodes[target].kind {
        Kind::Tuple { .. } | Kind::Call { .. } | Kind::Return | Kind::If => {
            loop_depth(nodes[target].inputs[0], nodes)
        }
        Kind::Region => {
            let l = loop_depth(nodes[target].inputs[0], nodes);
            let r = loop_depth(nodes[target].inputs[1], nodes);
            debug_assert_eq!(r, l);
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
                if let Kind::Tuple { index } = nodes[cursor].kind
                    && nodes[next].kind == Kind::If
                {
                    let other = *nodes[next]
                        .outputs
                        .iter()
                        .find(
                            |&&n| matches!(nodes[n].kind, Kind::Tuple { index: oi } if index != oi),
                        )
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
        Kind::CInt { .. } | Kind::Phi | Kind::BinOp { .. } | Kind::UnOp { .. } => {
            unreachable!()
        }
    };

    if target == 19 {
        //panic!("{}", nodes[target].loop_depth);
    }

    nodes[target].loop_depth
}

fn better(nodes: &mut Nodes, is: Nid, then: Nid) -> bool {
    loop_depth(is, nodes) < loop_depth(then, nodes)
        || idepth(nodes, is) > idepth(nodes, then)
        || nodes[then].kind == Kind::If
}

fn idepth(nodes: &mut Nodes, target: Nid) -> IDomDepth {
    if target == VOID {
        return 0;
    }
    if nodes[target].depth == 0 {
        nodes[target].depth = match nodes[target].kind {
            Kind::End | Kind::Start => unreachable!(),
            Kind::Loop
            | Kind::CInt { .. }
            | Kind::BinOp { .. }
            | Kind::UnOp { .. }
            | Kind::Call { .. }
            | Kind::Phi
            | Kind::Tuple { .. }
            | Kind::Return
            | Kind::If => idepth(nodes, nodes[target].inputs[0]),
            Kind::Region => {
                idepth(nodes, nodes[target].inputs[0]).max(idepth(nodes, nodes[target].inputs[1]))
            }
        } + 1;
    }
    nodes[target].depth
}

fn push_up(nodes: &mut Nodes, node: Nid) {
    if !nodes.visited.set(node) {
        return;
    }

    if nodes[node].kind.is_pinned() {
        for i in 0..nodes[node].inputs.len() {
            let i = nodes[node].inputs[i];
            push_up(nodes, i);
        }
    } else {
        let mut max = VOID;
        for i in 0..nodes[node].inputs.len() {
            let i = nodes[node].inputs[i];
            let is_call = matches!(nodes[i].kind, Kind::Call { .. });
            if nodes.is_cfg(i) && !is_call {
                continue;
            }
            push_up(nodes, i);
            if idepth(nodes, i) > idepth(nodes, max) {
                max = if is_call { i } else { idom(nodes, i) };
            }
        }

        #[cfg(debug_assertions)]
        {
            nodes.check_dominance(node, max, false);
        }

        if max == VOID {
            return;
        }

        let index = nodes[0].outputs.iter().position(|&p| p == node).unwrap();
        nodes[0].outputs.remove(index);
        nodes[node].inputs[0] = max;
        debug_assert!(
            !nodes[max].outputs.contains(&node) || matches!(nodes[max].kind, Kind::Call { .. }),
            "{node} {:?} {max} {:?}",
            nodes[node],
            nodes[max]
        );
        nodes[max].outputs.push(node);
    }
}

fn push_down(nodes: &mut Nodes, node: Nid) {
    if !nodes.visited.set(node) {
        return;
    }

    // TODO: handle memory nodes first

    if nodes[node].kind.is_pinned() {
        // TODO: use buffer to avoid allocation or better yet queue the location changes
        for i in nodes[node].outputs.clone() {
            push_down(nodes, i);
        }
    } else {
        let mut min = None::<Nid>;
        for i in 0..nodes[node].outputs.len() {
            let i = nodes[node].outputs[i];
            push_down(nodes, i);
            let i = use_block(node, i, nodes);
            min = min.map(|m| common_dom(i, m, nodes)).or(Some(i));
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

        #[cfg(debug_assertions)]
        {
            nodes.check_dominance(node, min, true);
        }

        let prev = nodes[node].inputs[0];
        if min != prev {
            debug_assert!(idepth(nodes, min) > idepth(nodes, prev));
            let index = nodes[prev].outputs.iter().position(|&p| p == node).unwrap();
            nodes[prev].outputs.remove(index);
            nodes[node].inputs[0] = min;
            nodes[min].outputs.push(node);
        }
    }
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
        Kind::Loop
        | Kind::CInt { .. }
        | Kind::BinOp { .. }
        | Kind::UnOp { .. }
        | Kind::Call { .. }
        | Kind::Phi
        | Kind::Tuple { .. }
        | Kind::Return
        | Kind::If => nodes[target].inputs[0],
        Kind::Region => {
            let &[lcfg, rcfg] = nodes[target].inputs.as_slice() else { unreachable!() };
            common_dom(lcfg, rcfg, nodes)
        }
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

fn scan_idom(
    nodes: &mut Nodes,
    target: Nid,
    pred: &mut impl FnMut(&mut Nodes, Nid) -> bool,
) -> Option<Nid> {
    if !pred(nodes, target) {
        return None;
    }
    Some(match nodes[target].kind {
        Kind::Start => VOID,
        Kind::End => unreachable!(),
        Kind::Loop
        | Kind::CInt { .. }
        | Kind::BinOp { .. }
        | Kind::UnOp { .. }
        | Kind::Call { .. }
        | Kind::Phi
        | Kind::Tuple { .. }
        | Kind::Return
        | Kind::If => nodes[target].inputs[0],
        Kind::Region => {
            let &[mut a, mut b] = nodes[target].inputs.as_slice() else { unreachable!() };
            while a != b {
                let [ldepth, rdepth] = [idepth(nodes, a), idepth(nodes, b)];
                if ldepth >= rdepth {
                    a = scan_idom(nodes, a, pred)?;
                }
                if ldepth <= rdepth {
                    b = scan_idom(nodes, b, pred)?;
                }
            }
            a
        }
    })
}

fn walk_use_doms(
    target: Nid,
    nodes: &mut Nodes,
    mut all: impl FnMut(&mut Nodes, Nid) -> bool,
) -> bool {
    let dom =
        if matches!(nodes[target].kind, Kind::Call { .. }) { target } else { idom(nodes, target) };
    for mut out in nodes[target].outputs.clone().into_iter().skip(1) {
        out = use_block(target, out, nodes);
        while out != dom {
            debug_assert!(idepth(nodes, out) > idepth(nodes, dom),);
            debug_assert_ne!(out, VOID);
            out = match scan_idom(nodes, out, &mut all) {
                Some(next) => next,
                None => return false,
            };
        }
    }

    true
}

fn check_no_calls(target: Nid, nodes: &mut Nodes) -> bool {
    walk_use_doms(target, nodes, |nd, ud| !matches!(nd[ud].kind, Kind::Call { .. } | Kind::Loop))
}

fn is_last_branch_use(target: Nid, us: Nid, nodes: &mut Nodes) -> bool {
    if nodes[target].outputs.len() == 1 {
        return true;
    }

    let usdom = if matches!(nodes[us].kind, Kind::Call { .. }) { us } else { idom(nodes, us) };
    let tadom =
        if matches!(nodes[target].kind, Kind::Call { .. }) { target } else { idom(nodes, target) };
    if loop_depth(usdom, nodes) > loop_depth(tadom, nodes) {
        return false;
    }

    let outputs = nodes[target].outputs.clone();
    'o: for o in outputs {
        if o == us || idepth(nodes, o) < idepth(nodes, us) {
            continue;
        }

        let mut odom = if matches!(nodes[o].kind, Kind::Call { .. }) { o } else { idom(nodes, o) };

        if odom == usdom {
            return nodes[usdom].outputs.find(us) < nodes[odom].outputs.find(o);
        }

        while odom != usdom {
            odom = idom(nodes, odom);
            if idepth(nodes, odom) < idepth(nodes, us) {
                continue 'o;
            }
            debug_assert_ne!(odom, VOID);
        }

        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    const README: &str = include_str!("../README.md");

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        let mut codegen =
            super::Codegen { files: crate::test_parse_files(ident, input), ..Default::default() };

        codegen.generate();

        {
            let errors = codegen.errors.borrow();
            if !errors.is_empty() {
                output.push_str(&errors);
                return;
            }
        }

        let mut out = Vec::new();
        codegen.tys.assemble(&mut out);

        let mut buf = Vec::<u8>::new();
        let err = codegen.tys.disasm(&out, &codegen.files, &mut buf, |_| {});
        output.push_str(String::from_utf8(buf).unwrap().as_str());
        if let Err(e) = err {
            writeln!(output, "!!! asm is invalid: {e}").unwrap();
            return;
        }

        println!("{output}");

        crate::test_run_vm(&out, output);
    }

    crate::run_tests! { generate:
        arithmetic => README;
        variables => README;
        functions => README;
        comments => README;
        if_statements => README;
        loops => README;
        fb_driver => README;
        //pointers => README;
        //structs => README;
        //different_types => README;
        //struct_operators => README;
        //directives => README;
        //global_variables => README;
        //generic_types => README;
        //generic_functions => README;
        //c_strings => README;
        //struct_patterns => README;
        //arrays => README;
        //struct_return_from_module_function => README;
        ////comptime_pointers => README;
        //sort_something_viredly => README;
        hex_octal_binary_literals => README;
        //comptime_min_reg_leak => README;
        ////structs_in_registers => README;
        //comptime_function_from_another_file => README;
        //inline => README;
        //inline_test => README;
        const_folding_with_arg => README;
        // FIXME: contains redundant copies
        branch_assignments => README;
        exhaustive_loop_testing => README;
    }
}
