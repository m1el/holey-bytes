#![allow(dead_code)]
use {
    crate::{
        ident::{self, Ident},
        instrs,
        lexer::{self, TokenKind},
        log,
        parser::{
            self,
            idfl::{self},
            Expr, ExprRef, FileId, Pos,
        },
        HashMap,
    },
    core::fmt,
    std::{
        cell::RefCell,
        collections::{hash_map, BTreeMap},
        fmt::{Display, Write},
        hash::{Hash as _, Hasher},
        mem,
        ops::{self, Range},
        rc::Rc,
    },
};

macro_rules! node_color {
    ($self:expr, $value:expr) => {
        $self.ci.colors[$self.ci.nodes[$value].color as usize - 1]
    };
}

macro_rules! node_loc {
    ($self:expr, $value:expr) => {
        $self.ci.colors[$self.ci.nodes[$value].color as usize - 1].loc
    };
}

struct Drom(&'static str);

impl Drop for Drom {
    fn drop(&mut self) {
        log::inf!("{}", self.0);
    }
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
    pub fn set(&mut self, idx: usize) -> bool {
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        let prev = self.data[data_idx] & (1 << sub_idx);
        self.data[data_idx] |= 1 << sub_idx;
        prev == 0
    }

    fn unset(&mut self, idx: usize) {
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        self.data[data_idx] &= !(1 << sub_idx);
    }
}

type Nid = u32;

mod reg {

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

mod ty {
    use {
        crate::{
            lexer::TokenKind,
            parser::{self, Expr},
            son::ArrayLen,
        },
        std::{num::NonZeroU32, ops::Range},
    };

    pub type Builtin = u32;
    pub type Struct = u32;
    pub type Ptr = u32;
    pub type Func = u32;
    pub type Global = u32;
    pub type Module = u32;
    pub type Param = u32;
    pub type Slice = u32;

    #[derive(Clone, Copy)]
    pub struct Tuple(pub u32);

    impl Tuple {
        const LEN_BITS: u32 = 5;
        const LEN_MASK: usize = Self::MAX_LEN - 1;
        const MAX_LEN: usize = 1 << Self::LEN_BITS;

        pub fn new(pos: usize, len: usize) -> Option<Self> {
            if len >= Self::MAX_LEN {
                return None;
            }

            Some(Self((pos << Self::LEN_BITS | len) as u32))
        }

        pub fn view(self, slice: &[Id]) -> &[Id] {
            &slice[self.0 as usize >> Self::LEN_BITS..][..self.len()]
        }

        pub fn range(self) -> Range<usize> {
            let start = self.0 as usize >> Self::LEN_BITS;
            start..start + self.len()
        }

        pub fn len(self) -> usize {
            self.0 as usize & Self::LEN_MASK
        }

        pub fn is_empty(self) -> bool {
            self.0 == 0
        }

        pub fn empty() -> Self {
            Self(0)
        }

        pub fn repr(&self) -> u32 {
            self.0
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
    pub struct Id(NonZeroU32);

    impl Default for Id {
        fn default() -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(UNDECLARED) })
        }
    }

    impl Id {
        pub const fn from_bt(bt: u32) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(bt) })
        }

        pub fn is_signed(self) -> bool {
            (I8..=INT).contains(&self.repr())
        }

        pub fn is_unsigned(self) -> bool {
            (U8..=UINT).contains(&self.repr())
        }

        pub fn is_integer(self) -> bool {
            (U8..=INT).contains(&self.repr())
        }

        pub fn strip_pointer(self) -> Self {
            match self.expand() {
                Kind::Ptr(_) => Kind::Builtin(UINT).compress(),
                _ => self,
            }
        }

        pub fn is_pointer(self) -> bool {
            matches!(Kind::from_ty(self), Kind::Ptr(_))
        }

        pub fn try_upcast(self, ob: Self) -> Option<Self> {
            let (oa, ob) = (Self(self.0.min(ob.0)), Self(self.0.max(ob.0)));
            let (a, b) = (oa.strip_pointer(), ob.strip_pointer());
            Some(match () {
                _ if oa == ob => oa,
                _ if oa.is_pointer() && ob.is_pointer() => return None,
                _ if a.is_signed() && b.is_signed() || a.is_unsigned() && b.is_unsigned() => ob,
                _ if a.is_unsigned() && b.is_signed() && a.repr() - U8 < b.repr() - I8 => ob,
                _ if oa.is_integer() && ob.is_pointer() => ob,
                _ => return None,
            })
        }

        pub fn expand(self) -> Kind {
            Kind::from_ty(self)
        }

        pub const fn repr(self) -> u32 {
            self.0.get()
        }
    }

    impl From<u64> for Id {
        fn from(id: u64) -> Self {
            Self(unsafe { NonZeroU32::new_unchecked(id as _) })
        }
    }

    impl From<u32> for Id {
        fn from(id: u32) -> Self {
            Kind::Builtin(id).compress()
        }
    }

    const fn array_to_lower_case<const N: usize>(array: [u8; N]) -> [u8; N] {
        let mut result = [0; N];
        let mut i = 0;
        while i < N {
            result[i] = array[i].to_ascii_lowercase();
            i += 1;
        }
        result
    }
    // const string to lower case

    macro_rules! builtin_type {
        ($($name:ident;)*) => {
            $(pub const $name: Builtin = ${index(0)} + 1;)*

            mod __lc_names {
                use super::*;
                $(pub const $name: &[u8] = &array_to_lower_case(unsafe {
                    *(stringify!($name).as_ptr() as *const [u8; stringify!($name).len()]) });)*
            }

            pub fn from_str(name: &str) -> Option<Builtin> {
                match name.as_bytes() {
                    $(__lc_names::$name => Some($name),)*
                    _ => None,
                }
            }

            pub fn to_str(ty: Builtin) -> &'static str {
                match ty {
                    $($name => unsafe { std::str::from_utf8_unchecked(__lc_names::$name) },)*
                    v => unreachable!("invalid type: {}", v),
                }
            }
        };
    }

    builtin_type! {
        UNDECLARED;
        NEVER;
        VOID;
        TYPE;
        BOOL;
        U8;
        U16;
        U32;
        UINT;
        I8;
        I16;
        I32;
        INT;
        LEFT_UNREACHABLE;
        RIGHT_UNREACHABLE;
    }

    macro_rules! type_kind {
        ($(#[$meta:meta])* $vis:vis enum $name:ident {$( $variant:ident, )*}) => {
            $(#[$meta])*
            $vis enum $name {
                $($variant($variant),)*
            }

            impl $name {
                const FLAG_BITS: u32 = (${count($variant)} as u32).next_power_of_two().ilog2();
                const FLAG_OFFSET: u32 = std::mem::size_of::<Id>() as u32 * 8 - Self::FLAG_BITS;
                const INDEX_MASK: u32 = (1 << (32 - Self::FLAG_BITS)) - 1;

                $vis fn from_ty(ty: Id) -> Self {
                    let (flag, index) = (ty.repr() >> Self::FLAG_OFFSET, ty.repr() & Self::INDEX_MASK);
                    match flag {
                        $(${index(0)} => Self::$variant(index),)*
                        i => unreachable!("{i}"),
                    }
                }

                $vis const fn compress(self) -> Id {
                    let (index, flag) = match self {
                        $(Self::$variant(index) => (index, ${index(0)}),)*
                    };
                   Id(unsafe { NonZeroU32::new_unchecked((flag << Self::FLAG_OFFSET) | index) })
                }

                $vis const fn inner(self) -> u32 {
                    match self {
                        $(Self::$variant(index) => index,)*
                    }
                }
            }
        };
    }

    type_kind! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Kind {
            Builtin,
            Struct,
            Ptr,
            Func,
            Global,
            Module,
            Slice,
        }
    }

    impl Default for Kind {
        fn default() -> Self {
            Self::Builtin(UNDECLARED)
        }
    }

    pub struct Display<'a> {
        tys: &'a super::Types,
        files: &'a [parser::Ast],
        ty: Id,
    }

    impl<'a> Display<'a> {
        pub(super) fn new(tys: &'a super::Types, files: &'a [parser::Ast], ty: Id) -> Self {
            Self { tys, files, ty }
        }

        fn rety(&self, ty: Id) -> Self {
            Self::new(self.tys, self.files, ty)
        }
    }

    impl<'a> std::fmt::Display for Display<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use Kind as TK;
            match TK::from_ty(self.ty) {
                TK::Module(idx) => write!(f, "module{}", idx),
                TK::Builtin(ty) => write!(f, "{}", to_str(ty)),
                TK::Ptr(ty) => {
                    write!(f, "^{}", self.rety(self.tys.ptrs[ty as usize].base))
                }
                _ if let Some((key, _)) = self
                    .tys
                    .syms
                    .iter()
                    .find(|(sym, &ty)| sym.file < self.files.len() as u32 && ty == self.ty)
                    && let Some(name) = self.files[key.file as usize].exprs().iter().find_map(
                        |expr| match expr {
                            Expr::BinOp {
                                left: &Expr::Ident { name, id, .. },
                                op: TokenKind::Decl,
                                ..
                            } if id == key.ident => Some(name),
                            _ => None,
                        },
                    ) =>
                {
                    write!(f, "{name}")
                }
                TK::Struct(idx) => {
                    let record = &self.tys.structs[idx as usize];
                    write!(f, "{{")?;
                    for (i, &super::Field { ref name, ty }) in record.fields.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{name}: {}", self.rety(ty))?;
                    }
                    write!(f, "}}")
                }
                TK::Func(idx) => write!(f, "fn{idx}"),
                TK::Global(idx) => write!(f, "global{idx}"),
                TK::Slice(idx) => {
                    let array = self.tys.arrays[idx as usize];
                    match array.len {
                        ArrayLen::MAX => write!(f, "[{}]", self.rety(array.ty)),
                        len => write!(f, "[{}; {len}]", self.rety(array.ty)),
                    }
                }
            }
        }
    }

    pub fn bin_ret(ty: Id, op: TokenKind) -> Id {
        use TokenKind as T;
        match op {
            T::Lt | T::Gt | T::Le | T::Ge | T::Ne | T::Eq => BOOL.into(),
            _ => ty,
        }
    }
}

struct LookupEntry {
    nid: u32,
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

struct Nodes {
    values: Vec<Result<Node, u32>>,
    visited: BitSet,
    free: u32,
    lookup: std::collections::hash_map::HashMap<
        LookupEntry,
        (),
        std::hash::BuildHasherDefault<IdentityHash>,
    >,
}

impl Default for Nodes {
    fn default() -> Self {
        Self {
            values: Default::default(),
            free: u32::MAX,
            lookup: Default::default(),
            visited: Default::default(),
        }
    }
}

impl Nodes {
    fn remove_low(&mut self, id: u32) -> Node {
        let value = mem::replace(&mut self.values[id as usize], Err(self.free)).unwrap();
        self.free = id;
        value
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = u32::MAX;
    }

    fn new_node_nop(
        &mut self,
        ty: impl Into<ty::Id>,
        kind: Kind,
        inps: impl Into<Vec<Nid>>,
    ) -> Nid {
        let ty = ty.into();

        let node =
            Node { inputs: inps.into(), kind, color: 0, depth: 0, lock_rc: 0, ty, outputs: vec![] };

        let mut lookup_meta = None;
        if !node.is_lazy_phi() {
            let (raw_entry, hash) = Self::find_node(&mut self.lookup, &self.values, &node);

            let entry = match raw_entry {
                hash_map::RawEntryMut::Occupied(mut o) => return o.get_key_value().0.nid,
                hash_map::RawEntryMut::Vacant(v) => v,
            };

            lookup_meta = Some((entry, hash));
        }

        if self.free == u32::MAX {
            self.free = self.values.len() as _;
            self.values.push(Err(u32::MAX));
        }

        let free = self.free;
        for &d in &node.inputs {
            debug_assert_ne!(d, free);
            self.values[d as usize].as_mut().unwrap().outputs.push(free);
        }
        self.free = mem::replace(&mut self.values[free as usize], Ok(node)).unwrap_err();

        if let Some((entry, hash)) = lookup_meta {
            entry.insert(LookupEntry { nid: free, hash }, ());
        }
        free
    }

    fn find_node<'a>(
        lookup: &'a mut std::collections::hash_map::HashMap<
            LookupEntry,
            (),
            std::hash::BuildHasherDefault<IdentityHash>,
        >,
        values: &[Result<Node, u32>],
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

    fn new_node(&mut self, ty: impl Into<ty::Id>, kind: Kind, inps: impl Into<Vec<u32>>) -> Nid {
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

    fn peephole_binop(&mut self, target: Nid, op: TokenKind) -> Option<Nid> {
        use {Kind as K, TokenKind as T};
        let &[ctrl, mut lhs, mut rhs] = self[target].inputs.as_slice() else { unreachable!() };
        let ty = self[target].ty;

        if let (&K::CInt { value: a }, &K::CInt { value: b }) = (&self[lhs].kind, &self[rhs].kind) {
            return Some(self.new_node(ty, K::CInt { value: op.apply(a, b) }, [ctrl]));
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
                let new_rhs = self.new_node_nop(ty, K::CInt { value: op.apply(av, bv) }, [ctrl]);
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

    fn fmt(&self, f: &mut fmt::Formatter, node: Nid, rcs: &mut [usize]) -> fmt::Result {
        let mut is_ready = || {
            if rcs[node as usize] == 0 {
                return false;
            }
            rcs[node as usize] = rcs[node as usize].saturating_sub(1);
            rcs[node as usize] == 0
        };

        match self[node].kind {
            Kind::BinOp { op } => {
                write!(f, "(")?;
                self.fmt(f, self[node].inputs[0], rcs)?;
                write!(f, " {op} ")?;
                self.fmt(f, self[node].inputs[1], rcs)?;
                write!(f, ")")?;
            }
            Kind::Return => {
                write!(f, "{}: return [{:?}] ", node, self[node].inputs[0])?;
                if self[node].inputs[1] != 0 {
                    self.fmt(f, self[node].inputs[1], rcs)?;
                }
                writeln!(f)?;
                self.fmt(f, self[node].inputs[1], rcs)?;
            }
            Kind::CInt { value } => write!(f, "{}", value)?,
            Kind::End => {
                if is_ready() {
                    writeln!(f, "{}: {:?}", node, self[node].kind)?;
                }
            }
            Kind::Tuple { index } => {
                if index != 0 && self[self[node].inputs[0]].kind == Kind::Start {
                    write!(f, "{:?}.{}", self[self[node].inputs[0]].kind, index)?;
                } else if is_ready() {
                    writeln!(f, "{}: {:?}", node, self[node].kind)?;
                    for &o in &self[node].outputs {
                        if self.is_cfg(o) {
                            self.fmt(f, o, rcs)?;
                        }
                    }
                }
            }
            Kind::Start => 'b: {
                if !is_ready() {
                    break 'b;
                }

                writeln!(f, "{}: {:?}", node, self[node].kind)?;

                for &o in &self[node].outputs {
                    self.fmt(f, o, rcs)?;
                }
            }
            Kind::Call { func } => {
                if is_ready() {
                    write!(f, "{}: call {}(", node, func)?;
                    for (i, &value) in self[node].inputs.iter().skip(1).enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        self.fmt(f, value, rcs)?;
                    }
                    writeln!(f, ")")?;
                    for &o in &self[node].outputs {
                        if self.is_cfg(o) {
                            self.fmt(f, o, rcs)?;
                        }
                    }
                } else {
                    write!(f, "call{node}")?;
                }
            }
            Kind::If => todo!(),
            Kind::Region => todo!(),
            Kind::Phi => todo!(),
            Kind::Loop => todo!(),
        }

        Ok(())
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
        if !self.visited.set(node as _) {
            return Ok(());
        }
        write!(out, "  {node:>2}: ")?;
        match self[node].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::If => write!(out, "  if:      "),
            Kind::Region => unreachable!(),
            Kind::Loop => unreachable!(),
            Kind::Return => write!(out, " ret: "),
            Kind::CInt { value } => write!(out, "cint: #{value:<4}"),
            Kind::Phi => write!(out, " phi:      "),
            Kind::Tuple { index } => write!(out, " arg: {index:<5}"),
            Kind::BinOp { op } => {
                write!(out, "{:>4}:      ", op.name())
            }
            Kind::Call { func } => {
                write!(out, "call: {func} {}", self[node].depth)
            }
        }?;

        writeln!(
            out,
            " {:<14} {}",
            format!("{:?}", self[node].inputs),
            format!("{:?}", self[node].outputs)
        )
    }

    fn basic_blocks_low(&mut self, out: &mut String, mut node: Nid) -> std::fmt::Result {
        while self.visited.set(node as _) {
            match dbg!(self[node].kind) {
                Kind::Start => {
                    writeln!(out, "start: {}", self[node].depth)?;
                    let mut cfg_index = Nid::MAX;
                    for o in self[node].outputs.clone() {
                        if self[o].kind == (Kind::Tuple { index: 0 }) {
                            cfg_index = o;
                            continue;
                        }
                        self.basic_blocks_instr(out, o)?;
                    }
                    node = cfg_index;
                }
                Kind::End => break,
                Kind::If => {
                    self.visited.unset(node as _);
                    self.basic_blocks_instr(out, node)?;
                    self.basic_blocks_low(out, self[node].outputs[0])?;
                    node = self[node].outputs[1];
                }
                Kind::Region => {
                    let mut cfg_index = Nid::MAX;
                    for o in self[node].outputs.clone().into_iter() {
                        if self.is_cfg(o) {
                            cfg_index = o;
                            continue;
                        }
                        self.basic_blocks_instr(out, o)?;
                    }
                    node = cfg_index;
                }
                Kind::Loop => {
                    writeln!(out, "loop{node}  {}", self[node].depth)?;
                    let mut cfg_index = Nid::MAX;
                    for o in self[node].outputs.clone().into_iter() {
                        if self.is_cfg(o) {
                            cfg_index = o;
                            continue;
                        }
                        self.basic_blocks_instr(out, o)?;
                    }
                    node = cfg_index;
                }
                Kind::Return => {
                    self.visited.unset(node as _);
                    self.basic_blocks_instr(out, node)?;
                    node = self[node].outputs[0];
                }
                Kind::CInt { .. } => unreachable!(),
                Kind::Phi => unreachable!(),
                Kind::Tuple { .. } => {
                    writeln!(out, "b{node}: {}", self[node].depth)?;
                    let mut cfg_index = Nid::MAX;
                    for o in self[node].outputs.clone().into_iter() {
                        if self.is_cfg(o) {
                            cfg_index = o;
                            continue;
                        }
                        self.basic_blocks_instr(out, o)?;
                    }
                    if !self[cfg_index].kind.ends_basic_block()
                        && !matches!(self[cfg_index].kind, Kind::Call { .. })
                    {
                        writeln!(out, "      goto: {cfg_index}")?;
                    }
                    node = cfg_index;
                }
                Kind::BinOp { .. } => unreachable!(),
                Kind::Call { .. } => {
                    self.visited.unset(node as _);
                    self.basic_blocks_instr(out, node)?;

                    let mut cfg_index = Nid::MAX;
                    for o in self[node].outputs.clone().into_iter() {
                        if self.is_cfg(o) {
                            cfg_index = o;
                            continue;
                        }
                        if self[o].inputs[0] == node {
                            self.basic_blocks_instr(out, o)?;
                        }
                    }
                    node = cfg_index;
                }
            }
        }

        Ok(())
    }

    fn basic_blocks(&mut self) {
        let mut out = String::new();
        self.visited.clear(self.values.len());
        self.basic_blocks_low(&mut out, 0).unwrap();
        println!("{out}");
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
        let mut failed = false;
        for (i, node) in self.iter() {
            debug_assert_eq!(node.lock_rc, 0, "{:?}", node.kind);
            if !matches!(node.kind, Kind::Return | Kind::End) && node.outputs.is_empty() {
                log::err!("outputs are empry {i} {:?}", node.kind);
                failed = true;
            }

            let mut allowed_cfgs = 1 + (node.kind == Kind::If) as usize;
            for &o in &node.outputs {
                if self.is_cfg(i) {
                    if allowed_cfgs == 0 && self.is_cfg(o) {
                        log::err!(
                            "multiple cfg outputs detected: {:?} -> {:?}",
                            node.kind,
                            self[o].kind
                        );
                        failed = true;
                    } else {
                        allowed_cfgs += self.is_cfg(o) as usize;
                    }
                }
                if matches!(node.kind, Kind::Region | Kind::Loop)
                    && !self.is_cfg(o)
                    && self[o].kind != Kind::Phi
                {
                    log::err!("unexpected output node on region: {:?}", self[o].kind);
                    failed = true;
                }

                let other = match &self.values[o as usize] {
                    Ok(other) => other,
                    Err(_) => {
                        log::err!("the edge points to dropped node: {i} {:?} {o}", node.kind,);
                        failed = true;
                        continue;
                    }
                };
                let occurs = self[o].inputs.iter().filter(|&&el| el == i).count();
                let self_occurs = self[i].outputs.iter().filter(|&&el| el == o).count();
                if occurs != self_occurs {
                    log::err!(
                        "the edge is not bidirectional: {i} {:?} {self_occurs} {o} {:?} {occurs}",
                        node.kind,
                        other.kind
                    );
                    failed = true;
                }
            }
        }
        if failed {
            panic!()
        }
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
                    && nodes.visited.set(n as usize)
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
            let inps = [loob.node, *lvalue, 0];
            self.unlock(inps[1]);
            let ty = self[inps[1]].ty;
            let phi = self.new_node_nop(ty, Kind::Phi, inps);
            self[phi].lock_rc += 2;
            *value = phi;
            *lvalue = phi;
        } else {
            self.unlock_remove(*value);
            *value = *lvalue;
            self.lock(*value);
        }
    }
}

impl ops::Index<u32> for Nodes {
    type Output = Node;

    fn index(&self, index: u32) -> &Self::Output {
        self.values[index as usize].as_ref().unwrap()
    }
}

impl ops::IndexMut<u32> for Nodes {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        self.values[index as usize].as_mut().unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Start,
    End,
    If,
    Region,
    Loop,
    Return,
    CInt { value: i64 },
    Phi,
    Tuple { index: u32 },
    BinOp { op: lexer::TokenKind },
    Call { func: ty::Func },
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

#[derive(Debug)]
struct Node {
    inputs: Vec<Nid>,
    outputs: Vec<Nid>,
    kind: Kind,
    color: u32,
    depth: u32,
    lock_rc: u32,
    ty: ty::Id,
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

impl fmt::Display for Nodes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(
            f,
            0,
            &mut self
                .values
                .iter()
                .map(|s| match s {
                    Ok(Node { kind: Kind::Start, .. }) => 1,
                    Ok(Node { kind: Kind::End, ref outputs, .. }) => outputs.len(),
                    Ok(val) => val.inputs.len(),
                    Err(_) => 0,
                })
                .collect::<Vec<_>>(),
        )
    }
}

type Offset = u32;
type Size = u32;
type ArrayLen = u32;

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
    depth: u32,
    call_count: u32,
    loc: Loc,
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    id: ty::Id,
    ret: Option<ty::Id>,

    task_base: usize,

    nodes: Nodes,
    start: Nid,
    end: Nid,
    ctrl: Nid,

    loop_depth: u32,
    colors: Vec<ColorMeta>,
    call_count: u32,
    filled: Vec<Nid>,
    delayed_frees: Vec<u32>,

    loops: Vec<Loop>,
    vars: Vec<Variable>,
    regs: reg::Alloc,
    ret_relocs: Vec<Reloc>,
    relocs: Vec<TypedReloc>,
    code: Vec<u8>,
}

impl ItemCtx {
    fn next_color(&mut self) -> u32 {
        self.colors.push(ColorMeta {
            rc: 0,
            call_count: self.call_count,
            depth: self.loop_depth,
            loc: Default::default(),
        });
        self.colors.len() as _ // leave out 0 (sentinel)
    }

    fn set_next_color(&mut self, node: Nid) {
        let color = self.next_color();
        self.set_color(node, color);
    }

    fn set_color(&mut self, node: Nid, color: u32) {
        if self.nodes[node].color != 0 {
            debug_assert_ne!(self.nodes[node].color, color);
            self.colors[self.nodes[node].color as usize - 1].rc -= 1;
        }
        self.nodes[node].color = color;
        self.colors[color as usize - 1].rc += 1;
    }

    fn recolor(&mut self, node: Nid, from: u32, to: u32) {
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
                        ..
                    })
                ) || matches!(
                    v,
                    Ok(Node { kind: Kind::Tuple { index: 1.. }, inputs, .. }) if inputs.first() == Some(&0)
                )
            })
            .count();
        let color_count = self.colors.iter().map(|c| c.rc).sum::<u32>();
        debug_assert_eq!(node_count, color_count as usize);
    }

    fn emit(&mut self, instr: (usize, [u8; instrs::MAX_SIZE])) {
        emit(&mut self.code, instr);
    }

    fn free_loc(&mut self, loc: impl Into<Option<Loc>>) {
        if let Some(loc) = loc.into() {
            self.regs.free(loc.reg);
        }
    }
}

fn emit(out: &mut Vec<u8>, (len, instr): (usize, [u8; instrs::MAX_SIZE])) {
    out.extend_from_slice(&instr[..len]);
}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
}

#[derive(PartialEq, Eq, Hash)]
struct SymKey {
    file: u32,
    ident: u32,
}

impl SymKey {
    pub fn pointer_to(ty: ty::Id) -> Self {
        Self { file: u32::MAX, ident: ty.repr() }
    }
}

#[derive(Clone, Copy)]
struct Sig {
    args: ty::Tuple,
    ret: ty::Id,
}

struct Func {
    file: FileId,
    expr: ExprRef,
    sig: Option<Sig>,
    offset: Offset,
    // TODO: change to indices into common vec
    relocs: Vec<TypedReloc>,
    code: Vec<u8>,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            file: u32::MAX,
            expr: Default::default(),
            sig: None,
            offset: u32::MAX,
            relocs: Default::default(),
            code: Default::default(),
        }
    }
}

struct TypedReloc {
    target: ty::Id,
    reloc: Reloc,
}

struct Global {
    file: FileId,
    name: Ident,
    ty: ty::Id,
    offset: Offset,
    data: Vec<u8>,
}

impl Default for Global {
    fn default() -> Self {
        Self {
            ty: Default::default(),
            offset: u32::MAX,
            data: Default::default(),
            file: 0,
            name: 0,
        }
    }
}

// TODO: make into bit struct (width: u2, sub_offset: u3, offset: u27)
#[derive(Clone, Copy, Debug)]
struct Reloc {
    offset: Offset,
    sub_offset: u8,
    width: u8,
}

impl Reloc {
    fn new(offset: usize, sub_offset: u8, width: u8) -> Self {
        Self { offset: offset as u32, sub_offset, width }
    }

    fn apply_jump(mut self, code: &mut [u8], to: u32, from: u32) -> i64 {
        self.offset += from;
        let offset = to as i64 - self.offset as i64;
        self.write_offset(code, offset);
        offset
    }

    fn write_offset(&self, code: &mut [u8], offset: i64) {
        let bytes = offset.to_ne_bytes();
        let slice = &mut code[self.offset as usize + self.sub_offset as usize..];
        slice[..self.width as usize].copy_from_slice(&bytes[..self.width as usize]);
    }
}

struct Field {
    name: Rc<str>,
    ty: ty::Id,
}

struct Struct {
    fields: Rc<[Field]>,
}

struct Ptr {
    base: ty::Id,
}

struct ParamAlloc(Range<u8>);

impl ParamAlloc {
    pub fn next(&mut self) -> u8 {
        self.0.next().expect("too many paramteters")
    }

    fn next_wide(&mut self) -> u8 {
        (self.next(), self.next()).0
    }
}

#[derive(Clone, Copy)]
struct Array {
    ty: ty::Id,
    len: ArrayLen,
}

#[derive(Default)]
struct Types {
    syms: HashMap<SymKey, ty::Id>,

    funcs: Vec<Func>,
    args: Vec<ty::Id>,
    globals: Vec<Global>,
    structs: Vec<Struct>,
    ptrs: Vec<Ptr>,
    arrays: Vec<Array>,
}

impl Types {
    fn parama(&self, ret: impl Into<ty::Id>) -> ParamAlloc {
        ParamAlloc(2 + (9..=16).contains(&self.size_of(ret.into())) as u8..12)
    }

    fn offset_of(&self, idx: ty::Struct, field: &str) -> Option<(Offset, ty::Id)> {
        let record = &self.structs[idx as usize];
        let until = record.fields.iter().position(|f| f.name.as_ref() == field)?;
        let mut offset = 0;
        for &Field { ty, .. } in &record.fields[..until] {
            offset = Self::align_up(offset, self.align_of(ty));
            offset += self.size_of(ty);
        }
        Some((offset, record.fields[until].ty))
    }

    fn make_ptr(&mut self, base: ty::Id) -> ty::Id {
        ty::Kind::Ptr(self.make_ptr_low(base)).compress()
    }

    fn make_ptr_low(&mut self, base: ty::Id) -> ty::Ptr {
        let id = SymKey::pointer_to(base);

        self.syms
            .entry(id)
            .or_insert_with(|| {
                self.ptrs.push(Ptr { base });
                ty::Kind::Ptr(self.ptrs.len() as u32 - 1).compress()
            })
            .expand()
            .inner()
    }

    fn make_array(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Id {
        ty::Kind::Slice(self.make_array_low(ty, len)).compress()
    }

    fn make_array_low(&mut self, ty: ty::Id, len: ArrayLen) -> ty::Slice {
        let id = SymKey {
            file: match len {
                ArrayLen::MAX => ArrayLen::MAX - 1,
                len => ArrayLen::MAX - len - 2,
            },
            ident: ty.repr(),
        };

        self.syms
            .entry(id)
            .or_insert_with(|| {
                self.arrays.push(Array { ty, len });
                ty::Kind::Slice(self.arrays.len() as u32 - 1).compress()
            })
            .expand()
            .inner()
    }

    fn align_up(value: Size, align: Size) -> Size {
        (value + align - 1) & !(align - 1)
    }

    fn size_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Ptr(_) => 8,
            ty::Kind::Builtin(ty::VOID) => 0,
            ty::Kind::Builtin(ty::NEVER) => unreachable!(),
            ty::Kind::Builtin(ty::INT | ty::UINT) => 8,
            ty::Kind::Builtin(ty::I32 | ty::U32 | ty::TYPE) => 4,
            ty::Kind::Builtin(ty::I16 | ty::U16) => 2,
            ty::Kind::Builtin(ty::I8 | ty::U8 | ty::BOOL) => 1,
            ty::Kind::Slice(arr) => {
                let arr = &self.arrays[arr as usize];
                match arr.len {
                    0 => 0,
                    ArrayLen::MAX => 16,
                    len => self.size_of(arr.ty) * len,
                }
            }
            ty::Kind::Struct(stru) => {
                let mut offset = 0u32;
                let record = &self.structs[stru as usize];
                for &Field { ty, .. } in record.fields.iter() {
                    let align = self.align_of(ty);
                    offset = Self::align_up(offset, align);
                    offset += self.size_of(ty);
                }
                offset
            }
            ty => unimplemented!("size_of: {:?}", ty),
        }
    }

    fn align_of(&self, ty: ty::Id) -> Size {
        match ty.expand() {
            ty::Kind::Struct(stru) => self.structs[stru as usize]
                .fields
                .iter()
                .map(|&Field { ty, .. }| self.align_of(ty))
                .max()
                .unwrap(),
            ty::Kind::Slice(arr) => {
                let arr = &self.arrays[arr as usize];
                match arr.len {
                    ArrayLen::MAX => 8,
                    _ => self.align_of(arr.ty),
                }
            }
            _ => self.size_of(ty).max(1),
        }
    }
}

mod task {
    use super::Offset;

    pub fn unpack(offset: Offset) -> Result<Offset, usize> {
        if offset >> 31 != 0 {
            Err((offset & !(1 << 31)) as usize)
        } else {
            Ok(offset)
        }
    }

    pub fn id(index: usize) -> Offset {
        1 << 31 | index as u32
    }
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

    fn assemble(&mut self, to: &mut Vec<u8>) {
        emit(to, instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
        emit(to, instrs::tx());
        self.dump_reachable(0, to);
        Reloc::new(0, 3, 4).apply_jump(to, self.tys.funcs[0].offset, 0);
    }

    fn dump_reachable(&mut self, from: ty::Func, to: &mut Vec<u8>) {
        let mut frontier = vec![ty::Kind::Func(from).compress()];

        while let Some(itm) = frontier.pop() {
            match itm.expand() {
                ty::Kind::Func(func) => {
                    let fuc = &mut self.tys.funcs[func as usize];
                    if task::unpack(fuc.offset).is_ok() {
                        continue;
                    }
                    fuc.offset = to.len() as _;
                    to.extend(&fuc.code);
                    frontier.extend(fuc.relocs.iter().map(|r| r.target));
                }
                ty::Kind::Global(glob) => {
                    let glb = &mut self.tys.globals[glob as usize];
                    if task::unpack(glb.offset).is_ok() {
                        continue;
                    }
                    glb.offset = to.len() as _;
                    to.extend(&glb.data);
                }
                _ => unreachable!(),
            }
        }

        for fuc in &self.tys.funcs {
            if task::unpack(fuc.offset).is_err() {
                continue;
            }

            for rel in &fuc.relocs {
                let offset = match rel.target.expand() {
                    ty::Kind::Func(fun) => self.tys.funcs[fun as usize].offset,
                    ty::Kind::Global(glo) => self.tys.globals[glo as usize].offset,
                    _ => unreachable!(),
                };
                rel.reloc.apply_jump(to, offset, fuc.offset);
            }
        }
    }

    pub fn disasm(
        &mut self,
        mut sluce: &[u8],
        output: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        use crate::DisasmItem;
        let functions = self
            .tys
            .funcs
            .iter()
            .filter(|f| task::unpack(f.offset).is_ok())
            .map(|f| {
                let file = &self.files[f.file as usize];
                let Expr::BinOp { left: &Expr::Ident { name, .. }, .. } = f.expr.get(file).unwrap()
                else {
                    unreachable!()
                };
                (f.offset, (name, f.code.len() as u32, DisasmItem::Func))
            })
            .chain(self.tys.globals.iter().filter(|g| task::unpack(g.offset).is_ok()).map(|g| {
                let file = &self.files[g.file as usize];

                (g.offset, (file.ident_str(g.name), self.tys.size_of(g.ty), DisasmItem::Global))
            }))
            .collect::<BTreeMap<_, _>>();
        crate::disasm(&mut sluce, &functions, output, |_| {})
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

    fn build_struct(&mut self, fields: &[(&str, Expr)]) -> ty::Struct {
        let fields = fields
            .iter()
            .map(|&(name, ty)| Field { name: name.into(), ty: self.ty(&ty) })
            .collect();
        self.tys.structs.push(Struct { fields });
        self.tys.structs.len() as u32 - 1
    }

    fn expr_ctx(&mut self, expr: &Expr, ctx: Ctx) -> Option<Nid> {
        let msg = "i know nothing about this name gal which is vired \
                            because we parsed succesfully";
        match *expr {
            Expr::Comment { .. } => Some(0),
            Expr::Ident { pos, id, .. } => {
                let Some(index) = self.ci.vars.iter().position(|v| v.id == id) else {
                    self.report(pos, msg);
                    return Some(self.ci.end);
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
                Some(0)
            }
            Expr::BinOp { left: &Expr::Ident { id, pos, .. }, op: TokenKind::Assign, right } => {
                let value = self.expr(right)?;
                self.ci.nodes.lock(value);

                let Some(var) = self.ci.vars.iter_mut().find(|v| v.id == id) else {
                    self.report(pos, msg);
                    return Some(self.ci.end);
                };

                let prev = std::mem::replace(&mut var.value, value);
                self.ci.nodes.unlock_remove(prev);
                Some(0)
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
                let inps = [0, lhs, rhs];
                Some(self.ci.nodes.new_node(ty::bin_ret(ty, op), Kind::BinOp { op }, inps))
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
                        return Some(0);
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
                    return Some(0);
                } else if rcntrl == Nid::MAX {
                    for else_var in &self.ci.vars {
                        self.ci.nodes.unlock_remove(else_var.value);
                    }
                    self.ci.vars = then_scope;
                    self.ci.ctrl = lcntrl;
                    return Some(0);
                }

                self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Region, [lcntrl, rcntrl]);

                else_scope = std::mem::take(&mut self.ci.vars);

                for (i, (else_var, then_var)) in
                    else_scope.iter_mut().zip(&mut then_scope).enumerate()
                {
                    if else_var.value == then_var.value {
                        self.ci.nodes.unlock_remove(then_var.value);
                        continue;
                    }

                    self.ci.nodes.load_loop_value(i, &mut then_var.value, &mut self.ci.loops);
                    self.ci.nodes.load_loop_value(i, &mut else_var.value, &mut self.ci.loops);

                    self.ci.nodes.unlock(then_var.value);

                    let ty = self.ci.nodes[else_var.value].ty;
                    debug_assert_eq!(
                        ty,
                        self.ci.nodes[then_var.value].ty,
                        "TODO: typecheck properly: {} != {}\n{}",
                        self.ty_display(ty),
                        self.ty_display(self.ci.nodes[then_var.value].ty),
                        self.errors.borrow()
                    );

                    let inps = [self.ci.ctrl, then_var.value, else_var.value];
                    self.ci.nodes.unlock(else_var.value);
                    else_var.value = self.ci.nodes.new_node(ty, Kind::Phi, inps);
                    self.ci.nodes.lock(else_var.value);
                }

                self.ci.vars = else_scope;

                Some(0)
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
                    var.value = 0;
                }
                self.ci.nodes[0].lock_rc += self.ci.vars.len() as u32;

                self.expr(body);

                if self.ci.loops.last_mut().unwrap().ctrl[0] != Nid::MAX {
                    self.jump_to(0, 0);
                    self.ci.ctrl = self.ci.loops.last_mut().unwrap().ctrl[0];
                }

                let Loop { node, ctrl: [.., bre], ctrl_scope: [.., mut bre_scope], scope } =
                    self.ci.loops.pop().unwrap();

                self.ci.nodes.modify_input(node, 1, self.ci.ctrl);

                self.ci.ctrl = bre;
                if bre == Nid::MAX {
                    return None;
                }

                self.ci.nodes.lock(self.ci.ctrl);

                std::mem::swap(&mut self.ci.vars, &mut bre_scope);

                for ((dest_var, mut scope_var), loop_var) in
                    self.ci.vars.iter_mut().zip(scope).zip(bre_scope)
                {
                    self.ci.nodes.unlock(loop_var.value);

                    if loop_var.value != 0 {
                        self.ci.nodes.unlock(scope_var.value);
                        if loop_var.value != scope_var.value {
                            scope_var.value =
                                self.ci.nodes.modify_input(scope_var.value, 2, loop_var.value);
                            self.ci.nodes.lock(scope_var.value);
                        } else {
                            let phi = &self.ci.nodes[scope_var.value];
                            debug_assert_eq!(phi.kind, Kind::Phi);
                            debug_assert_eq!(phi.inputs[2], 0);
                            let prev = phi.inputs[1];
                            self.ci.nodes.replace(scope_var.value, prev);
                            scope_var.value = prev;
                            self.ci.nodes.lock(prev);
                        }
                    }

                    if dest_var.value == 0 {
                        self.ci.nodes.unlock_remove(dest_var.value);
                        dest_var.value = scope_var.value;
                        self.ci.nodes.lock(dest_var.value);
                    }

                    self.ci.nodes.unlock_remove(scope_var.value);
                }

                self.ci.nodes.unlock(self.ci.ctrl);

                Some(0)
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
                    return Some(self.ci.end);
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

                let mut inps = vec![self.ci.ctrl];
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
                    0
                };

                let inps = [self.ci.ctrl, value];

                let out = &mut String::new();
                self.report_log_to(pos, "returning here", out);
                self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Return, inps);

                self.ci.nodes[self.ci.end].inputs.push(self.ci.ctrl);
                self.ci.nodes[self.ci.ctrl].outputs.push(self.ci.end);

                let expected = *self.ci.ret.get_or_insert(self.tof(value));
                _ = self.assert_ty(pos, self.tof(value), expected, true, "return value");

                None
            }
            Expr::Block { stmts, .. } => {
                let base = self.ci.vars.len();

                let mut ret = Some(0);
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
                [0],
            )),
            ref e => {
                self.report_unhandled_ast(e, "bruh");
                Some(self.ci.end)
            }
        }
    }

    fn jump_to(&mut self, pos: Pos, id: usize) -> Option<Nid> {
        let Some(loob) = self.ci.loops.last_mut() else {
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
            loob.ctrl[id] =
                self.ci.nodes.new_node(ty::VOID, Kind::Region, [self.ci.ctrl, loob.ctrl[id]]);

            for (else_var, then_var) in loob.ctrl_scope[id].iter_mut().zip(&self.ci.vars) {
                if else_var.value == then_var.value {
                    continue;
                }

                let ty = self.ci.nodes[else_var.value].ty;
                debug_assert_eq!(ty, self.ci.nodes[then_var.value].ty, "TODO: typecheck properly");

                let inps = [loob.ctrl[id], then_var.value, else_var.value];
                self.ci.nodes.unlock(else_var.value);
                else_var.value = self.ci.nodes.new_node(ty, Kind::Phi, inps);
                self.ci.nodes.lock(else_var.value);
            }
        }

        self.ci.ctrl = self.ci.end;
        None
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
            self.handle_task(task);
        }
    }

    fn handle_task(&mut self, FTask { file, id }: FTask) {
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

        self.ci.start = self.ci.nodes.new_node(ty::VOID, Kind::Start, []);
        self.ci.end = self.ci.nodes.new_node(ty::NEVER, Kind::End, []);
        self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [self.ci.start]);

        let Expr::BinOp {
            left: Expr::Ident { name, .. },
            op: TokenKind::Decl,
            right: &Expr::Closure { body, args, .. },
        } = expr
        else {
            unreachable!("{expr}")
        };

        let mut sig_args = sig.args.range();
        for (arg, index) in args.iter().zip(1u32..) {
            let ty = self.tys.args[sig_args.next().unwrap()];
            let value = self.ci.nodes.new_node(ty, Kind::Tuple { index }, [self.ci.start]);
            self.ci.nodes.lock(value);
            let sym = parser::find_symbol(&ast.symbols, arg.id);
            assert!(sym.flags & idfl::COMPTIME == 0, "TODO");
            self.ci.vars.push(Variable { id: arg.id, value });
        }

        let mut orig_vars = self.ci.vars.clone();

        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        for (i, var) in self.ci.vars.drain(..).enumerate() {
            let ty = self.ci.nodes[var.value].ty;
            if self.ci.nodes.unlock_remove(var.value) {
                // mark as unused
                orig_vars[i].id = u32::MAX;
                orig_vars[i].value = ty.repr();
            }
        }

        if self.errors.borrow().is_empty() {
            self.gcm();

            //self.ci.nodes.graphviz();
            log::inf!("{id} {name}: ");
            self.ci.nodes.basic_blocks();

            return;

            #[cfg(debug_assertions)]
            {
                self.ci.nodes.check_final_integrity();
            }

            log::trc!("{}", self.ci.nodes);

            '_open_function: {
                self.ci.emit(instrs::addi64(reg::STACK_PTR, reg::STACK_PTR, 0));
                self.ci.emit(instrs::st(reg::RET_ADDR, reg::STACK_PTR, 0, 0));
            }

            self.ci.regs.init();

            let call_count = self.ci.call_count;
            '_color_args: {
                for var in &orig_vars {
                    if var.id != u32::MAX {
                        self.ci.set_next_color(var.value);
                    }
                }
            }
            self.color_control(self.ci.nodes[self.ci.start].outputs[0]);
            #[cfg(debug_assertions)]
            {
                self.ci.check_color_integrity();
            }

            self.ci.vars = orig_vars;
            self.ci.call_count = call_count;
            self.emit_control(self.ci.nodes[self.ci.start].outputs[0]);
            self.ci.vars.clear();

            if let Some(last_ret) = self.ci.ret_relocs.last()
                && last_ret.offset as usize == self.ci.code.len() - 5
            {
                self.ci.code.truncate(self.ci.code.len() - 5);
                self.ci.ret_relocs.pop();
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

    fn color_control(&mut self, mut ctrl: Nid) -> Option<Nid> {
        for _ in 0..30 {
            match self.ci.nodes[ctrl].kind {
                Kind::Start => unreachable!(),
                Kind::End => unreachable!(),
                Kind::Return => {
                    let ret = self.ci.nodes[ctrl].inputs[1];
                    if ret != 0 {
                        _ = self.color_expr_consume(ret);
                        if node_color!(self, ret).call_count == self.ci.call_count {
                            node_loc!(self, ret) =
                                match self.tys.size_of(self.ci.ret.expect("TODO")) {
                                    0 => Loc::default(),
                                    1..=8 => Loc { reg: 1 },
                                    s => todo!("{s}"),
                                };
                        }
                        self.ci.regs.mark_leaked(1);
                    }
                    return None;
                }
                Kind::CInt { .. } => unreachable!(),
                Kind::Tuple { .. } => {
                    ctrl = self.ci.nodes[ctrl].outputs[0];
                }
                Kind::BinOp { .. } => unreachable!(),
                Kind::Call { .. } => {
                    for i in 1..self.ci.nodes[ctrl].inputs.len() {
                        let arg = self.ci.nodes[ctrl].inputs[i];
                        _ = self.color_expr_consume(arg);
                        self.ci.set_next_color(arg);
                    }

                    self.ci.call_count -= 1;

                    self.ci.set_next_color(ctrl);

                    ctrl = *self.ci.nodes[ctrl]
                        .outputs
                        .iter()
                        .find(|&&o| self.ci.nodes.is_cfg(o))
                        .unwrap();
                }
                Kind::If => {
                    _ = self.color_expr_consume(self.ci.nodes[ctrl].inputs[1]);

                    let left_unreachable = self.color_control(self.ci.nodes[ctrl].outputs[0]);
                    let right_unreachable = self.color_control(self.ci.nodes[ctrl].outputs[1]);

                    let dest = match (left_unreachable, right_unreachable) {
                        (None, None) => return None,
                        (None, Some(n)) | (Some(n), None) => return Some(n),
                        (Some(l), Some(r)) if l == r => l,
                        (Some(left), Some(right)) => {
                            todo!("{:?} {:?}", self.ci.nodes[left], self.ci.nodes[right]);
                        }
                    };

                    if self.ci.nodes[dest].kind == Kind::Loop {
                        return Some(dest);
                    }

                    debug_assert_eq!(self.ci.nodes[dest].kind, Kind::Region);

                    for i in 0..self.ci.nodes[dest].outputs.len() {
                        let o = self.ci.nodes[dest].outputs[i];
                        if self.ci.nodes[o].kind == Kind::Phi {
                            self.color_phi(o);
                            self.ci.nodes[o].depth = self.ci.loop_depth;
                        }
                    }

                    ctrl = *self.ci.nodes[dest]
                        .outputs
                        .iter()
                        .find(|&&o| self.ci.nodes[o].kind != Kind::Phi)
                        .unwrap();
                }
                Kind::Region => return Some(ctrl),
                Kind::Phi => todo!(),
                Kind::Loop => {
                    if self.ci.nodes[ctrl].lock_rc != 0 {
                        return Some(ctrl);
                    }

                    for i in 0..self.ci.nodes[ctrl].outputs.len() {
                        let maybe_phi = self.ci.nodes[ctrl].outputs[i];
                        let Node { kind: Kind::Phi, ref inputs, .. } = self.ci.nodes[maybe_phi]
                        else {
                            continue;
                        };

                        _ = self.color_expr_consume(inputs[1]);
                        self.ci.nodes[maybe_phi].depth = self.ci.loop_depth;
                        self.ci.set_next_color(maybe_phi);
                    }

                    self.ci.nodes[ctrl].lock_rc = self.ci.code.len() as _;
                    self.ci.loop_depth += 1;

                    self.color_control(
                        *self.ci.nodes[ctrl]
                            .outputs
                            .iter()
                            .find(|&&o| self.ci.nodes[o].kind != Kind::Phi)
                            .unwrap(),
                    );

                    for i in 0..self.ci.nodes[ctrl].outputs.len() {
                        self.color_phi(self.ci.nodes[ctrl].outputs[i]);
                    }

                    self.ci.loop_depth -= 1;
                    self.ci.nodes[ctrl].lock_rc = 0;

                    return None;
                }
            }
        }

        unreachable!()
    }

    fn color_phi(&mut self, maybe_phi: Nid) {
        let Node { kind: Kind::Phi, ref inputs, .. } = self.ci.nodes[maybe_phi] else {
            return;
        };
        let &[region, left, right] = inputs.as_slice() else { unreachable!() };

        let lcolor = self.color_expr_consume(left);
        let rcolor = self.color_expr_consume(right);

        if self.ci.nodes[maybe_phi].color != 0 {
            // loop phi
            if let Some(c) = rcolor
                && !self.ci.nodes.climb_expr(right, |i, n| {
                    matches!(n.kind, Kind::Phi) && n.inputs[0] == region && i != maybe_phi
                })
            {
                self.ci.recolor(right, c, self.ci.nodes[maybe_phi].color);
            }
        } else {
            let color = match (lcolor, rcolor) {
                (None, None) => self.ci.next_color(),
                (None, Some(c)) | (Some(c), None) => c,
                (Some(lc), Some(rc)) => {
                    self.ci.recolor(right, rc, lc);
                    lc
                }
            };
            self.ci.set_color(maybe_phi, color);
        }
    }

    #[must_use = "dont forget to drop the location"]
    fn color_expr_consume(&mut self, expr: Nid) -> Option<u32> {
        if self.ci.nodes[expr].lock_rc == 0 && self.ci.nodes[expr].kind != Kind::Phi {
            self.ci.nodes[expr].depth = self.ci.loop_depth;
            self.color_expr(expr);
        }
        self.use_colored_expr(expr)
    }

    fn color_expr(&mut self, expr: Nid) {
        match self.ci.nodes[expr].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::Return => unreachable!(),
            Kind::CInt { .. } => self.ci.set_next_color(expr),
            Kind::Tuple { index } => {
                debug_assert!(index != 0);
            }
            Kind::BinOp { .. } => {
                let &[_, left, right] = self.ci.nodes[expr].inputs.as_slice() else {
                    unreachable!()
                };
                let lcolor = self.color_expr_consume(left);
                let rcolor = self.color_expr_consume(right);
                let color = lcolor.or(rcolor).unwrap_or_else(|| self.ci.next_color());
                self.ci.set_color(expr, color);
            }
            Kind::Call { .. } => {}
            Kind::If => todo!(),
            Kind::Region => todo!(),
            Kind::Phi => {}
            Kind::Loop => todo!(),
        }
    }

    #[must_use]
    fn use_colored_expr(&mut self, expr: Nid) -> Option<u32> {
        self.ci.nodes[expr].lock_rc += 1;
        debug_assert_ne!(self.ci.nodes[expr].color, 0, "{:?}", self.ci.nodes[expr].kind);
        (self.ci.nodes[expr].lock_rc as usize >= self.ci.nodes[expr].outputs.len()
            && self.ci.nodes[expr].depth == self.ci.loop_depth)
            .then_some(self.ci.nodes[expr].color)
    }

    fn emit_control(&mut self, mut ctrl: Nid) -> Option<Nid> {
        for _ in 0..30 {
            match self.ci.nodes[ctrl].kind {
                Kind::Start => unreachable!(),
                Kind::End => unreachable!(),
                Kind::Return => {
                    let ret = self.ci.nodes[ctrl].inputs[1];
                    if ret != 0 {
                        // NOTE: this is safer less efficient way, maybe it will be needed
                        // self.emit_expr_consume(ret);
                        // if node_color!(self, ret).call_count != self.ci.call_count {
                        //     let src = node_loc!(self, ret);
                        //     let loc = match self.tys.size_of(self.ci.ret.expect("TODO")) {
                        //         0 => Loc::default(),
                        //         1..=8 => Loc { reg: 1 },
                        //         s => todo!("{s}"),
                        //     };
                        //     if src != loc {
                        //         let inst = instrs::cp(loc.reg, src.reg);
                        //         self.ci.emit(inst);
                        //     }
                        // }

                        node_loc!(self, ret) = match self.tys.size_of(self.ci.ret.expect("TODO")) {
                            0 => Loc::default(),
                            1..=8 => Loc { reg: 1 },
                            s => todo!("{s}"),
                        };
                        self.emit_expr_consume(ret);
                    }
                    self.ci.ret_relocs.push(Reloc::new(self.ci.code.len(), 1, 4));
                    self.ci.emit(instrs::jmp(0));
                    return None;
                }
                Kind::CInt { .. } => unreachable!(),
                Kind::Tuple { .. } => {
                    ctrl = self.ci.nodes[ctrl].outputs[0];
                }
                Kind::BinOp { .. } => unreachable!(),
                Kind::Call { func } => {
                    let ret = self.tof(ctrl);

                    let mut parama = self.tys.parama(ret);
                    for i in 1..self.ci.nodes[ctrl].inputs.len() {
                        let arg = self.ci.nodes[ctrl].inputs[i];

                        let dst = match self.tys.size_of(self.tof(arg)) {
                            0 => continue,
                            1..=8 => Loc { reg: parama.next() },
                            s => todo!("{s}"),
                        };
                        self.emit_expr_consume(arg);
                        self.ci.emit(instrs::cp(dst.reg, node_loc!(self, arg).reg));
                    }

                    let reloc = Reloc::new(self.ci.code.len(), 3, 4);
                    self.ci
                        .relocs
                        .push(TypedReloc { target: ty::Kind::Func(func).compress(), reloc });
                    self.ci.emit(instrs::jal(reg::RET_ADDR, reg::ZERO, 0));

                    self.ci.call_count -= 1;

                    'b: {
                        let ret_loc = match self.tys.size_of(ret) {
                            0 => break 'b,
                            1..=8 => Loc { reg: 1 },
                            s => todo!("{s}"),
                        };

                        if self.ci.nodes[ctrl].outputs.len() == 1 {
                            break 'b;
                        }

                        if self.ci.call_count == 0 {
                            node_loc!(self, ctrl) = ret_loc;
                        } else {
                            self.emit_pass_low(ret_loc, ctrl);
                        }
                    }

                    ctrl = *self.ci.nodes[ctrl]
                        .outputs
                        .iter()
                        .find(|&&o| self.ci.nodes.is_cfg(o))
                        .unwrap();
                }
                Kind::If => {
                    let cond = self.ci.nodes[ctrl].inputs[1];

                    let jump_offset: i64;
                    let mut swapped = false;
                    'resolve_cond: {
                        'optimize_cond: {
                            let Kind::BinOp { op } = self.ci.nodes[cond].kind else {
                                break 'optimize_cond;
                            };

                            let &[_, left, right] = self.ci.nodes[cond].inputs.as_slice() else {
                                unreachable!()
                            };

                            let Some((op, swpd)) =
                                Self::cond_op(op, self.ci.nodes[left].ty.is_signed())
                            else {
                                break 'optimize_cond;
                            };
                            swapped = swpd;

                            self.emit_expr_consume(left);
                            self.emit_expr_consume(right);

                            jump_offset = self.ci.code.len() as _;
                            self.ci.emit(op(
                                node_loc!(self, left).reg,
                                node_loc!(self, right).reg,
                                0,
                            ));

                            break 'resolve_cond;
                        }

                        self.emit_expr_consume(cond);
                        jump_offset = self.ci.code.len() as _;
                        self.ci.emit(instrs::jeq(node_loc!(self, cond).reg, reg::ZERO, 0));
                    }

                    let [loff, roff] = [swapped as usize, !swapped as usize];

                    let filled_base = self.ci.filled.len();
                    let left_unreachable = self.emit_control(self.ci.nodes[ctrl].outputs[loff]);
                    for fld in self.ci.filled.drain(filled_base..) {
                        self.ci.nodes[fld].depth = 0;
                    }
                    let mut skip_then_offset = self.ci.code.len() as i64;
                    if let Some(region) = left_unreachable {
                        for i in 0..self.ci.nodes[region].outputs.len() {
                            let o = self.ci.nodes[region].outputs[i];
                            if self.ci.nodes[o].kind != Kind::Phi {
                                continue;
                            }
                            let out = self.ci.nodes[o].inputs[1 + loff];
                            self.emit_expr_consume(out);
                            self.emit_pass(out, o);
                        }

                        skip_then_offset = self.ci.code.len() as i64;
                        self.ci.emit(instrs::jmp(0));
                    }

                    let right_base = self.ci.code.len();
                    let filled_base = self.ci.filled.len();
                    let right_unreachable = self.emit_control(self.ci.nodes[ctrl].outputs[roff]);

                    for fld in self.ci.filled.drain(filled_base..) {
                        self.ci.nodes[fld].depth = 0;
                    }
                    if let Some(region) = left_unreachable {
                        for i in 0..self.ci.nodes[region].outputs.len() {
                            let o = self.ci.nodes[region].outputs[i];
                            if self.ci.nodes[o].kind != Kind::Phi {
                                continue;
                            }
                            let out = self.ci.nodes[o].inputs[1 + roff];
                            self.emit_expr_consume(out);
                            self.emit_pass(out, o);
                        }

                        let right_end = self.ci.code.len();
                        if right_base == right_end {
                            self.ci.code.truncate(skip_then_offset as _);
                        } else {
                            write_reloc(
                                &mut self.ci.code,
                                skip_then_offset as usize + 1,
                                right_end as i64 - skip_then_offset,
                                4,
                            );
                            skip_then_offset += instrs::jmp(69).0 as i64;
                        }
                    }

                    write_reloc(
                        &mut self.ci.code,
                        jump_offset as usize + 3,
                        skip_then_offset - jump_offset,
                        2,
                    );

                    let dest = left_unreachable.or(right_unreachable)?;

                    if self.ci.nodes[dest].kind == Kind::Loop {
                        return Some(dest);
                    }

                    debug_assert_eq!(self.ci.nodes[dest].kind, Kind::Region);

                    ctrl = *self.ci.nodes[dest]
                        .outputs
                        .iter()
                        .find(|&&o| self.ci.nodes[o].kind != Kind::Phi)
                        .unwrap();
                }
                Kind::Region => return Some(ctrl),
                Kind::Phi => todo!(),
                Kind::Loop => {
                    if self.ci.nodes[ctrl].lock_rc != 0 {
                        return Some(ctrl);
                    }

                    for i in 0..self.ci.nodes[ctrl].outputs.len() {
                        let o = self.ci.nodes[ctrl].outputs[i];
                        if self.ci.nodes[o].kind != Kind::Phi {
                            continue;
                        }
                        let out = self.ci.nodes[o].inputs[1];
                        self.emit_expr_consume(out);
                        self.emit_pass(out, o);
                    }

                    self.ci.nodes[ctrl].lock_rc = self.ci.code.len() as _;
                    self.ci.loop_depth += 1;

                    let end = self.emit_control(
                        *self.ci.nodes[ctrl]
                            .outputs
                            .iter()
                            .find(|&&o| self.ci.nodes[o].kind != Kind::Phi)
                            .unwrap(),
                    );

                    debug_assert_eq!(end, Some(ctrl));

                    for i in 0..self.ci.nodes[ctrl].outputs.len() {
                        let o = self.ci.nodes[ctrl].outputs[i];
                        if self.ci.nodes[o].kind != Kind::Phi {
                            continue;
                        }
                        let out = self.ci.nodes[o].inputs[2];
                        // TODO: this can be improved if we juggle ownership of the Phi inputs
                        self.emit_expr(out);
                    }

                    for i in 0..self.ci.nodes[ctrl].outputs.len() {
                        let o = self.ci.nodes[ctrl].outputs[i];
                        if self.ci.nodes[o].kind != Kind::Phi {
                            continue;
                        }
                        let out = self.ci.nodes[o].inputs[2];
                        self.use_expr(out);
                        self.emit_pass(out, o);
                    }

                    self.ci.emit(instrs::jmp(
                        self.ci.nodes[ctrl].lock_rc as i32 - self.ci.code.len() as i32,
                    ));

                    self.ci.loop_depth -= 1;
                    for free in self.ci.delayed_frees.extract_if(|&mut color| {
                        self.ci.colors[color as usize].depth == self.ci.loop_depth
                    }) {
                        let color = &self.ci.colors[free as usize];
                        debug_assert_ne!(color.loc, Loc::default());
                        self.ci.regs.free(color.loc.reg);
                    }

                    return None;
                }
            }
        }

        unreachable!()
    }

    fn emit_expr_consume(&mut self, expr: Nid) {
        self.emit_expr(expr);
        self.use_expr(expr);
    }

    fn emit_expr(&mut self, expr: Nid) {
        if self.ci.nodes[expr].depth == u32::MAX {
            return;
        }
        self.ci.nodes[expr].depth = u32::MAX;
        self.ci.filled.push(expr);

        match self.ci.nodes[expr].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::Return => unreachable!(),
            Kind::CInt { value } => {
                _ = self.lazy_init(expr);
                let instr = instrs::li64(node_loc!(self, expr).reg, value as _);
                self.ci.emit(instr);
            }
            Kind::Tuple { index } => {
                debug_assert!(index != 0);

                _ = self.lazy_init(expr);

                let mut params = self.tys.parama(self.ci.ret.unwrap());
                for (i, var) in self.ci.vars.iter().enumerate() {
                    if var.id == u32::MAX {
                        match self.tys.size_of(ty::Id::from_bt(var.value)) {
                            0 => {}
                            1..=8 => _ = params.next(),
                            s => todo!("{s}"),
                        }
                        continue;
                    }

                    match self.tys.size_of(self.ci.nodes[var.value].ty) {
                        0 => {}
                        1..=8 => {
                            let reg = params.next();
                            if i == index as usize - 1 {
                                emit(&mut self.ci.code, instrs::cp(node_loc!(self, expr).reg, reg));
                            }
                        }
                        s => todo!("{s}"),
                    }
                }
            }
            Kind::BinOp { op } => {
                _ = self.lazy_init(expr);
                let ty = self.tof(expr);
                let &[_, left, right] = self.ci.nodes[expr].inputs.as_slice() else {
                    unreachable!()
                };
                self.emit_expr_consume(left);

                if let Kind::CInt { value } = self.ci.nodes[right].kind
                    && (node_loc!(self, right) == Loc::default()
                        || self.ci.nodes[right].depth != u32::MAX)
                    && let Some(op) = Self::imm_math_op(op, ty.is_signed(), self.tys.size_of(ty))
                {
                    let instr =
                        op(node_loc!(self, expr).reg, node_loc!(self, left).reg, value as _);
                    self.ci.emit(instr);
                } else {
                    self.emit_expr_consume(right);

                    let op = Self::math_op(op, ty.is_signed(), self.tys.size_of(ty))
                        .expect("TODO: what now?");

                    let instr = op(
                        node_loc!(self, expr).reg,
                        node_loc!(self, left).reg,
                        node_loc!(self, right).reg,
                    );
                    self.ci.emit(instr);
                }
            }
            Kind::Call { .. } => {}
            Kind::If => todo!(),
            Kind::Region => todo!(),
            Kind::Phi => {}
            Kind::Loop => todo!(),
        }
    }

    fn use_expr(&mut self, expr: Nid) {
        let node = &mut self.ci.nodes[expr];
        node.lock_rc = node.lock_rc.saturating_sub(1);
        if node.lock_rc != 0 {
            return;
        }

        let color = &mut self.ci.colors[node.color as usize - 1];
        color.rc -= 1;
        if color.rc == 0 {
            if color.depth != self.ci.loop_depth {
                self.ci.delayed_frees.push(node.color);
            } else if color.loc != Loc::default() {
                self.ci.regs.free(color.loc.reg);
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn imm_math_op(
        op: TokenKind,
        signed: bool,
        size: u32,
    ) -> Option<fn(u8, u8, u64) -> (usize, [u8; instrs::MAX_SIZE])> {
        use {instrs::*, TokenKind as T};

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

        def_op!(basic_op | a, b, c | a, b, c as _);
        def_op!(sub_op | a, b, c | a, b, c.wrapping_neg() as _);

        let ops = match op {
            T::Add => basic_op!(addi8, addi16, addi32, addi64),
            T::Sub => sub_op!(addi8, addi16, addi32, addi64),
            T::Mul => basic_op!(muli8, muli16, muli32, muli64),
            T::Band => return Some(andi),
            T::Bor => return Some(ori),
            T::Xor => return Some(xori),
            T::Shr if signed => basic_op!(srui8, srui16, srui32, srui64),
            T::Shr => basic_op!(srui8, srui16, srui32, srui64),
            T::Shl => basic_op!(slui8, slui16, slui32, slui64),
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
    }

    #[allow(clippy::type_complexity)]
    fn cond_op(
        op: TokenKind,
        signed: bool,
    ) -> Option<(fn(u8, u8, i16) -> (usize, [u8; instrs::MAX_SIZE]), bool)> {
        Some((
            match op {
                TokenKind::Le if signed => instrs::jgts,
                TokenKind::Le => instrs::jgtu,
                TokenKind::Lt if signed => instrs::jlts,
                TokenKind::Lt => instrs::jltu,
                TokenKind::Eq => instrs::jne,
                TokenKind::Ne => instrs::jeq,
                _ => return None,
            },
            matches!(op, TokenKind::Lt | TokenKind::Gt),
        ))
    }

    #[allow(clippy::type_complexity)]
    fn math_op(
        op: TokenKind,
        signed: bool,
        size: u32,
    ) -> Option<fn(u8, u8, u8) -> (usize, [u8; instrs::MAX_SIZE])> {
        use {instrs::*, TokenKind as T};

        macro_rules! div { ($($op:ident),*) => {[$(|a, b, c| $op(a, reg::ZERO, b, c)),*]}; }
        macro_rules! rem { ($($op:ident),*) => {[$(|a, b, c| $op(reg::ZERO, a, b, c)),*]}; }

        let ops = match op {
            T::Add => [add8, add16, add32, add64],
            T::Sub => [sub8, sub16, sub32, sub64],
            T::Mul => [mul8, mul16, mul32, mul64],
            T::Div if signed => div!(dirs8, dirs16, dirs32, dirs64),
            T::Div => div!(diru8, diru16, diru32, diru64),
            T::Mod if signed => rem!(dirs8, dirs16, dirs32, dirs64),
            T::Mod => rem!(diru8, diru16, diru32, diru64),
            T::Band => return Some(and),
            T::Bor => return Some(or),
            T::Xor => return Some(xor),
            T::Shl => [slu8, slu16, slu32, slu64],
            T::Shr if signed => [srs8, srs16, srs32, srs64],
            T::Shr => [sru8, sru16, sru32, sru64],
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
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
        let str = &self.cfile().file;
        let (line, mut col) = lexer::line_col(str.as_bytes(), pos);
        _ = writeln!(out, "{}:{}:{}: {}", self.cfile().path, line, col, msg);

        let line = &str[str[..pos as usize].rfind('\n').map_or(0, |i| i + 1)
            ..str[pos as usize..].find('\n').unwrap_or(str.len()) + pos as usize];
        col += line.matches('\t').count() * 3;

        _ = writeln!(out, "{}", line.replace("\t", "    "));
        _ = writeln!(out, "{}^", " ".repeat(col - 1));
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

    #[must_use]
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
        fn idepth(nodes: &mut Nodes, target: Nid) -> u32 {
            if target == 0 {
                return 0;
            }
            if nodes[target].depth == 0 {
                let dm = idom(nodes, target);
                nodes[target].depth = idepth(nodes, dm) + 1;
            }
            nodes[target].depth
        }

        fn idom(nodes: &mut Nodes, target: Nid) -> Nid {
            match nodes[target].kind {
                Kind::Start => 0,
                Kind::End => unreachable!(),
                Kind::Loop
                | Kind::CInt { .. }
                | Kind::BinOp { .. }
                | Kind::Call { .. }
                | Kind::Phi
                | Kind::Tuple { .. }
                | Kind::Return
                | Kind::If => nodes[target].inputs[0],
                Kind::Region => {
                    let &[mut lcfg, mut rcfg] = nodes[target].inputs.as_slice() else {
                        unreachable!()
                    };

                    while lcfg != rcfg {
                        let [ldepth, rdepth] = [idepth(nodes, lcfg), idepth(nodes, rcfg)];
                        if ldepth >= rdepth {
                            lcfg = idom(nodes, lcfg);
                        }
                        if ldepth <= rdepth {
                            rcfg = idom(nodes, rcfg);
                        }
                    }

                    lcfg
                }
            }
        }

        fn push_up(nodes: &mut Nodes, node: Nid) {
            if !nodes.visited.set(node as _) {
                return;
            }

            if nodes[node].kind.is_pinned() {
                for i in 0..nodes[node].inputs.len() {
                    let i = nodes[node].inputs[i];
                    push_up(nodes, i);
                }
            } else {
                let mut max = 0;
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

                if max == 0 {
                    return;
                }

                let index = nodes[0].outputs.iter().position(|&p| p == node).unwrap();
                nodes[0].outputs.remove(index);
                nodes[node].inputs[0] = max;
                debug_assert!(
                    !nodes[max].outputs.contains(&node)
                        || matches!(nodes[max].kind, Kind::Call { .. }),
                    "{node} {:?} {max} {:?}",
                    nodes[node],
                    nodes[max]
                );
                nodes[max].outputs.push(node);
            }
        }

        fn push_down(nodes: &mut Nodes, node: Nid, lowest_pos: &mut [u32]) {
            if !nodes.visited.set(node as _) {
                return;
            }

            // TODO: handle memory nodes first

            if nodes[node].kind.is_pinned() {
                for i in 0..nodes[node].inputs.len() {
                    let i = nodes[node].inputs[i];
                    push_up(nodes, i);
                }
            } else {
                let mut max = 0;
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

                if max == 0 {
                    return;
                }

                let index = nodes[0].outputs.iter().position(|&p| p == node).unwrap();
                nodes[0].outputs.remove(index);
                nodes[node].inputs[0] = max;
                debug_assert!(
                    !nodes[max].outputs.contains(&node)
                        || matches!(nodes[max].kind, Kind::Call { .. }),
                    "{node} {:?} {max} {:?}",
                    nodes[node],
                    nodes[max]
                );
                nodes[max].outputs.push(node);
            }
        }

        self.ci.nodes.visited.clear(self.ci.nodes.values.len());
        push_up(&mut self.ci.nodes, self.ci.end);
        // TODO: handle infinte loops
        self.ci.nodes.visited.clear(self.ci.nodes.values.len());
        //push_down(&mut self.ci.nodes, self.ci.start);
    }
}

#[derive(Default)]
pub struct LoggedMem {
    pub mem: hbvm::mem::HostMemory,
}

impl hbvm::mem::Memory for LoggedMem {
    unsafe fn load(
        &mut self,
        addr: hbvm::mem::Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), hbvm::mem::LoadError> {
        log::trc!(
            "load: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(addr.get() as *const u8, count)
                .iter()
                .rev()
                .map(|&b| format!("{b:02x}"))
                .collect::<String>()
        );
        self.mem.load(addr, target, count)
    }

    unsafe fn store(
        &mut self,
        addr: hbvm::mem::Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), hbvm::mem::StoreError> {
        log::trc!(
            "store: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(source, count)
                .iter()
                .rev()
                .map(|&b| format!("{b:02x}"))
                .collect::<String>()
        );
        self.mem.store(addr, source, count)
    }

    unsafe fn prog_read<T: Copy>(&mut self, addr: hbvm::mem::Address) -> T {
        log::trc!(
            "read-typed: {:x} {} {:?}",
            addr.get(),
            std::any::type_name::<T>(),
            if core::mem::size_of::<T>() == 1
                && let Some(nm) =
                    instrs::NAMES.get(std::ptr::read(addr.get() as *const u8) as usize)
            {
                nm.to_string()
            } else {
                core::slice::from_raw_parts(addr.get() as *const u8, core::mem::size_of::<T>())
                    .iter()
                    .map(|&b| format!("{:02x}", b))
                    .collect::<String>()
            }
        );
        self.mem.prog_read(addr)
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::{
            parser::{self, FileId},
            son::LoggedMem,
        },
        std::io,
    };

    const README: &str = include_str!("../README.md");

    fn generate(ident: &'static str, input: &'static str, output: &mut String) {
        fn find_block(mut input: &'static str, test_name: &'static str) -> &'static str {
            const CASE_PREFIX: &str = "#### ";
            const CASE_SUFFIX: &str = "\n```hb";
            loop {
                let Some(pos) = input.find(CASE_PREFIX) else {
                    unreachable!("test {test_name} not found");
                };

                input = unsafe { input.get_unchecked(pos + CASE_PREFIX.len()..) };
                if !input.starts_with(test_name) {
                    continue;
                }
                input = unsafe { input.get_unchecked(test_name.len()..) };
                if !input.starts_with(CASE_SUFFIX) {
                    continue;
                }
                input = unsafe { input.get_unchecked(CASE_SUFFIX.len()..) };

                let end = input.find("```").unwrap_or(input.len());
                break unsafe { input.get_unchecked(..end) };
            }
        }

        let input = find_block(input, ident);

        let mut module_map = Vec::new();
        let mut last_start = 0;
        let mut last_module_name = "test";
        for (i, m) in input.match_indices("// in module: ") {
            parser::test::format(ident, input[last_start..i].trim());
            module_map.push((last_module_name, &input[last_start..i]));
            let (module_name, _) = input[i + m.len()..].split_once('\n').unwrap();
            last_module_name = module_name;
            last_start = i + m.len() + module_name.len() + 1;
        }
        parser::test::format(ident, input[last_start..].trim());
        module_map.push((last_module_name, input[last_start..].trim()));

        let loader = |path: &str, _: &str| {
            module_map
                .iter()
                .position(|&(name, _)| name == path)
                .map(|i| i as FileId)
                .ok_or(io::Error::from(io::ErrorKind::NotFound))
        };

        let mut codegen = super::Codegen {
            files: module_map
                .iter()
                .map(|&(path, content)| parser::Ast::new(path, content.to_owned(), &loader))
                .collect(),
            ..Default::default()
        };

        codegen.generate();

        {
            let errors = codegen.errors.borrow();
            if !errors.is_empty() {
                output.push_str(&errors);
                return;
            }
        }

        let mut out = Vec::new();
        codegen.assemble(&mut out);

        let mut buf = Vec::<u8>::new();
        let err = codegen.disasm(&out, &mut buf);
        output.push_str(String::from_utf8(buf).unwrap().as_str());
        if let Err(e) = err {
            writeln!(output, "!!! asm is invalid: {e}").unwrap();
            return;
        }

        return;

        let mut stack = [0_u64; 128];

        let mut vm = unsafe {
            hbvm::Vm::<_, { 1024 * 10 }>::new(
                LoggedMem::default(),
                hbvm::mem::Address::new(out.as_ptr() as u64),
            )
        };

        vm.write_reg(super::reg::STACK_PTR, unsafe { stack.as_mut_ptr().add(stack.len()) } as u64);

        use std::fmt::Write;
        let stat = loop {
            match vm.run() {
                Ok(hbvm::VmRunOk::End) => break Ok(()),
                Ok(hbvm::VmRunOk::Ecall) => match vm.read_reg(2).0 {
                    1 => writeln!(output, "ev: Ecall").unwrap(), // compatibility with a test
                    69 => {
                        let [size, align] = [vm.read_reg(3).0 as usize, vm.read_reg(4).0 as usize];
                        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
                        let ptr = unsafe { std::alloc::alloc(layout) };
                        vm.write_reg(1, ptr as u64);
                    }
                    96 => {
                        let [ptr, size, align] = [
                            vm.read_reg(3).0 as usize,
                            vm.read_reg(4).0 as usize,
                            vm.read_reg(5).0 as usize,
                        ];

                        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
                        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
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

        writeln!(output, "code size: {}", out.len()).unwrap();
        writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
        writeln!(output, "status: {:?}", stat).unwrap();
    }

    crate::run_tests! { generate:
        arithmetic => README;
        variables => README;
        functions => README;
        comments => README;
        if_statements => README;
        loops => README;
        //fb_driver => README;
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
        //exhaustive_loop_testing => README;
    }
}
