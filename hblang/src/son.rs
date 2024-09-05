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
        collections::BTreeMap,
        mem,
        ops::{self, Range},
        rc::{self, Rc},
        u32,
    },
};

type Nid = u32;
const NILL: u32 = u32::MAX;

mod reg {
    use crate::log;

    pub const STACK_PTR: Reg = 254;
    pub const ZERO: Reg = 0;
    pub const RET: Reg = 1;
    pub const RET_ADDR: Reg = 31;

    type Reg = u8;

    #[derive(Default, Debug, PartialEq, Eq)]
    pub struct Id(Reg, bool);

    impl Id {
        pub const RET: Self = Id(RET, false);

        pub fn get(&self) -> Reg {
            self.0
        }

        pub fn as_ref(&self) -> Self {
            Self(self.0, false)
        }

        pub fn is_ref(&self) -> bool {
            !self.1
        }
    }

    impl From<u8> for Id {
        fn from(value: u8) -> Self {
            Self(value, false)
        }
    }

    impl Drop for Id {
        fn drop(&mut self) {
            if !std::thread::panicking() && self.1 {
                log::err!("reg id leaked: {:?}", self.0);
            }
        }
    }

    #[derive(Default, PartialEq, Eq)]
    pub struct Alloc {
        free: Vec<Reg>,
        max_used: Reg,
    }

    impl Alloc {
        pub fn init(&mut self) {
            self.free.clear();
            self.free.extend((32..=253).rev());
            self.max_used = RET_ADDR;
        }

        pub fn allocate(&mut self) -> Id {
            let reg = self.free.pop().expect("TODO: we need to spill");
            self.max_used = self.max_used.max(reg);
            Id(reg, true)
        }

        pub fn free(&mut self, reg: Id) {
            if reg.1 {
                self.free.push(reg.0);
                std::mem::forget(reg);
            }
        }

        pub fn pushed_size(&self) -> usize {
            ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
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

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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

struct Nodes {
    values: Vec<PoolSlot>,
    free: u32,
    lookup: HashMap<(Kind, [Nid; MAX_INPUTS]), Nid>,
}

impl Default for Nodes {
    fn default() -> Self {
        Self { values: Default::default(), free: u32::MAX, lookup: Default::default() }
    }
}

impl Nodes {
    fn add(&mut self, value: Node) -> u32 {
        if self.free == u32::MAX {
            self.free = self.values.len() as _;
            self.values.push(PoolSlot::Next(u32::MAX));
        }

        let free = self.free;
        self.free = match mem::replace(&mut self.values[free as usize], PoolSlot::Value(value)) {
            PoolSlot::Value(_) => unreachable!(),
            PoolSlot::Next(free) => free,
        };
        free
    }

    fn remove_low(&mut self, id: u32) -> Node {
        let value = match mem::replace(&mut self.values[id as usize], PoolSlot::Next(self.free)) {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        };
        self.free = id;
        value
    }

    fn clear(&mut self) {
        self.values.clear();
        self.lookup.clear();
        self.free = u32::MAX;
    }

    fn new_node<const SIZE: usize>(
        &mut self,
        ty: impl Into<ty::Id>,
        kind: Kind,
        inps: [Nid; SIZE],
    ) -> Nid {
        let mut inputs = [NILL; MAX_INPUTS];
        inputs[..inps.len()].copy_from_slice(&inps);

        if let Some(&id) = self.lookup.get(&(kind.clone(), inputs)) {
            debug_assert_eq!(&self[id].kind, &kind);
            debug_assert_eq!(self[id].inputs, inputs);
            return id;
        }

        let id = self.add(Node {
            inputs,
            kind: kind.clone(),
            loc: Default::default(),
            depth: u32::MAX,
            lock_rc: 0,
            ty: ty.into(),
            outputs: vec![],
        });

        let prev = self.lookup.insert((kind, inputs), id);
        debug_assert_eq!(prev, None);

        self.add_deps(id, &inps);
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

    fn unlock(&mut self, target: Nid) {
        self[target].lock_rc -= 1;
    }

    fn remove(&mut self, target: Nid) {
        if !self[target].is_dangling() {
            return;
        }
        for i in 0..self[target].inputs().len() {
            let inp = self[target].inputs[i];
            let index = self[inp].outputs.iter().position(|&p| p == target).unwrap();
            self[inp].outputs.swap_remove(index);
            self.remove(inp);
        }
        let res = self.lookup.remove(&(self[target].kind.clone(), self[target].inputs));
        debug_assert_eq!(res, Some(target));
        self.remove_low(target);
    }

    fn peephole(&mut self, target: Nid) -> Option<Nid> {
        match self[target].kind {
            Kind::Start => {}
            Kind::End => {}
            Kind::BinOp { op } => return self.peephole_binop(target, op),
            Kind::Return => {}
            Kind::Tuple { .. } => {}
            Kind::ConstInt { .. } => {}
            Kind::Call { .. } => {}
            Kind::If => {}
            Kind::Region => {}
            Kind::Phi => {}
        }
        None
    }

    fn peephole_binop(&mut self, target: Nid, op: TokenKind) -> Option<Nid> {
        use TokenKind as T;
        let [mut lhs, mut rhs, ..] = self[target].inputs;

        if lhs == rhs {
            match op {
                T::Sub => {
                    return Some(self.new_node(self[target].ty, Kind::ConstInt { value: 0 }, []));
                }
                T::Add => {
                    let rhs = self.new_node(self[target].ty, Kind::ConstInt { value: 2 }, []);
                    return Some(
                        self.new_node(self[target].ty, Kind::BinOp { op: T::Mul }, [lhs, rhs]),
                    );
                }
                _ => {}
            }
        }

        if let (&Kind::ConstInt { value: a }, &Kind::ConstInt { value: b }) =
            (&self[lhs].kind, &self[rhs].kind)
        {
            return Some(self.new_node(
                self[target].ty,
                Kind::ConstInt { value: op.apply(a, b) },
                [],
            ));
        }

        let mut changed = false;
        if op.is_comutative() && self[lhs].kind < self[rhs].kind {
            std::mem::swap(&mut lhs, &mut rhs);
            changed = true;
        }

        if let Kind::ConstInt { value } = self[rhs].kind {
            match (op, value) {
                (T::Add | T::Sub | T::Shl, 0) | (T::Mul | T::Div, 1) => return Some(lhs),
                (T::Mul, 0) => return Some(rhs),
                _ => {}
            }
        }

        if op.is_comutative() && self[lhs].kind == (Kind::BinOp { op }) {
            if let Kind::ConstInt { value: a } = self[self[lhs].inputs[1]].kind
                && let Kind::ConstInt { value: b } = self[rhs].kind
            {
                let new_rhs =
                    self.new_node(self[target].ty, Kind::ConstInt { value: op.apply(a, b) }, []);
                return Some(self.new_node(self[target].ty, Kind::BinOp { op }, [
                    self[lhs].inputs[0],
                    new_rhs,
                ]));
            }

            if self.is_const(self[lhs].inputs[1]) {
                let new_lhs =
                    self.new_node(self[target].ty, Kind::BinOp { op }, [self[lhs].inputs[0], rhs]);
                return Some(self.new_node(self[target].ty, Kind::BinOp { op }, [
                    new_lhs,
                    self[lhs].inputs[1],
                ]));
            }
        }

        if op == T::Add
            && self[lhs].kind == (Kind::BinOp { op: T::Mul })
            && self[lhs].inputs[0] == rhs
            && let Kind::ConstInt { value } = self[self[lhs].inputs[1]].kind
        {
            let new_rhs = self.new_node(self[target].ty, Kind::ConstInt { value: value + 1 }, []);
            return Some(
                self.new_node(self[target].ty, Kind::BinOp { op: T::Mul }, [rhs, new_rhs]),
            );
        }

        if op == T::Sub && self[lhs].kind == (Kind::BinOp { op }) {
            // (a - b) - c => a - (b + c)
            let [a, b, ..] = self[lhs].inputs;
            let c = rhs;
            let new_rhs = self.new_node(self[target].ty, Kind::BinOp { op: T::Add }, [b, c]);
            return Some(self.new_node(self[target].ty, Kind::BinOp { op }, [a, new_rhs]));
        }

        if changed {
            return Some(self.new_node(self[target].ty, self[target].kind.clone(), [lhs, rhs]));
        }

        None
    }

    fn is_const(&self, id: Nid) -> bool {
        matches!(self[id].kind, Kind::ConstInt { .. })
    }

    fn replace(&mut self, target: Nid, with: Nid) {
        //for i in 0..self[target].inputs().len() {
        //    let inp = self[target].inputs[i];
        //    let index = self[inp].outputs.iter().position(|&p| p == target).unwrap();
        //    self[inp].outputs[index] = with;
        //}

        for i in 0..self[target].outputs.len() {
            let out = self[target].outputs[i];
            let index = self[out].inputs().iter().position(|&p| p == target).unwrap();
            let rpl = self.modify_input(out, index, with);
            self[with].outputs.push(rpl);
        }

        self.remove_low(target);
    }

    fn modify_input(&mut self, target: Nid, inp_index: usize, with: Nid) -> Nid {
        let out = self.lookup.remove(&(self[target].kind.clone(), self[target].inputs));
        debug_assert!(out == Some(target));
        debug_assert_ne!(self[target].inputs[inp_index], with);

        self[target].inputs[inp_index] = with;
        if let Err(other) =
            self.lookup.try_insert((self[target].kind.clone(), self[target].inputs), target)
        {
            let rpl = *other.entry.get();
            self.replace(target, rpl);
            return rpl;
        }

        target
    }

    fn add_deps(&mut self, id: Nid, deps: &[Nid]) {
        for &d in deps {
            debug_assert_ne!(d, id);
            self[d].outputs.push(id);
        }
    }

    fn unlock_free(&mut self, id: Nid) {
        self[id].lock_rc -= 1;
        if self[id].is_dangling() {
            self.remove_low(id);
        }
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
                if self[node].inputs[2] != NILL {
                    self.fmt(f, self[node].inputs[2], rcs)?;
                }
                writeln!(f)?;
                self.fmt(f, self[node].inputs[1], rcs)?;
            }
            Kind::ConstInt { value } => write!(f, "{}", value)?,
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
            Kind::Call { func, ref args } => {
                if is_ready() {
                    write!(f, "{}: call {}(", node, func)?;
                    for (i, &value) in args.iter().enumerate() {
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
        }

        Ok(())
    }

    fn is_cfg(&self, o: Nid) -> bool {
        matches!(
            self[o].kind,
            Kind::Start | Kind::End | Kind::Return | Kind::Tuple { .. } | Kind::Call { .. }
        )
    }

    fn check_final_integrity(&self) {
        for slot in &self.values {
            match slot {
                PoolSlot::Value(node) => {
                    debug_assert_eq!(node.lock_rc, 0);
                }
                PoolSlot::Next(_) => {}
            }
        }
    }
}

impl ops::Index<u32> for Nodes {
    type Output = Node;

    fn index(&self, index: u32) -> &Self::Output {
        match &self.values[index as usize] {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        }
    }
}

impl ops::IndexMut<u32> for Nodes {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match &mut self.values[index as usize] {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        }
    }
}

#[derive(Debug)]
enum PoolSlot {
    Value(Node),
    Next(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Start,
    End,
    If,
    Region,
    Return,
    ConstInt { value: i64 },
    Phi,
    Tuple { index: u32 },
    BinOp { op: lexer::TokenKind },
    Call { func: ty::Func, args: Vec<Nid> },
}

impl Kind {
    fn disc(&self) -> u8 {
        unsafe { *(self as *const _ as *const u8) }
    }
}

const MAX_INPUTS: usize = 3;

#[derive(Debug)]
struct Node {
    inputs: [Nid; MAX_INPUTS],
    kind: Kind,
    loc: Loc,
    depth: u32,
    lock_rc: u32,
    ty: ty::Id,
    outputs: Vec<Nid>,
}

impl Node {
    fn is_dangling(&self) -> bool {
        self.outputs.len() + self.lock_rc as usize == 0
    }

    fn inputs(&self) -> &[Nid] {
        let len = self.inputs.iter().position(|&n| n == NILL).unwrap_or(MAX_INPUTS);
        &self.inputs[..len]
    }

    fn inputs_mut(&mut self) -> &mut [Nid] {
        let len = self.inputs.iter().position(|&n| n == NILL).unwrap_or(MAX_INPUTS);
        &mut self.inputs[..len]
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
                    PoolSlot::Value(Node { kind: Kind::Start, .. }) => 1,
                    PoolSlot::Value(Node { kind: Kind::End, ref outputs, .. }) => outputs.len(),
                    PoolSlot::Value(val) => val.inputs().len(),
                    PoolSlot::Next(_) => 0,
                })
                .collect::<Vec<_>>(),
        )
    }
}

type Offset = u32;
type Size = u32;
type ArrayLen = u32;

#[derive(Debug)]
struct Loop {
    var_count: u32,
    offset: u32,
    break_relocs: Vec<Reloc>,
}

#[derive(Clone, Copy)]
struct Variable {
    id: Ident,
    value: Nid,
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

    call_count: usize,

    loops: Vec<Loop>,
    vars: Vec<Variable>,
    regs: reg::Alloc,
    ret_relocs: Vec<Reloc>,
    relocs: Vec<TypedReloc>,
    code: Vec<u8>,
}

impl ItemCtx {
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

#[derive(Debug, Default)]
struct Loc {
    reg: reg::Id,
}
impl Loc {
    fn as_ref(&self) -> Loc {
        Loc { reg: self.reg.as_ref() }
    }
}

#[derive(Default, Debug)]
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
            Expr::Comment { .. } => Some(NILL),
            Expr::Ident { pos, id, .. } => Some(
                self.ci
                    .vars
                    .iter()
                    .find(|v| v.id == id)
                    .unwrap_or_else(|| self.report(pos, msg))
                    .value,
            ),
            Expr::BinOp { left: &Expr::Ident { id, .. }, op: TokenKind::Decl, right } => {
                let value = self.expr(right)?;
                self.ci.nodes.lock(value);
                self.ci.vars.push(Variable { id, value });
                Some(NILL)
            }
            Expr::BinOp { left: &Expr::Ident { id, pos, .. }, op: TokenKind::Assign, right } => {
                let value = self.expr(right)?;
                self.ci.nodes.lock(value);

                let Some(var) = self.ci.vars.iter_mut().find(|v| v.id == id) else {
                    self.report(pos, msg);
                };

                let prev = std::mem::replace(&mut var.value, value);
                self.ci.nodes.unlock_free(prev);
                Some(NILL)
            }
            Expr::BinOp { left, op, right } => {
                let lhs = self.expr_ctx(left, ctx)?;
                self.ci.nodes.lock(lhs);
                let rhs = self.expr_ctx(right, Ctx::default().with_ty(self.tof(lhs)));
                self.ci.nodes.unlock(lhs);
                let rhs = rhs?;
                let ty = self.assert_ty(left.pos(), self.tof(rhs), self.tof(lhs), false);
                let id =
                    self.ci.nodes.new_node(ty::bin_ret(ty, op), Kind::BinOp { op }, [lhs, rhs]);
                Some(id)
            }
            Expr::If { pos, cond, then, else_ } => {
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
                        return Some(NILL);
                    }
                }

                let else_scope = self.ci.vars.clone();

                self.ci.ctrl =
                    self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [if_node]);
                let lcntrl = self.expr(then).map_or(NILL, |_| self.ci.ctrl);

                let then_scope = std::mem::replace(&mut self.ci.vars, else_scope);
                self.ci.ctrl =
                    self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 1 }, [if_node]);
                let rcntrl = if let Some(else_) = else_ {
                    self.expr(else_).map_or(NILL, |_| self.ci.ctrl)
                } else {
                    self.ci.ctrl
                };

                if lcntrl == NILL && rcntrl == NILL {
                    return None;
                } else if lcntrl == NILL {
                    return Some(NILL);
                } else if rcntrl == NILL {
                    self.ci.vars = then_scope;
                    return Some(NILL);
                }

                let region = self.ci.nodes.new_node(ty::VOID, Kind::Region, [lcntrl, rcntrl]);

                for (else_var, then_var) in self.ci.vars.iter_mut().zip(then_scope) {
                    if else_var.value == then_var.value {
                        continue;
                    }

                    let ty = self.ci.nodes[else_var.value].ty;
                    debug_assert_eq!(
                        ty, self.ci.nodes[then_var.value].ty,
                        "TODO: typecheck properly"
                    );

                    let inps = [region, then_var.value, else_var.value];
                    else_var.value = self.ci.nodes.new_node(ty, Kind::Phi, inps);
                }

                Some(NILL)
            }
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
                };

                let fuc = &self.tys.funcs[func as usize];
                let sig = fuc.sig.expect("TODO: generic functions");
                let ast = self.files[fuc.file as usize].clone();
                let Expr::BinOp { right: &Expr::Closure { args: cargs, .. }, .. } =
                    fuc.expr.get(&ast).unwrap()
                else {
                    unreachable!()
                };

                if args.len() != cargs.len() {
                    let s = if cargs.len() == 1 { "" } else { "s" };
                    self.report(
                        pos,
                        format_args!(
                            "expected {} function argumenr{s}, got {}",
                            cargs.len(),
                            args.len()
                        ),
                    );
                }

                let mut inps = vec![];
                for ((arg, _carg), tyx) in args.iter().zip(cargs).zip(sig.args.range()) {
                    let ty = self.tys.args[tyx];
                    if self.tys.size_of(ty) == 0 {
                        continue;
                    }
                    let value = self.expr_ctx(arg, Ctx::default().with_ty(ty))?;
                    _ = self.assert_ty(arg.pos(), self.tof(value), ty, true);
                    inps.push(value);
                }
                self.ci.ctrl =
                    self.ci
                        .nodes
                        .new_node(sig.ret, Kind::Call { func, args: inps.clone() }, [self.ci.ctrl]);
                self.ci.nodes.add_deps(self.ci.ctrl, &inps);

                Some(self.ci.ctrl)
            }
            Expr::Return { pos, val } => {
                let (ty, value) = if let Some(val) = val {
                    let value = self.expr_ctx(val, Ctx { ty: self.ci.ret })?;

                    (self.tof(value), value)
                } else {
                    (ty::VOID.into(), NILL)
                };

                if value == NILL {
                    let inps = [self.ci.ctrl, self.ci.end];
                    self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Return, inps);
                } else {
                    let inps = [self.ci.ctrl, self.ci.end, value];
                    self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Return, inps);
                }

                let expected = *self.ci.ret.get_or_insert(ty);
                _ = self.assert_ty(pos, ty, expected, true);

                None
            }
            Expr::Block { stmts, .. } => {
                let base = self.ci.vars.len();

                let mut ret = Some(NILL);
                for stmt in stmts {
                    ret = ret.and(self.expr(stmt));
                    if let Some(id) = ret {
                        _ = self.assert_ty(stmt.pos(), self.tof(id), ty::VOID.into(), true);
                    } else {
                        break;
                    }
                }

                for var in self.ci.vars.drain(base..) {
                    self.ci.nodes.unlock_free(var.value);
                }

                ret
            }
            Expr::Number { value, .. } => Some(self.ci.nodes.new_node(
                ctx.ty.unwrap_or(ty::INT.into()),
                Kind::ConstInt { value },
                [],
            )),
            ref e => self.report_unhandled_ast(e, "bruh"),
        }
    }

    #[inline(always)]
    fn tof(&self, id: Nid) -> ty::Id {
        if id == NILL {
            return ty::VOID.into();
        }
        self.ci.nodes[id].ty
    }

    //#[must_use]
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
        self.ci.end = self.ci.nodes.new_node(ty::VOID, Kind::End, []);
        self.ci.ctrl = self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [self.ci.start]);

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
            let value = self.ci.nodes.new_node(ty, Kind::Tuple { index }, [self.ci.start]);
            self.ci.nodes.lock(value);
            let sym = parser::find_symbol(&ast.symbols, arg.id);
            assert!(sym.flags & idfl::COMPTIME == 0, "TODO");
            self.ci.vars.push(Variable { id: arg.id, value });
        }

        if self.expr(body).is_some() {
            self.report(body.pos(), "expected all paths in the fucntion to return");
        }

        for var in self.ci.vars.iter() {
            self.ci.nodes.unlock(var.value);
        }

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

        '_copy_args: {
            let mut params = self.tys.parama(sig.ret);
            for var in self.ci.vars.drain(..) {
                if self.ci.nodes[var.value].outputs.is_empty() {
                    continue;
                }

                match self.tys.size_of(self.ci.nodes[var.value].ty) {
                    0 => {}
                    1..=8 => {
                        let reg = self.ci.regs.allocate();
                        emit(&mut self.ci.code, instrs::cp(reg.get(), params.next()));
                        self.ci.nodes[var.value].loc = Loc { reg };
                    }
                    s => todo!("{s}"),
                }
            }
        }

        self.emit_control(self.ci.nodes[self.ci.start].outputs[0]);

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

        self.tys.funcs[id as usize].code.append(&mut self.ci.code);
        self.tys.funcs[id as usize].relocs.append(&mut self.ci.relocs);
        self.ci.nodes.clear();
        self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
    }

    fn emit_control(&mut self, mut ctrl: Nid) -> Option<Nid> {
        loop {
            match self.ci.nodes[ctrl].kind.clone() {
                Kind::Start => unreachable!(),
                Kind::End => unreachable!(),
                Kind::Return => {
                    if let Some(&ret) = self.ci.nodes[ctrl].inputs().get(2) {
                        let ret_loc = match self.tys.size_of(self.ci.ret.expect("TODO")) {
                            0 => Loc::default(),
                            1..=8 => Loc { reg: 1u8.into() },
                            s => todo!("{s}"),
                        };
                        let dst = self.emit_expr(ret, GenCtx::default().with_loc(ret_loc));
                        self.ci.free_loc(dst);
                    }
                    self.ci.ret_relocs.push(Reloc::new(self.ci.code.len(), 1, 4));
                    self.ci.emit(instrs::jmp(0));
                    break None;
                }
                Kind::ConstInt { .. } => unreachable!(),
                Kind::Tuple { .. } => {
                    ctrl = self.ci.nodes[ctrl].outputs[0];
                }
                Kind::BinOp { .. } => unreachable!(),
                Kind::Call { func, args } => {
                    let ret = self.tof(ctrl);

                    let mut parama = self.tys.parama(ret);
                    for &arg in args.iter() {
                        let dst = match self.tys.size_of(self.tof(arg)) {
                            0 => continue,
                            1..=8 => Loc { reg: parama.next().into() },
                            s => todo!("{s}"),
                        };
                        let dst = self.emit_expr(arg, GenCtx::default().with_loc(dst));
                        self.ci.free_loc(dst);
                    }

                    let reloc = Reloc::new(self.ci.code.len(), 3, 4);
                    self.ci
                        .relocs
                        .push(TypedReloc { target: ty::Kind::Func(func).compress(), reloc });
                    self.ci.emit(instrs::jal(reg::RET_ADDR, reg::ZERO, 0));
                    self.make_func_reachable(func);

                    self.ci.call_count -= 1;

                    'b: {
                        let ret_loc = match self.tys.size_of(ret) {
                            0 => break 'b,
                            1..=8 => Loc { reg: 1u8.into() },
                            s => todo!("{s}"),
                        };

                        if self.ci.nodes[ctrl].outputs.len() == 1 {
                            break 'b;
                        }

                        let loc;
                        if self.ci.call_count != 0 {
                            loc = Loc { reg: self.ci.regs.allocate() };
                            self.ci.emit(instrs::cp(loc.reg.get(), ret_loc.reg.get()));
                        } else {
                            loc = ret_loc;
                        }

                        self.ci.nodes[ctrl].loc = loc;
                        self.ci.nodes[ctrl].lock_rc += 1;
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
                    match self.ci.nodes[cond].kind {
                        Kind::BinOp { op: TokenKind::Le } => {
                            let [left, right, ..] = self.ci.nodes[cond].inputs;
                            let ldst = self.emit_expr(left, GenCtx::default());
                            let rdst = self.emit_expr(right, GenCtx::default());
                            jump_offset = self.ci.code.len() as _;
                            let op = if self.ci.nodes[left].ty.is_signed() {
                                instrs::jgts
                            } else {
                                instrs::jgtu
                            };
                            self.ci.emit(op(
                                self.ci.nodes[left].loc.reg.get(),
                                self.ci.nodes[right].loc.reg.get(),
                                0,
                            ));
                            self.ci.free_loc(ldst);
                            self.ci.free_loc(rdst);
                        }
                        _ => {
                            let cdst = self.emit_expr(cond, GenCtx::default());
                            let cond = self.ci.nodes[cond].loc.reg.get();
                            jump_offset = self.ci.code.len() as _;
                            self.ci.emit(instrs::jeq(cond, reg::ZERO, 0));
                            self.ci.free_loc(cdst);
                        }
                    }

                    let left_unreachable = self.emit_control(self.ci.nodes[ctrl].outputs[0]);
                    let mut skip_then_offset = self.ci.code.len() as i64;
                    if let Some(region) = left_unreachable {
                        for i in 0..self.ci.nodes[region].outputs.len() {
                            let o = self.ci.nodes[region].outputs[i];
                            if self.ci.nodes[o].kind != Kind::Phi {
                                continue;
                            }
                            let out = self.ci.nodes[o].inputs[1];
                            let dst = self.emit_expr(out, GenCtx::default());
                            self.ci.nodes[o].loc = dst.unwrap_or_else(|| {
                                let reg = self.ci.regs.allocate();
                                self.ci
                                    .emit(instrs::cp(reg.get(), self.ci.nodes[out].loc.reg.get()));
                                Loc { reg }
                            });
                        }

                        self.ci.emit(instrs::jmp(0));
                    }

                    let right_base = self.ci.code.len();
                    let right_unreachable = self.emit_control(self.ci.nodes[ctrl].outputs[1]);
                    if let Some(region) = left_unreachable {
                        for i in 0..self.ci.nodes[region].outputs.len() {
                            let o = self.ci.nodes[region].outputs[i];
                            if self.ci.nodes[o].kind != Kind::Phi {
                                continue;
                            }
                            let out = self.ci.nodes[o].inputs[2];
                            // TODO: this can be improved if we juggle ownership of the Phi inputs
                            let dst = self.emit_expr(out, GenCtx::default());
                            self.ci.free_loc(dst);
                            self.ci.emit(instrs::cp(
                                self.ci.nodes[o].loc.reg.get(),
                                self.ci.nodes[out].loc.reg.get(),
                            ));
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

                    if left_unreachable.is_none() && right_unreachable.is_none() {
                        break None;
                    }

                    debug_assert_eq!(left_unreachable, right_unreachable);
                }
                Kind::Region => break Some(ctrl),
                Kind::Phi => todo!(),
            }
        }
    }

    #[must_use = "dont forget to drop the location"]
    fn emit_expr(&mut self, expr: Nid, ctx: GenCtx) -> Option<Loc> {
        match self.ci.nodes[expr].kind {
            Kind::Start => unreachable!(),
            Kind::End => unreachable!(),
            Kind::Return => unreachable!(),
            Kind::ConstInt { value } => {
                if let Some(loc) = ctx.loc {
                    self.ci.emit(instrs::li64(loc.reg.get(), value as _));
                } else {
                    let reg = self.ci.regs.allocate();
                    self.ci.emit(instrs::li64(reg.get(), value as _));
                    self.ci.nodes[expr].loc = Loc { reg };
                }
            }
            Kind::Tuple { index } => {
                debug_assert!(index != 0);
            }
            Kind::BinOp { op } => {
                let ty = self.tof(expr);
                let [left, right, ..] = self.ci.nodes[expr].inputs;
                let mut ldst = self.emit_expr(left, GenCtx::default());

                if let Kind::ConstInt { value } = self.ci.nodes[right].kind
                    && let Some(op) = Self::imm_math_op(op, ty.is_signed(), self.tys.size_of(ty))
                {
                    let loc = ctx
                        .loc
                        .or_else(|| ldst.take())
                        .unwrap_or_else(|| Loc { reg: self.ci.regs.allocate() });

                    self.ci.free_loc(ldst);

                    self.ci.emit(op(loc.reg.get(), self.ci.nodes[left].loc.reg.get(), value as _));
                    self.ci.nodes[expr].loc = loc;
                } else {
                    let mut rdst = self.emit_expr(right, GenCtx::default());

                    let op = Self::math_op(op, ty.is_signed(), self.tys.size_of(ty))
                        .expect("TODO: what now?");

                    let loc = ctx
                        .loc
                        .or_else(|| ldst.take())
                        .or_else(|| rdst.take())
                        .unwrap_or_else(|| Loc { reg: self.ci.regs.allocate() });

                    self.ci.free_loc(ldst);
                    self.ci.free_loc(rdst);

                    self.ci.emit(op(
                        loc.reg.get(),
                        self.ci.nodes[left].loc.reg.get(),
                        self.ci.nodes[right].loc.reg.get(),
                    ));
                    self.ci.nodes[expr].loc = loc;
                }
            }
            Kind::Call { .. } => {}
            Kind::If => todo!(),
            Kind::Region => todo!(),
            Kind::Phi => todo!(),
        }

        self.ci.nodes[expr].lock_rc += 1;
        if self.ci.nodes[expr].lock_rc as usize == self.ci.nodes[expr].outputs.len() {
            let re = self.ci.nodes[expr].loc.as_ref();
            return Some(std::mem::replace(&mut self.ci.nodes[expr].loc, re));
        }

        None
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
            ref e => self.report_unhandled_ast(e, "type"),
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
                    self.report(pos, format_args!("undefined indentifier: {name}"))
                }
                Err("main") => self.report(
                    pos,
                    format_args!(
                        "missing main function in '{}', compiler can't \
                        emmit libraries since such concept is not defined",
                        f.path
                    ),
                ),
                Err(name) => self.report(pos, format_args!("undefined indentifier: {name}")),
            }
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

                        let args = self.pack_args(pos, arg_base);
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
    fn assert_ty(&self, pos: Pos, ty: ty::Id, expected: ty::Id, preserve_expected: bool) -> ty::Id {
        if let Some(res) = ty.try_upcast(expected)
            && (!preserve_expected || res == expected)
        {
            res
        } else {
            let ty = self.ty_display(ty);
            let expected = self.ty_display(expected);
            self.report(pos, format_args!("expected {expected}, got {ty}"));
        }
    }

    fn report_log(&self, pos: Pos, msg: impl std::fmt::Display) {
        let (line, col) = lexer::line_col(self.cfile().file.as_bytes(), pos);
        println!("{}:{}:{}: {}", self.cfile().path, line, col, msg);
    }

    #[track_caller]
    fn report(&self, pos: Pos, msg: impl std::fmt::Display) -> ! {
        self.report_log(pos, msg);
        unreachable!();
    }

    #[track_caller]
    fn report_unhandled_ast(&self, ast: &Expr, hint: &str) -> ! {
        self.report(
            ast.pos(),
            format_args!(
                "compiler does not (yet) know how to handle ({hint}):\n\
                {ast:}\n\
                info for weak people:\n\
                {ast:#?}"
            ),
        )
    }

    fn cfile(&self) -> &parser::Ast {
        &self.files[self.ci.file as usize]
    }

    fn pack_args(&mut self, pos: Pos, arg_base: usize) -> ty::Tuple {
        let needle = &self.tys.args[arg_base..];
        if needle.is_empty() {
            return ty::Tuple::empty();
        }
        let len = needle.len();
        // FIXME: maybe later when this becomes a bottleneck we use more
        // efficient search (SIMD?, indexing?)
        let sp = self.tys.args.windows(needle.len()).position(|val| val == needle).unwrap();
        self.tys.args.truncate((sp + needle.len()).max(arg_base));
        ty::Tuple::new(sp, len)
            .unwrap_or_else(|| self.report(pos, "amount of arguments not supported"))
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

        let mut out = Vec::new();
        codegen.assemble(&mut out);

        let mut buf = Vec::<u8>::new();
        let err = codegen.disasm(&out, &mut buf);
        output.push_str(String::from_utf8(buf).unwrap().as_str());
        if let Err(e) = err {
            writeln!(output, "!!! asm is invalid: {e}").unwrap();
            return;
        }

        let mut stack = [0_u64; 128];

        let mut vm = unsafe {
            hbvm::Vm::<_, 0>::new(
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
        const_folding_with_arg => README;
        variables => README;
        functions => README;
        comments => README;
        if_statements => README;
        //loops => README;
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
        //hex_octal_binary_literals => README;
        //comptime_min_reg_leak => README;
        ////structs_in_registers => README;
        //comptime_function_from_another_file => README;
        //inline => README;
        //inline_test => README;
    }
}
