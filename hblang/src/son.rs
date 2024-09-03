#![allow(dead_code)]
use {
    crate::{
        ident::{self, Ident},
        lexer::{self, TokenKind},
        log,
        parser::{self, idfl, Expr, ExprRef, FileId, Pos},
        HashMap,
    },
    core::fmt,
    std::{
        mem,
        ops::{self, Range},
        rc::Rc,
    },
};

type Nid = u32;
const NILL: u32 = u32::MAX;

pub struct Nodes {
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
    pub fn add(&mut self, value: Node) -> u32 {
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

    pub fn remove_low(&mut self, id: u32) -> Node {
        let value = match mem::replace(&mut self.values[id as usize], PoolSlot::Next(self.free)) {
            PoolSlot::Value(value) => value,
            PoolSlot::Next(_) => unreachable!(),
        };
        self.free = id;
        value
    }

    pub fn clear(&mut self) {
        self.values.clear();
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

        if let Some(&id) = self.lookup.get(&(kind, inputs)) {
            debug_assert_eq!(self[id].kind, kind);
            debug_assert_eq!(self[id].inputs, inputs);
            return id;
        }

        let id = self.add(Node {
            inputs,
            kind,
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
        let res = self.lookup.remove(&(self[target].kind, self[target].inputs));
        debug_assert_eq!(res, Some(target));
        self.remove_low(target);
    }

    fn peephole(&mut self, target: Nid) -> Option<Nid> {
        match self[target].kind {
            Kind::Start => {}
            Kind::End => {}
            Kind::BinOp { op } => return self.peephole_binop(target, op),
            Kind::Return => {}
            Kind::Tuple { index } => {}
            Kind::ConstInt { value } => {}
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

        if let (Kind::ConstInt { value: a }, Kind::ConstInt { value: b }) =
            (self[lhs].kind, self[rhs].kind)
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
            return Some(self.new_node(self[target].ty, self[target].kind, [lhs, rhs]));
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
        let out = self.lookup.remove(&(self[target].kind, self[target].inputs));
        debug_assert!(out == Some(target));
        debug_assert_ne!(self[target].inputs[inp_index], with);

        self[target].inputs[inp_index] = with;
        if let Err(other) = self.lookup.try_insert((self[target].kind, self[target].inputs), target)
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
                self.fmt(f, self[node].inputs[1], rcs)?;
                writeln!(f)?;
                self.fmt(f, self[node].inputs[2], rcs)?;
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
        }

        Ok(())
    }

    fn is_cfg(&self, o: Nid) -> bool {
        matches!(self[o].kind, Kind::Start | Kind::End | Kind::Return | Kind::Tuple { .. })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Kind {
    Start,
    End,
    Return,
    ConstInt { value: i64 },
    Tuple { index: u32 },
    BinOp { op: lexer::TokenKind },
}

impl Kind {
    fn disc(&self) -> u8 {
        unsafe { *(self as *const _ as *const u8) }
    }
}

const MAX_INPUTS: usize = 3;

#[derive(Debug)]
pub struct Node {
    pub inputs: [Nid; MAX_INPUTS],
    pub kind: Kind,
    pub depth: u32,
    pub lock_rc: u32,
    pub ty: ty::Id,
    pub outputs: Vec<Nid>,
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

#[derive(Clone, Copy, Debug)]
struct Loop {
    var_count: u32,
    offset: u32,
    reloc_base: u32,
}

struct Variable {
    id: Ident,
    value: Nid,
}

#[derive(Default)]
struct ItemCtx {
    file: FileId,
    id: ty::Kind,
    ret: Option<ty::Id>,
    start: Nid,
    end: Nid,
    cfg: Nid,

    task_base: usize,
    snap: Snapshot,

    nodes: Nodes,
    loops: Vec<Loop>,
    vars: Vec<Variable>,
}

impl ItemCtx {}

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

#[derive(Clone, Copy)]
struct Func {
    file: FileId,
    expr: ExprRef,
    sig: Option<Sig>,
    offset: Offset,
}

struct Global {
    offset: Offset,
    ty: ty::Id,
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

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Snapshot {
    code: usize,
    string_data: usize,
    funcs: usize,
    globals: usize,
    strings: usize,
}

impl Snapshot {
    fn _sub(&mut self, other: &Self) {
        self.code -= other.code;
        self.string_data -= other.string_data;
        self.funcs -= other.funcs;
        self.globals -= other.globals;
        self.strings -= other.strings;
    }

    fn _add(&mut self, other: &Self) {
        self.code += other.code;
        self.string_data += other.string_data;
        self.funcs += other.funcs;
        self.globals += other.globals;
        self.strings += other.strings;
    }
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
        match *expr {
            Expr::Comment { .. } => Some(NILL),
            Expr::Ident { pos, id, .. } => {
                let msg = "i know nothing about this name gal which is vired\
                            because we parsed succesfully";
                Some(
                    self.ci
                        .vars
                        .iter()
                        .find(|v| v.id == id)
                        .unwrap_or_else(|| self.report(pos, msg))
                        .value,
                )
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
            Expr::Return { pos, val } => {
                let ty = if let Some(val) = val {
                    let value = self.expr_ctx(val, Ctx { ty: self.ci.ret })?;
                    let inps = [self.ci.cfg, value, self.ci.end];
                    self.ci.cfg = self.ci.nodes.new_node(ty::VOID, Kind::Return, inps);
                    self.tof(value)
                } else {
                    ty::VOID.into()
                };

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
                ctx.ty.unwrap_or(ty::UINT.into()),
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
        let func = self.tys.funcs[id as usize];
        debug_assert!(func.file == file);
        let sig = func.sig.unwrap();
        let ast = self.files[file as usize].clone();
        let expr = func.expr.get(&ast).unwrap();

        let repl = ItemCtx {
            file,
            id: ty::Kind::Func(id),
            ret: Some(sig.ret),
            ..self.pool.cis.pop().unwrap_or_default()
        };
        let prev_ci = std::mem::replace(&mut self.ci, repl);

        self.ci.start = self.ci.nodes.new_node(ty::VOID, Kind::Start, []);
        self.ci.end = self.ci.nodes.new_node(ty::VOID, Kind::End, []);
        self.ci.cfg = self.ci.nodes.new_node(ty::VOID, Kind::Tuple { index: 0 }, [self.ci.start]);

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

        for var in self.ci.vars.drain(..) {
            self.ci.nodes.unlock(var.value);
        }

        //self.pool.cis.push(std::mem::replace(&mut self.ci, prev_ci));
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
        log::dbg!("find_or_declare: {lit_name} {file}");

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
                && func.offset != u32::MAX
                && let Err(idx) = task::unpack(func.offset)
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
                    offset: u32::MAX,
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

#[cfg(test)]
mod tests {
    use {
        crate::parser::{self, FileId},
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

        use std::fmt::Write;

        write!(output, "{}", codegen.ci.nodes).unwrap();
    }

    crate::run_tests! { generate:
        arithmetic => README;
        const_folding_with_arg => README;
        //variables => README;
        //functions => README;
        //comments => README;
        //if_statements => README;
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
