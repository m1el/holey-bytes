use std::{
    cell::{Cell, RefCell},
    ops::Range,
};

use hbvm::Vm;

use crate::{
    ident::{self, Ident},
    parser::{idfl, ExprRef, FileId, Pos},
    HashMap,
};

use {
    crate::{
        instrs, lexer, log,
        parser::{self},
    },
    std::rc::Rc,
};

use {lexer::TokenKind as T, parser::Expr as E};

type FuncId = u32;
type Reg = u8;
type Type = u32;
type GlobalId = u32;

const VM_STACK_SIZE: usize = 1024 * 1024 * 2;

fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

struct ItemId {
    file: FileId,
    expr: parser::ExprRef,
    id:   u32,
}

#[derive(Debug, PartialEq, Eq)]
struct LinReg(Reg, Rc<RefCell<RegAlloc>>);

#[cfg(debug_assertions)]
impl Drop for LinReg {
    fn drop(&mut self) {
        self.1.borrow_mut().free(self.0)
    }
}

struct Stack {
    offset: u64,
    size:   u64,
    alloc:  Cell<Option<Rc<RefCell<StackAlloc>>>>,
}

impl std::fmt::Debug for Stack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stack")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl PartialEq for Stack {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.size == other.size
    }
}

impl Eq for Stack {}

impl Stack {
    fn leak(&self) {
        self.alloc.set(None);
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        if let Some(f) = self.alloc.get_mut().as_mut() {
            f.borrow_mut().free(self.offset, self.size)
        }
    }
}

#[derive(Default, PartialEq, Eq, Debug)]
struct StackAlloc {
    ranges: Vec<Range<u64>>,
    height: u64,
}

impl StackAlloc {
    fn alloc(&mut self, size: u64) -> (u64, u64) {
        if let Some((index, range)) = self
            .ranges
            .iter_mut()
            .enumerate()
            .filter(|(_, range)| range.end - range.start >= size)
            .min_by_key(|(_, range)| range.end - range.start)
        {
            let offset = range.start;
            range.start += size;
            if range.start == range.end {
                self.ranges.swap_remove(index);
            }
            return (offset, size);
        }

        let offset = self.height;
        self.height += size;
        (offset, size)
    }

    fn free(&mut self, offset: u64, size: u64) {
        let range = offset..offset + size;
        // FIXME: we do more wor then we need to, rather we keep the sequence sorted and only scan
        // element before and after the modified range
        self.ranges.push(range);
        self.ranges.sort_by_key(|range| range.start);
        self.ranges.dedup_by(|b, a| {
            if a.end == b.start {
                a.end = b.end;
                true
            } else {
                false
            }
        });
    }

    fn clear(&mut self) {
        assert!(self.ranges.len() <= 1, "{:?}", self.ranges);
        self.ranges.clear();
        self.height = 0;
    }
}

#[derive(Default, Debug)]
enum Ctx {
    #[default]
    None,
    Inferred(Type),
    Dest(Value),
    DestUntyped(Loc),
}

impl Ctx {
    fn ty(&self) -> Option<Type> {
        Some(match self {
            Self::Inferred(ty) => *ty,
            Self::Dest(Value { ty, .. }) => *ty,
            _ => return None,
        })
    }

    fn loc(self) -> Option<Loc> {
        Some(match self {
            Self::Dest(Value { loc, .. }) => loc,
            Self::DestUntyped(loc, ..) => loc,
            _ => return None,
        })
    }
}

mod traps {
    macro_rules! traps {
        ($($name:ident;)*) => {$(
            pub const $name: u64 = ${index(0)};
        )*};
    }

    traps! {
        MAKE_STRUCT;
    }
}

pub mod bt {
    use super::*;

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
            $(pub const $name: Type = ${index(0)};)*

            mod __lc_names {
                use super::*;
                $(pub const $name: &[u8] = &array_to_lower_case(unsafe {
                    *(stringify!($name).as_ptr() as *const [u8; stringify!($name).len()]) });)*
            }

            pub fn from_str(name: &str) -> Option<Type> {
                match name.as_bytes() {
                    $(__lc_names::$name => Some($name),)*
                    _ => None,
                }
            }

            pub fn to_str(ty: Type) -> &'static str {
                match ty {
                    $(${index(0)} => unsafe { std::str::from_utf8_unchecked(__lc_names::$name) },)*
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

    pub fn is_signed(ty: Type) -> bool {
        (I8..=INT).contains(&ty)
    }

    pub fn is_unsigned(ty: Type) -> bool {
        (U8..=UINT).contains(&ty)
    }

    pub fn strip_pointer(ty: Type) -> Type {
        match TypeKind::from_ty(ty) {
            TypeKind::Pointer(_) => INT,
            _ => ty,
        }
    }

    pub fn is_pointer(ty: Type) -> bool {
        matches!(TypeKind::from_ty(ty), TypeKind::Pointer(_))
    }

    pub fn try_upcast(oa: Type, ob: Type) -> Option<Type> {
        let (oa, ob) = (oa.min(ob), oa.max(ob));
        let (a, b) = (strip_pointer(oa), strip_pointer(ob));
        Some(match () {
            _ if oa == ob => oa,
            _ if is_signed(a) && is_signed(b) || is_unsigned(a) && is_unsigned(b) => ob,
            _ if is_unsigned(a) && is_signed(b) && a - U8 < b - I8 => ob,
            _ => return None,
        })
    }
}

macro_rules! type_kind {
    ($name:ident {$( $variant:ident, )*}) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum $name {
            $($variant(Type),)*
        }

        impl $name {
            const FLAG_BITS: u32 = (${count($variant)} as u32).next_power_of_two().ilog2();
            const FLAG_OFFSET: u32 = std::mem::size_of::<Type>() as u32 * 8 - Self::FLAG_BITS;
            const INDEX_MASK: u32 = (1 << (32 - Self::FLAG_BITS)) - 1;

            fn from_ty(ty: Type) -> Self {
                let (flag, index) = (ty >> Self::FLAG_OFFSET, ty & Self::INDEX_MASK);
                match flag {
                    $(${index(0)} => Self::$variant(index),)*
                    _ => unreachable!(),
                }
            }

            const fn encode(self) -> Type {
                let (index, flag) = match self {
                    $(Self::$variant(index) => (index, ${index(0)}),)*
                };
                (flag << Self::FLAG_OFFSET) | index
            }
        }
    };
}

type_kind! {
    TypeKind {
        Builtin,
        Struct,
        Pointer,
        Func,
        Global,
        Module,
    }
}

impl Default for TypeKind {
    fn default() -> Self {
        Self::Builtin(bt::UNDECLARED)
    }
}

const STACK_PTR: Reg = 254;
const ZERO: Reg = 0;
const RET_ADDR: Reg = 31;

struct Reloc {
    id: Type,
    offset: u32,
    instr_offset: u16,
    size: u16,
}

#[derive(Default)]
pub struct Block {
    code:   Vec<u8>,
    relocs: Vec<Reloc>,
}

impl Block {
    pub fn extend(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub fn offset(&mut self, id: FuncId, instr_offset: u16, size: u16) {
        self.relocs.push(Reloc {
            id: TypeKind::Func(id).encode(),
            offset: self.code.len() as u32,
            instr_offset,
            size,
        });
    }

    fn encode(&mut self, (len, instr): (usize, [u8; instrs::MAX_SIZE])) {
        let name = instrs::NAMES[instr[0] as usize];
        log::dbg!(
            "{:08x}: {}: {}",
            self.code.len(),
            name,
            instr
                .iter()
                .take(len)
                .skip(1)
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        );
        self.code.extend_from_slice(&instr[..len]);
    }

    fn short_cut_bin_op(&mut self, dest: Reg, src: Reg, imm: u64) -> bool {
        if imm == 0 && dest != src {
            self.encode(instrs::cp(dest, src));
        }
        imm != 0
    }

    fn addi64(&mut self, dest: Reg, src: Reg, imm: u64) {
        if self.short_cut_bin_op(dest, src, imm) {
            self.encode(instrs::addi64(dest, src, imm));
        }
    }

    fn call(&mut self, func: FuncId) {
        self.offset(func, 3, 4);
        self.encode(instrs::jal(RET_ADDR, ZERO, 0));
    }

    fn ret(&mut self) {
        self.encode(instrs::jala(ZERO, RET_ADDR, 0));
    }

    fn prelude(&mut self) {
        self.encode(instrs::jal(RET_ADDR, ZERO, 0));
        self.encode(instrs::tx());
    }

    fn relocate(&mut self, labels: &[Func], globals: &[Global], shift: i64, skip: usize) {
        for reloc in self.relocs.iter().skip(skip) {
            let offset = match TypeKind::from_ty(reloc.id) {
                TypeKind::Func(id) => labels[id as usize].offset,
                TypeKind::Global(id) => globals[id as usize].offset,
                v => unreachable!("invalid reloc: {:?}", v),
            };
            let offset = if reloc.size == 8 {
                reloc.offset as i64
            } else {
                offset as i64 - reloc.offset as i64
            } + shift;

            write_reloc(
                &mut self.code,
                reloc.offset as usize + reloc.instr_offset as usize,
                offset,
                reloc.size,
            );
        }
    }

    fn append(&mut self, data: &mut Block, code_offset: usize, reloc_offset: usize) {
        for reloc in &mut data.relocs[reloc_offset..] {
            reloc.offset += self.code.len() as u32;
            reloc.offset -= code_offset as u32;
        }
        self.relocs.extend(data.relocs.drain(reloc_offset..));
        self.code.extend(data.code.drain(code_offset..));
    }
}

fn write_reloc(doce: &mut [u8], offset: usize, value: i64, size: u16) {
    debug_assert!(size <= 8);
    debug_assert!(size.is_power_of_two());
    debug_assert!(
        doce[offset..offset + size as usize].iter().all(|&b| b == 0),
        "{:?}",
        &doce[offset..offset + size as usize]
    );
    let value = value.to_ne_bytes();
    doce[offset..offset + size as usize].copy_from_slice(&value[..size as usize]);
}

#[derive(Default, PartialEq, Eq)]
pub struct RegAlloc {
    free:     Vec<Reg>,
    max_used: Reg,
}

impl std::fmt::Debug for RegAlloc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegAlloc").finish()
    }
}

impl RegAlloc {
    fn init_callee(&mut self) {
        self.free.clear();
        self.free.extend((32..=253).rev());
        self.max_used = RET_ADDR;
    }

    fn allocate(&mut self) -> Reg {
        let reg = self.free.pop().expect("TODO: we need to spill");
        self.max_used = self.max_used.max(reg);
        reg
    }

    fn free(&mut self, reg: Reg) {
        self.free.push(reg);
    }

    fn pushed_size(&self) -> usize {
        ((self.max_used as usize).saturating_sub(RET_ADDR as usize) + 1) * 8
    }
}

#[derive(Clone)]
struct Func {
    offset: u32,
    relocs: u32,
    args:   Rc<[Type]>,
    ret:    Type,
}

struct Variable {
    id:    Ident,
    value: Value,
}

struct RetReloc {
    offset:       u32,
    instr_offset: u16,
    size:         u16,
}

struct Loop {
    var_count: usize,
    offset:    u32,
    relocs:    Vec<RetReloc>,
}

struct Struct {
    fields: Rc<[(Rc<str>, Type)]>,
}

struct TypeDisplay<'a> {
    codegen: &'a Codegen,
    ty:      Type,
}

impl<'a> TypeDisplay<'a> {
    fn new(codegen: &'a Codegen, ty: Type) -> Self {
        Self { codegen, ty }
    }

    fn rety(&self, ty: Type) -> Self {
        Self::new(self.codegen, ty)
    }
}

impl<'a> std::fmt::Display for TypeDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TypeKind as TK;
        match TK::from_ty(self.ty) {
            TK::Module(idx) => write!(f, "module{}", idx),
            TK::Builtin(ty) => write!(f, "{}", bt::to_str(ty)),
            TK::Pointer(ty) => {
                write!(f, "^{}", self.rety(self.codegen.pointers[ty as usize]))
            }
            _ if let Some((key, _)) =
                self.codegen.symbols.iter().find(|(_, &ty)| ty == self.ty)
                && let Some(name) = self.codegen.files[key.file as usize]
                    .exprs()
                    .iter()
                    .find_map(|expr| match expr {
                        E::BinOp {
                            left: &E::Ident { name, id, .. },
                            op: T::Decl,
                            ..
                        } if id == key.id => Some(name),
                        _ => None,
                    }) =>
            {
                write!(f, "{name}")
            }
            TK::Struct(idx) => {
                let record = &self.codegen.structs[idx as usize];
                write!(f, "{{")?;
                for (i, &(ref name, ty)) in record.fields.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, self.rety(ty))?;
                }
                write!(f, "}}")
            }
            TK::Func(idx) => write!(f, "fn{}", idx),
            TK::Global(idx) => write!(f, "global{}", idx),
        }
    }
}

struct Global {
    code:   u32,
    offset: u32,
    dep:    GlobalId,
    ty:     Type,
}

#[derive(Default)]
struct Linked {
    globals: usize,
    relocs:  usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SymKey {
    id:   Ident,
    file: FileId,
}

#[derive(Default)]
pub struct Codegen {
    cf:    parser::Ast,
    cf_id: FileId,

    ret:      Type,
    ret_reg:  Option<Reg>,
    cur_item: TypeKind,

    gpa:        Rc<RefCell<RegAlloc>>,
    sa:         Rc<RefCell<StackAlloc>>,
    ret_relocs: Vec<RetReloc>,
    loops:      Vec<Loop>,
    vars:       Vec<Variable>,

    to_generate: Vec<ItemId>,

    code: Block,
    data: Block,

    symbols:  HashMap<SymKey, Type>,
    funcs:    Vec<Func>,
    globals:  Vec<Global>,
    structs:  Vec<Struct>,
    pointers: Vec<Type>,

    pub files: Vec<parser::Ast>,

    vm:     Vm<LoggedMem, 0>,
    stack:  Vec<u8>,
    linked: Linked,
}

impl Codegen {
    fn with_cached_progress(&mut self, f: impl FnOnce(&mut Self)) {
        let ret = std::mem::take(&mut self.ret);
        let ret_reg = std::mem::take(&mut self.ret_reg);
        let cur_item = std::mem::take(&mut self.cur_item);

        let gpa = std::mem::take(&mut *self.gpa.borrow_mut());
        let sa = std::mem::take(&mut *self.sa.borrow_mut());
        let ret_relocs = self.ret_relocs.len();
        let loops = self.loops.len();
        let vars = self.vars.len();

        f(self);

        self.ret = ret;
        self.ret_reg = ret_reg;
        self.cur_item = cur_item;

        *self.gpa.borrow_mut() = gpa;
        *self.sa.borrow_mut() = sa;
        self.ret_relocs.truncate(ret_relocs);
        self.loops.truncate(loops);
        self.vars.truncate(vars);
    }

    fn lazy_init(&mut self) {
        if self.stack.capacity() == 0 {
            self.stack.reserve(VM_STACK_SIZE);
            self.vm.write_reg(
                STACK_PTR,
                unsafe { self.stack.as_ptr().add(self.stack.capacity()) } as u64,
            );
        }
    }

    pub fn generate(&mut self) {
        self.lazy_init();
        self.find_and_declare(0, 0, Err("main"));
        self.code.prelude();
        self.complete_call_graph();
    }

    fn complete_call_graph(&mut self) {
        while let Some(item) = self.to_generate.pop() {
            self.with_cached_progress(|s| s.generate_item(item));
        }
    }

    fn generate_item(&mut self, item: ItemId) {
        let ast = self.files[item.file as usize].clone();
        let expr = item.expr.get(&ast).unwrap();

        self.cf = ast.clone();
        self.cf_id = item.file;

        match expr {
            E::BinOp {
                left: E::Ident { name, .. },
                op: T::Decl,
                right: E::Closure { body, args, .. },
            } => {
                log::dbg!("fn: {}", name);

                self.cur_item = TypeKind::Func(item.id);
                self.funcs[item.id as usize].offset = self.code.code.len() as _;
                self.funcs[item.id as usize].relocs = self.code.relocs.len() as _;

                let func = self.funcs[item.id as usize].clone();
                self.gpa.borrow_mut().init_callee();

                self.gen_prelude();

                log::dbg!("fn-args");
                let mut parama = self.param_alloc(func.ret);
                for (arg, &ty) in args.iter().zip(func.args.iter()) {
                    let sym = parser::find_symbol(&self.cf.symbols, arg.id);
                    let loc = self.load_arg(sym.flags, ty, &mut parama);
                    self.vars.push(Variable {
                        id:    arg.id,
                        value: Value { ty, loc },
                    });
                }

                if self.size_of(func.ret) > 16 {
                    let reg = self.gpa.borrow_mut().allocate();
                    self.code.encode(instrs::cp(reg, 1));
                    self.ret_reg = Some(reg);
                } else {
                    self.ret_reg = None;
                }

                self.ret = func.ret;

                log::dbg!("fn-body");
                if self.expr(body).is_some() {
                    self.report(body.pos(), "expected all paths in the fucntion to return");
                }

                log::dbg!("fn-prelude, stack: {:x}", self.sa.borrow().height);

                log::dbg!("fn-relocs");
                self.reloc_prelude(item.id);

                log::dbg!("fn-ret");
                self.reloc_rets();
                self.ret();
                self.sa.borrow_mut().clear();
            }
            value => todo!("{value:?}"),
        }
    }

    fn align_of(&self, ty: Type) -> u64 {
        use TypeKind as TK;
        match TypeKind::from_ty(ty) {
            TK::Struct(t) => self.structs[t as usize]
                .fields
                .iter()
                .map(|&(_, ty)| self.align_of(ty))
                .max()
                .unwrap(),
            _ => self.size_of(ty).max(1),
        }
    }

    fn size_of(&self, ty: Type) -> u64 {
        use TypeKind as TK;
        match TK::from_ty(ty) {
            TK::Pointer(_) => 8,
            TK::Builtin(bt::VOID) => 0,
            TK::Builtin(bt::NEVER) => unreachable!(),
            TK::Builtin(bt::INT | bt::UINT) => 8,
            TK::Builtin(bt::I32 | bt::U32 | bt::TYPE) => 4,
            TK::Builtin(bt::I16 | bt::U16) => 2,
            TK::Builtin(bt::I8 | bt::U8 | bt::BOOL) => 1,
            TK::Struct(ty) => {
                let mut offset = 0;
                let record = &self.structs[ty as usize];
                for &(_, ty) in record.fields.iter() {
                    let align = self.align_of(ty);
                    offset = align_up(offset, align);
                    offset += self.size_of(ty);
                }
                offset
            }
            _ => unimplemented!("size_of: {}", self.display_ty(ty)),
        }
    }

    fn display_ty(&self, ty: Type) -> TypeDisplay {
        TypeDisplay::new(self, ty)
    }

    fn unwrap_struct(&self, ty: Type, pos: Pos, context: impl std::fmt::Display) -> Type {
        match TypeKind::from_ty(ty) {
            TypeKind::Struct(idx) => idx,
            _ => self.report(
                pos,
                format_args!("expected struct, got {} ({context})", self.display_ty(ty)),
            ),
        }
    }

    fn offset_of(&self, pos: Pos, idx: u32, field: Result<&str, usize>) -> (u64, Type) {
        let record = &self.structs[idx as usize];
        let mut offset = 0;
        for (i, &(ref name, ty)) in record.fields.iter().enumerate() {
            if Ok(name.as_ref()) == field || Err(i) == field {
                return (offset, ty);
            }
            let align = self.align_of(ty);
            offset = align_up(offset, align);
            offset += self.size_of(ty);
        }

        match field {
            Ok(i) => self.report(pos, format_args!("field not found: {i}")),
            Err(field) => self.report(pos, format_args!("field not found: {field}")),
        }
    }

    fn alloc_reg(&mut self) -> LinReg {
        LinReg(self.gpa.borrow_mut().allocate(), self.gpa.clone())
    }

    fn alloc_stack(&mut self, size: u64) -> Rc<Stack> {
        let (offset, size) = self.sa.borrow_mut().alloc(size);
        Stack {
            offset,
            size,
            alloc: Cell::new(Some(self.sa.clone())),
        }
        .into()
    }

    fn loc_to_reg(&mut self, loc: Loc, size: u64) -> LinReg {
        match loc {
            Loc::RegRef(rr) => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::cp(reg.0, rr));
                reg
            }
            Loc::Reg(reg) => reg,
            Loc::Deref(dreg, .., offset) => {
                let reg = self.alloc_reg();
                self.code
                    .encode(instrs::ld(reg.0, dreg.0, offset, size as _));
                reg
            }
            Loc::DerefRef(dreg, .., offset) => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::ld(reg.0, dreg, offset, size as _));
                reg
            }
            Loc::Imm(imm) => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::li64(reg.0, imm));
                reg
            }
            Loc::Stack(stack, off) => {
                let reg = self.alloc_reg();
                self.load_stack(reg.0, stack.offset + off, size as _);
                reg
            }
        }
    }

    fn store_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.code.encode(instrs::st(reg, STACK_PTR, offset, size));
    }

    fn load_stack(&mut self, reg: Reg, offset: u64, size: u16) {
        self.code.encode(instrs::ld(reg, STACK_PTR, offset, size));
    }

    fn reloc_rets(&mut self) {
        let len = self.code.code.len() as i64;
        for reloc in self.ret_relocs.iter() {
            write_reloc(
                &mut self.code.code,
                reloc.offset as usize + reloc.instr_offset as usize,
                len - reloc.offset as i64,
                reloc.size,
            );
        }
    }

    fn ty(&mut self, expr: &parser::Expr) -> Type {
        let offset = self.code.code.len();
        let reloc_offset = self.code.relocs.len();

        let value = self.expr(expr).unwrap();
        _ = self.assert_ty(expr.pos(), value.ty, bt::TYPE);
        if let Loc::Imm(ty) = value.loc {
            return ty as _;
        }

        self.code.encode(instrs::tx());

        let mut curr_temp = Block::default();
        curr_temp.append(&mut self.code, offset, reloc_offset);

        let mut curr_fn = Block::default();
        match self.cur_item {
            TypeKind::Func(id) => {
                let func = &self.funcs[id as usize];
                curr_fn.append(&mut self.code, func.offset as _, func.relocs as _);
                log::dbg!("{:?}", curr_fn.code);
            }
            foo => todo!("{foo:?}"),
        }

        let offset = self.code.code.len();
        self.code.append(&mut curr_temp, 0, 0);

        self.complete_call_graph();

        self.link();

        self.vm.pc = hbvm::mem::Address::new(&self.code.code[offset] as *const u8 as _);
        loop {
            match self.vm.run().unwrap() {
                hbvm::VmRunOk::End => break,
                hbvm::VmRunOk::Ecall => self.handle_ecall(),
                _ => unreachable!(),
            }
        }

        match self.cur_item {
            TypeKind::Func(id) => {
                self.funcs[id as usize].offset = self.code.code.len() as _;
                self.funcs[id as usize].relocs = self.code.relocs.len() as _;
                self.code.append(&mut curr_fn, 0, 0);
            }
            foo => todo!("{foo:?}"),
        }

        match value.loc {
            Loc::RegRef(reg) | Loc::Reg(LinReg(reg, ..)) => self.vm.read_reg(reg).0 as _,
            Loc::Deref(LinReg(reg, ..), .., off) | Loc::DerefRef(reg, .., off) => {
                let ptr = unsafe { (self.vm.read_reg(reg).0 as *const u8).add(off as _) };
                unsafe { std::ptr::read(ptr as *const Type) }
            }
            v => unreachable!("{v:?}"),
        }
    }

    fn handle_ecall(&mut self) {
        // the ecalls have exception, we cant pass the return value in two registers otherwise its
        // hard to tell where the trap code is
        match self.vm.read_reg(2).0 {
            traps::MAKE_STRUCT => unsafe {
                let file_id = self.vm.read_reg(3).0 as u32;
                let expr = std::mem::transmute::<_, parser::ExprRef>(self.vm.read_reg(4));
                let mut captures_addr = (self.vm.read_reg(STACK_PTR).0 as *const u8)
                    .add(self.vm.read_reg(5).0 as usize);
                let ast = self.files[file_id as usize].clone();
                let &E::Struct {
                    pos,
                    fields,
                    captured,
                } = expr.get(&ast).unwrap()
                else {
                    unreachable!()
                };

                let prev_len = self.vars.len();
                for &id in captured {
                    let ty: Type = std::ptr::read_unaligned(captures_addr.cast());
                    captures_addr = captures_addr.add(4);
                    let mut imm = [0u8; 8];
                    assert!(self.size_of(ty) as usize <= imm.len());
                    std::ptr::copy_nonoverlapping(
                        captures_addr,
                        imm.as_mut_ptr(),
                        self.size_of(ty) as usize,
                    );
                    self.vars.push(Variable {
                        id,
                        value: Value {
                            ty,
                            loc: Loc::Imm(u64::from_ne_bytes(imm)),
                        },
                    });
                }

                let Value {
                    loc: Loc::Imm(ty), ..
                } = self
                    .expr(&E::Struct {
                        pos,
                        fields,
                        captured: &[],
                    })
                    .unwrap()
                else {
                    unreachable!()
                };

                self.vars.truncate(prev_len);

                self.vm.write_reg(1, ty);
            },
            trap => todo!("unknown trap: {trap}"),
        }
    }

    fn expr(&mut self, expr: &parser::Expr) -> Option<Value> {
        self.expr_ctx(expr, Ctx::default())
    }

    fn handle_global(&mut self, id: GlobalId) -> Option<Value> {
        let ptr = self.alloc_reg();

        let global = &mut self.globals[id as usize];
        match self.cur_item {
            TypeKind::Global(gl) => global.dep = global.dep.max(gl),
            _ => {}
        }

        self.code.relocs.push(Reloc {
            id: TypeKind::Global(id as _).encode(),
            offset: self.code.code.len() as u32,
            instr_offset: 3,
            size: 4,
        });
        self.code.encode(instrs::lra(ptr.0, 0, 0));

        Some(Value {
            ty:  global.ty,
            loc: Loc::Deref(ptr, None, 0),
        })
    }

    fn expr_ctx(&mut self, expr: &parser::Expr, mut ctx: Ctx) -> Option<Value> {
        use instrs as i;

        let value = match *expr {
            E::Mod { id, .. } => Some(Value::ty(TypeKind::Module(id).encode())),
            E::Struct {
                fields, captured, ..
            } => {
                if captured.is_empty() {
                    let fields = fields
                        .iter()
                        .map(|&(name, ty)| (name.into(), self.ty(&ty)))
                        .collect();
                    self.structs.push(Struct { fields });
                    Some(Value::ty(
                        TypeKind::Struct(self.structs.len() as u32 - 1).encode(),
                    ))
                } else {
                    let values = captured
                        .iter()
                        .map(|&id| E::Ident {
                            id,
                            name: "booodab",
                            index: u16::MAX,
                        })
                        .map(|expr| self.expr(&expr))
                        .collect::<Option<Vec<_>>>()?;
                    let values_size = values
                        .iter()
                        .map(|value| 4 + self.size_of(value.ty))
                        .sum::<u64>();

                    let stack = self.alloc_stack(values_size);
                    let ptr = Loc::DerefRef(STACK_PTR, None, stack.offset);
                    let mut offset = 0;
                    for value in values {
                        self.assign(bt::TYPE, ptr.offset_ref(offset), Loc::Imm(value.ty as _));
                        offset += 4;
                        self.assign(value.ty, ptr.offset_ref(offset), value.loc);
                        offset += self.size_of(value.ty);
                    }

                    // eca MAKE_STRUCT(FileId, ExprRef, *Captures) -> Type;
                    let mut parama = self.param_alloc(bt::TYPE);
                    self.pass_arg(&Value::imm(traps::MAKE_STRUCT), &mut parama);
                    self.pass_arg(&Value::imm(self.cf_id as _), &mut parama);
                    self.pass_arg(
                        &Value::imm(unsafe { std::mem::transmute(parser::ExprRef::new(expr)) }),
                        &mut parama,
                    );
                    self.pass_arg(&Value::imm(stack.offset), &mut parama);
                    self.code.encode(i::eca());

                    Some(Value {
                        ty:  bt::TYPE,
                        loc: Loc::RegRef(1),
                    })
                }
            }
            E::UnOp {
                op: T::Xor, val, ..
            } => {
                let val = self.ty(val);
                Some(Value::ty(self.alloc_pointer(val)))
            }
            E::Directive {
                name: "TypeOf",
                args: [expr],
                ..
            } => {
                let offset = self.code.code.len() as u32;
                let reloc_offset = self.code.relocs.len();
                let ty = self
                    .expr_ctx(expr, Ctx::DestUntyped(Loc::DerefRef(0, None, 0)))
                    .unwrap()
                    .ty;
                self.code.code.truncate(offset as usize);
                self.code.relocs.truncate(reloc_offset);

                Some(Value {
                    ty:  bt::TYPE,
                    loc: Loc::Imm(ty as _),
                })
            }
            E::Directive {
                name: "eca",
                args: [ret_ty, args @ ..],
                ..
            } => {
                let ty = self.ty(ret_ty);

                let mut parama = self.param_alloc(ty);
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    let arg = self.expr(arg)?;
                    self.pass_arg(&arg, &mut parama);
                    values.push(arg.loc);
                }
                drop(values);

                let loc = self.alloc_ret_loc(ty, ctx);

                self.code.encode(i::eca());

                self.post_process_ret_loc(ty, &loc);

                return Some(Value { ty, loc });
            }
            E::Directive {
                name: "sizeof",
                args: [ty],
                ..
            } => {
                let ty = self.ty(ty);
                let loc = Loc::Imm(self.size_of(ty));
                return Some(Value { ty: bt::UINT, loc });
            }
            E::Directive {
                name: "alignof",
                args: [ty],
                ..
            } => {
                let ty = self.ty(ty);
                let loc = Loc::Imm(self.align_of(ty));
                return Some(Value { ty: bt::UINT, loc });
            }
            E::Directive {
                name: "intcast",
                args: [val],
                ..
            } => {
                let Some(ty) = ctx.ty() else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, <expr>)`",
                    );
                };
                let mut val = self.expr(val)?;

                let from_size = self.size_of(val.ty);
                let to_size = self.size_of(ty);

                if from_size < to_size && bt::is_signed(val.ty) {
                    let reg = self.loc_to_reg(val.loc, from_size);
                    let op = [i::sxt8, i::sxt16, i::sxt32][from_size.ilog2() as usize];
                    self.code.encode(op(reg.0, reg.0));
                    val.loc = Loc::Reg(reg);
                }

                Some(Value { ty, loc: val.loc })
            }
            E::Directive {
                name: "bitcast",
                args: [val],
                ..
            } => {
                let Some(ty) = ctx.ty() else {
                    self.report(
                        expr.pos(),
                        "type to cast to is unknown, use `@as(<type>, <expr>)`",
                    );
                };

                let size = self.size_of(ty);

                ctx = match ctx {
                    Ctx::Dest(Value { loc, .. }) | Ctx::DestUntyped(loc, ..) => {
                        Ctx::DestUntyped(loc)
                    }
                    _ => Ctx::None,
                };

                let val = self.expr_ctx(val, ctx)?;

                if self.size_of(val.ty) != size {
                    self.report(
                        expr.pos(),
                        format_args!(
                            "cannot bitcast {} to {} (different sizes: {} != {size})",
                            self.display_ty(val.ty),
                            self.display_ty(ty),
                            self.size_of(val.ty),
                        ),
                    );
                }

                // TODO: maybe check align

                return Some(Value { ty, loc: val.loc });
            }
            E::Directive {
                name: "as",
                args: [ty, val],
                ..
            } => {
                let ty = self.ty(ty);
                let ctx = match ctx {
                    Ctx::Dest(dest) => Ctx::Dest(dest),
                    Ctx::DestUntyped(loc) => Ctx::Dest(Value { ty, loc }),
                    _ => Ctx::Inferred(ty),
                };
                return self.expr_ctx(val, ctx);
            }
            E::Bool { value, .. } => Some(Value {
                ty:  bt::BOOL,
                loc: Loc::Imm(value as u64),
            }),
            E::Ctor {
                pos, ty, fields, ..
            } => {
                let Some(ty) = ty.map(|ty| self.ty(ty)).or(ctx.ty()) else {
                    self.report(pos, "expected type, (it cannot be inferred)");
                };
                let size = self.size_of(ty);

                let loc = match ctx.loc() {
                    Some(loc) => loc,
                    _ => Loc::Stack(self.alloc_stack(size), 0),
                };

                let stuct = self.unwrap_struct(ty, pos, "struct literal");
                let field_count = self.structs[stuct as usize].fields.len();
                if field_count != fields.len() {
                    self.report(
                        pos,
                        format_args!("expected {} fields, got {}", field_count, fields.len()),
                    );
                }

                for (i, (name, field)) in fields.iter().enumerate() {
                    let (offset, ty) = self.offset_of(field.pos(), stuct, name.ok_or(i));
                    let loc = loc.offset_ref(offset);
                    self.expr_ctx(field, Ctx::Dest(Value { ty, loc }))?;
                }

                return Some(Value { ty, loc });
            }
            E::Field { target, field } => {
                let checkpoint = self.code.code.len();
                let mut tal = self.expr(target)?;
                if let TypeKind::Pointer(ty) = TypeKind::from_ty(tal.ty) {
                    tal.ty = self.pointers[ty as usize];
                    tal.loc = match tal.loc {
                        Loc::Reg(r) => Loc::Deref(r, None, 0),
                        Loc::RegRef(r) => Loc::DerefRef(r, None, 0),
                        l => {
                            let ptr = self.loc_to_reg(l, 8);
                            Loc::Deref(ptr, None, 0)
                        }
                    };
                }

                match TypeKind::from_ty(tal.ty) {
                    TypeKind::Struct(idx) => {
                        let (offset, ty) = self.offset_of(target.pos(), idx, Ok(field));
                        let loc = tal.loc.offset(offset);
                        Some(Value { ty, loc })
                    }
                    TypeKind::Builtin(bt::TYPE) => {
                        self.code.code.truncate(checkpoint);
                        match TypeKind::from_ty(self.ty(target)) {
                            TypeKind::Module(idx) => Some(Value::ty(
                                self.find_and_declare(target.pos(), idx, Err(field))
                                    .encode(),
                            )),
                            _ => todo!(),
                        }
                    }
                    smh => self.report(
                        target.pos(),
                        format_args!("the field operation is not supported: {smh:?}"),
                    ),
                }
            }
            E::UnOp {
                op: T::Band,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                let loc = match val.loc {
                    Loc::Deref(r, stack, off) => {
                        if let Some(stack) = stack {
                            stack.leak()
                        }
                        self.code.addi64(r.0, r.0, off);
                        Loc::Reg(r)
                    }
                    Loc::DerefRef(r, stack, off) => {
                        if let Some(stack) = stack {
                            stack.leak()
                        }
                        let reg = self.alloc_reg();
                        self.code.addi64(reg.0, r, off);
                        Loc::Reg(reg)
                    }
                    Loc::Stack(stack, off) => {
                        stack.leak();
                        let reg = self.alloc_reg();
                        self.code
                            .encode(i::addi64(reg.0, STACK_PTR, stack.offset + off));
                        Loc::Reg(reg)
                    }
                    l => self.report(
                        pos,
                        format_args!("cant take pointer of {} ({:?})", self.display_ty(val.ty), l),
                    ),
                };
                Some(Value {
                    ty: self.alloc_pointer(val.ty),
                    loc,
                })
            }
            E::UnOp {
                op: T::Mul,
                val,
                pos,
            } => {
                let val = self.expr(val)?;
                match TypeKind::from_ty(val.ty) {
                    TypeKind::Pointer(ty) => Some(Value {
                        ty:  self.pointers[ty as usize],
                        loc: Loc::Deref(self.loc_to_reg(val.loc, self.size_of(val.ty)), None, 0),
                    }),
                    _ => self.report(
                        pos,
                        format_args!("expected pointer, got {}", self.display_ty(val.ty)),
                    ),
                }
            }
            E::BinOp {
                left: E::Ident { id, .. },
                op: T::Decl,
                right,
            } => {
                let val = self.expr(right)?;
                let loc = self.make_loc_owned(val.loc, val.ty);
                let sym = parser::find_symbol(&self.cf.symbols, *id);
                let loc = match loc {
                    Loc::Reg(r) if sym.flags & idfl::REFERENCED != 0 => {
                        let size = self.size_of(val.ty);
                        let stack = self.alloc_stack(size);
                        self.store_stack(r.0, stack.offset, size as _);
                        Loc::Stack(stack, 0)
                    }
                    l => l,
                };
                self.vars.push(Variable {
                    id:    *id,
                    value: Value { ty: val.ty, loc },
                });
                Some(Value::VOID)
            }
            E::Call { func, args } => {
                let func = self.ty(func);
                let TypeKind::Func(func) = TypeKind::from_ty(func) else {
                    todo!()
                };

                let fn_label = self.funcs[func as usize].clone();

                let mut parama = self.param_alloc(fn_label.ret);
                let mut values = Vec::with_capacity(args.len());
                for (earg, &ty) in args.iter().zip(fn_label.args.iter()) {
                    let arg = self.expr_ctx(earg, Ctx::Inferred(ty))?;
                    _ = self.assert_ty(earg.pos(), ty, arg.ty);
                    self.pass_arg(&arg, &mut parama);
                    values.push(arg.loc);
                }
                drop(values);

                let loc = self.alloc_ret_loc(fn_label.ret, ctx);

                self.code.call(func);

                self.post_process_ret_loc(fn_label.ret, &loc);

                return Some(Value {
                    ty: fn_label.ret,
                    loc,
                });
            }
            E::Ident { id, .. } if ident::is_null(id) => Some(Value::ty(id)),
            E::Ident { id, index, .. }
                if let Some((var_index, var)) =
                    self.vars.iter_mut().enumerate().find(|(_, v)| v.id == id) =>
            {
                let sym = parser::find_symbol(&self.cf.symbols, id);

                let loc = match idfl::index(sym.flags) == index
                    && !self.loops.last().is_some_and(|l| l.var_count > var_index)
                {
                    true => std::mem::replace(&mut var.value.loc, Loc::Imm(0)),
                    false => var.value.loc.take_ref(),
                };

                Some(Value {
                    ty: var.value.ty,
                    loc,
                })
            }
            E::Ident { id, .. } => match self
                .symbols
                .get(&SymKey {
                    id,
                    file: self.cf_id,
                })
                .copied()
                .map(TypeKind::from_ty)
                .unwrap_or_else(|| self.find_and_declare(ident::pos(id), self.cf_id, Ok(id)))
            {
                TypeKind::Global(id) => self.handle_global(id),
                tk => Some(Value::ty(tk.encode())),
            },
            E::Return { val, .. } => {
                if let Some(val) = val {
                    let size = self.size_of(self.ret);
                    let loc = match size {
                        0 => Loc::Imm(0),
                        ..=16 => Loc::RegRef(1),
                        _ => Loc::DerefRef(1, None, 0),
                    };
                    self.expr_ctx(val, Ctx::Dest(Value { loc, ty: self.ret }))?;
                }
                self.ret_relocs.push(RetReloc {
                    offset:       self.code.code.len() as u32,
                    instr_offset: 1,
                    size:         4,
                });
                self.code.encode(i::jmp(0));
                None
            }
            E::Block { stmts, .. } => {
                for stmt in stmts {
                    self.expr(stmt)?;
                }
                Some(Value::VOID)
            }
            E::Number { value, .. } => Some(Value {
                ty:  ctx.ty().map(bt::strip_pointer).unwrap_or(bt::INT),
                loc: Loc::Imm(value),
            }),
            E::If {
                cond, then, else_, ..
            } => 'b: {
                log::dbg!("if-cond");
                let cond = self.expr_ctx(cond, Ctx::Inferred(bt::BOOL))?;
                let reg = self.loc_to_reg(cond.loc, 1);
                let jump_offset = self.code.code.len() as u32;
                self.code.encode(i::jeq(reg.0, 0, 0));

                log::dbg!("if-then");
                let then_unreachable = self.expr(then).is_none();
                let mut else_unreachable = false;

                let mut jump = self.code.code.len() as i64 - jump_offset as i64;

                if let Some(else_) = else_ {
                    log::dbg!("if-else");
                    let else_jump_offset = self.code.code.len() as u32;
                    if !then_unreachable {
                        self.code.encode(i::jmp(0));
                        jump = self.code.code.len() as i64 - jump_offset as i64;
                    }

                    else_unreachable = self.expr(else_).is_none();

                    if !then_unreachable {
                        let jump = self.code.code.len() as i64 - else_jump_offset as i64;
                        log::dbg!("if-else-jump: {}", jump);
                        write_reloc(&mut self.code.code, else_jump_offset as usize + 1, jump, 4);
                    }
                }

                log::dbg!("if-then-jump: {}", jump);
                write_reloc(&mut self.code.code, jump_offset as usize + 3, jump, 2);

                if then_unreachable && else_unreachable {
                    break 'b None;
                }

                Some(Value::VOID)
            }
            E::Loop { body, .. } => 'a: {
                log::dbg!("loop");

                let loop_start = self.code.code.len() as u32;
                self.loops.push(Loop {
                    var_count: self.vars.len() as _,
                    offset:    loop_start,
                    relocs:    Default::default(),
                });
                let body_unreachable = self.expr(body).is_none();

                log::dbg!("loop-end");
                if !body_unreachable {
                    let loop_end = self.code.code.len();
                    self.code
                        .encode(i::jmp(loop_start as i32 - loop_end as i32));
                }

                let loop_end = self.code.code.len() as u32;

                let loop_ = self.loops.pop().unwrap();
                let is_unreachable = loop_.relocs.is_empty();
                for reloc in loop_.relocs {
                    let dest = &mut self.code.code
                        [reloc.offset as usize + reloc.instr_offset as usize..]
                        [..reloc.size as usize];
                    let offset = loop_end as i32 - reloc.offset as i32;
                    dest.copy_from_slice(&offset.to_ne_bytes());
                }

                self.vars.drain(loop_.var_count..);

                if is_unreachable {
                    log::dbg!("infinite loop");
                    break 'a None;
                }

                Some(Value::VOID)
            }
            E::Break { .. } => {
                let loop_ = self.loops.last_mut().unwrap();
                let offset = self.code.code.len() as u32;
                self.code.encode(i::jmp(0));
                loop_.relocs.push(RetReloc {
                    offset,
                    instr_offset: 1,
                    size: 4,
                });
                None
            }
            E::Continue { .. } => {
                let loop_ = self.loops.last().unwrap();
                let offset = self.code.code.len() as u32;
                self.code
                    .encode(i::jmp(loop_.offset as i32 - offset as i32));
                None
            }
            E::BinOp {
                left,
                op: op @ (T::And | T::Or),
                right,
            } => {
                let lhs = self.expr_ctx(left, Ctx::Inferred(bt::BOOL))?;
                let lhs = self.loc_to_reg(lhs.loc, 1);
                let jump_offset = self.code.code.len() + 3;
                let op = if op == T::And { i::jeq } else { i::jne };
                self.code.encode(op(lhs.0, 0, 0));

                if let Some(rhs) = self.expr_ctx(right, Ctx::Inferred(bt::BOOL)) {
                    let rhs = self.loc_to_reg(rhs.loc, 1);
                    self.code.encode(i::cp(lhs.0, rhs.0));
                }

                let jump = self.code.code.len() as i64 - jump_offset as i64;
                write_reloc(&mut self.code.code, jump_offset, jump, 2);

                Some(Value {
                    ty:  bt::BOOL,
                    loc: Loc::Reg(lhs),
                })
            }
            E::BinOp { left, op, right } => 'ops: {
                let left = self.expr(left)?;

                if op == T::Assign {
                    self.expr_ctx(right, Ctx::Dest(left)).unwrap();
                    return Some(Value::VOID);
                }

                if let TypeKind::Struct(_) = TypeKind::from_ty(left.ty) {
                    let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                    _ = self.assert_ty(expr.pos(), left.ty, right.ty);
                    return self.struct_op(op, left.ty, ctx, left.loc, right.loc);
                }

                let lsize = self.size_of(left.ty);
                let ty = ctx.ty().unwrap_or(left.ty);

                let (lhs, loc) = match std::mem::take(&mut ctx).loc() {
                    Some(Loc::RegRef(reg)) if Loc::RegRef(reg) == left.loc && reg != 1 => {
                        (reg, Loc::RegRef(reg))
                    }
                    Some(loc) => {
                        debug_assert!(!matches!(loc, Loc::Reg(LinReg(RET_ADDR, ..))));
                        ctx = Ctx::Dest(Value { ty, loc });
                        let reg = self.loc_to_reg(left.loc, lsize);
                        (reg.0, Loc::Reg(reg))
                    }
                    None => {
                        let reg = self.loc_to_reg(left.loc, lsize);
                        (reg.0, Loc::Reg(reg))
                    }
                };
                let right = self.expr_ctx(right, Ctx::Inferred(left.ty))?;
                let rsize = self.size_of(right.ty);

                let ty = self.assert_ty(expr.pos(), left.ty, right.ty);
                let size = self.size_of(ty);
                let signed = bt::is_signed(ty);

                if let Loc::Imm(mut imm) = right.loc
                    && let Some(oper) = Self::imm_math_op(op, signed, size)
                {
                    if matches!(op, T::Add | T::Sub)
                        && let TypeKind::Pointer(ty) = TypeKind::from_ty(ty)
                    {
                        let size = self.size_of(self.pointers[ty as usize]);
                        imm *= size;
                    }

                    self.code.encode(oper(lhs, lhs, imm));
                    break 'ops Some(Value { ty, loc });
                }

                let rhs = self.loc_to_reg(right.loc, rsize);

                if matches!(op, T::Add | T::Sub) {
                    let min_size = lsize.min(rsize);
                    if bt::is_signed(ty) && min_size < size {
                        let operand = if lsize < rsize { lhs } else { rhs.0 };
                        let op = [i::sxt8, i::sxt16, i::sxt32][min_size.ilog2() as usize];
                        self.code.encode(op(operand, operand));
                    }

                    if bt::is_pointer(left.ty) ^ bt::is_pointer(right.ty) {
                        let (offset, ty) = if bt::is_pointer(left.ty) {
                            (rhs.0, left.ty)
                        } else {
                            (lhs, right.ty)
                        };

                        let TypeKind::Pointer(ty) = TypeKind::from_ty(ty) else {
                            unreachable!()
                        };

                        let size = self.size_of(self.pointers[ty as usize]);
                        self.code.encode(i::muli64(offset, offset, size as _));
                    }
                }

                if let Some(op) = Self::math_op(op, signed, size) {
                    self.code.encode(op(lhs, lhs, rhs.0));
                    break 'ops Some(Value { ty, loc });
                }

                'cmp: {
                    let against = match op {
                        T::Le | T::Gt => 1,
                        T::Ne | T::Eq => 0,
                        T::Ge | T::Lt => (-1i64) as _,
                        _ => break 'cmp,
                    };

                    let op_fn = if signed { i::cmps } else { i::cmpu };
                    self.code.encode(op_fn(lhs, lhs, rhs.0));
                    self.code.encode(i::cmpui(lhs, lhs, against));
                    if matches!(op, T::Eq | T::Lt | T::Gt) {
                        self.code.encode(i::not(lhs, lhs));
                    }

                    break 'ops Some(Value { ty: bt::BOOL, loc });
                }

                unimplemented!("{:#?}", op)
            }
            ast => unimplemented!("{:#?}", ast),
        }?;

        match ctx {
            Ctx::Dest(dest) => {
                _ = self.assert_ty(expr.pos(), value.ty, dest.ty);
                self.assign(dest.ty, dest.loc, value.loc)?;
                Some(Value {
                    ty:  dest.ty,
                    loc: Loc::Imm(0),
                })
            }
            Ctx::DestUntyped(loc) => {
                // Wo dont check since bitcast does
                self.assign(value.ty, loc, value.loc);
                Some(Value {
                    ty:  value.ty,
                    loc: Loc::Imm(0),
                })
            }
            _ => Some(value),
        }
    }

    fn math_op(
        op: T,
        signed: bool,
        size: u64,
    ) -> Option<fn(u8, u8, u8) -> (usize, [u8; instrs::MAX_SIZE])> {
        use instrs as i;

        macro_rules! div { ($($op:ident),*) => {[$(|a, b, c| i::$op(a, ZERO, b, c)),*]}; }
        macro_rules! rem { ($($op:ident),*) => {[$(|a, b, c| i::$op(ZERO, a, b, c)),*]}; }

        let ops = match op {
            T::Add => [i::add8, i::add16, i::add32, i::add64],
            T::Sub => [i::sub8, i::sub16, i::sub32, i::sub64],
            T::Mul => [i::mul8, i::mul16, i::mul32, i::mul64],
            T::Div if signed => div!(dirs8, dirs16, dirs32, dirs64),
            T::Div => div!(diru8, diru16, diru32, diru64),
            T::Mod if signed => rem!(dirs8, dirs16, dirs32, dirs64),
            T::Mod => rem!(diru8, diru16, diru32, diru64),
            T::Band => return Some(i::and),
            T::Bor => return Some(i::or),
            T::Xor => return Some(i::xor),
            T::Shl => [i::slu8, i::slu16, i::slu32, i::slu64],
            T::Shr if signed => [i::srs8, i::srs16, i::srs32, i::srs64],
            T::Shr => [i::sru8, i::sru16, i::sru32, i::sru64],
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
    }

    fn imm_math_op(
        op: T,
        signed: bool,
        size: u64,
    ) -> Option<fn(u8, u8, u64) -> (usize, [u8; instrs::MAX_SIZE])> {
        use instrs as i;

        macro_rules! def_op {
            ($name:ident |$a:ident, $b:ident, $c:ident| $($tt:tt)*) => {
                macro_rules! $name {
                    ($$($$op:ident),*) => {
                        [$$(
                            |$a, $b, $c: u64| i::$$op($($tt)*),
                        )*]
                    }
                }
            };
        }

        def_op!(basic_op | a, b, c | a, b, c as _);
        def_op!(sub_op | a, b, c | b, a, c.wrapping_neg() as _);

        let ops = match op {
            T::Add => basic_op!(addi8, addi16, addi32, addi64),
            T::Sub => sub_op!(addi8, addi16, addi32, addi64),
            T::Mul => basic_op!(muli8, muli16, muli32, muli64),
            T::Band => return Some(i::andi),
            T::Bor => return Some(i::ori),
            T::Xor => return Some(i::xori),
            T::Shr if signed => basic_op!(srui8, srui16, srui32, srui64),
            T::Shr => basic_op!(srui8, srui16, srui32, srui64),
            T::Shl => basic_op!(slui8, slui16, slui32, slui64),
            _ => return None,
        };

        Some(ops[size.ilog2() as usize])
    }

    fn struct_op(&mut self, op: T, ty: Type, ctx: Ctx, left: Loc, right: Loc) -> Option<Value> {
        if let TypeKind::Struct(stuct) = TypeKind::from_ty(ty) {
            let dst = match ctx {
                Ctx::Dest(dest) => dest.loc,
                _ => Loc::Stack(self.alloc_stack(self.size_of(ty)), 0),
            };
            let mut offset = 0;
            for &(_, ty) in self.structs[stuct as usize].fields.clone().iter() {
                let align = self.align_of(ty);
                offset = align_up(offset, align);
                let size = self.size_of(ty);
                let ctx = Ctx::Dest(Value::new(ty, dst.offset_ref(offset)));
                let left = left.offset_ref(offset);
                let right = right.offset_ref(offset);
                self.struct_op(op, ty, ctx, left, right)?;
                offset += size;
            }

            return Some(Value { ty, loc: dst });
        }

        let size = self.size_of(ty);
        let signed = bt::is_signed(ty);
        let (lhs, owned) = self.loc_to_reg_ref(&left, size);

        if let Loc::Imm(imm) = right
            && let Some(op) = Self::imm_math_op(op, signed, size)
        {
            self.code.encode(op(lhs, lhs, imm));
            return if let Ctx::Dest(dest) = ctx {
                self.assign(dest.ty, dest.loc, owned.map_or(Loc::RegRef(lhs), Loc::Reg));
                Some(Value::VOID)
            } else {
                Some(Value::new(ty, owned.map_or(Loc::RegRef(lhs), Loc::Reg)))
            };
        }

        let rhs = self.loc_to_reg(right, size);

        if let Some(op) = Self::math_op(op, signed, size) {
            self.code.encode(op(lhs, lhs, rhs.0));
            return if let Ctx::Dest(dest) = ctx {
                self.assign(dest.ty, dest.loc, owned.map_or(Loc::RegRef(lhs), Loc::Reg));
                Some(Value::VOID)
            } else {
                Some(Value::new(ty, owned.map_or(Loc::RegRef(lhs), Loc::Reg)))
            };
        }

        unimplemented!("{:#?}", op)
    }

    fn loc_to_reg_ref(&mut self, loc: &Loc, size: u64) -> (u8, Option<LinReg>) {
        match *loc {
            Loc::RegRef(reg) => (reg, None),
            Loc::Reg(LinReg(reg, ..)) => (reg, None),
            Loc::Deref(LinReg(reg, ..), .., off) | Loc::DerefRef(reg, .., off) => {
                let new = self.alloc_reg();
                self.code.encode(instrs::ld(new.0, reg, off, size as _));
                (new.0, Some(new))
            }
            Loc::Stack(ref stack, off) => {
                let new = self.alloc_reg();
                self.load_stack(new.0, stack.offset + off, size as _);
                (new.0, Some(new))
            }
            Loc::Imm(imm) => {
                let new = self.alloc_reg();
                self.code.encode(instrs::li64(new.0, imm));
                (new.0, Some(new))
            }
        }
    }

    fn assign_opaque(&mut self, size: u64, right: Loc, left: Loc) -> Option<Value> {
        if left == right {
            return Some(Value::VOID);
        }

        match size {
            0 => {}
            ..=8 if let Loc::Imm(imm) = left
                && let Loc::RegRef(reg) = right =>
            {
                self.code.encode(instrs::li64(reg, imm))
            }
            ..=8 => {
                let lhs = self.loc_to_reg(left, size);
                match right {
                    Loc::RegRef(reg) => self.code.encode(instrs::cp(reg, lhs.0)),
                    Loc::Deref(reg, .., off) => {
                        self.code.encode(instrs::st(lhs.0, reg.0, off, size as _));
                    }
                    Loc::DerefRef(reg, .., off) => {
                        self.code.encode(instrs::st(lhs.0, reg, off, size as _));
                    }
                    Loc::Stack(stack, off) => {
                        self.store_stack(lhs.0, stack.offset + off, size as _);
                    }
                    l => unimplemented!("{:?}", l),
                }
            }
            ..=16 if matches!(right, Loc::RegRef(1)) => {
                let (lhs, loff) = left.ref_to_ptr();
                self.code.encode(instrs::st(1, lhs, loff, 16));
            }
            ..=u64::MAX => {
                let rhs = self.to_ptr(right);
                let lhs = self.to_ptr(left);
                self.code
                    .encode(instrs::bmc(lhs.0, rhs.0, size.try_into().unwrap()));
            }
        }

        Some(Value::VOID)
    }

    fn assign(&mut self, ty: Type, right: Loc, left: Loc) -> Option<Value> {
        self.assign_opaque(self.size_of(ty), right, left)
    }

    fn to_ptr(&mut self, loc: Loc) -> LinReg {
        match loc {
            Loc::Deref(reg, .., off) => {
                self.code.addi64(reg.0, reg.0, off);
                reg
            }
            Loc::DerefRef(reg, .., off) => {
                let new = self.alloc_reg();
                self.code.addi64(new.0, reg, off);
                new
            }
            Loc::Stack(stack, off) => {
                let reg = self.alloc_reg();
                self.code.addi64(reg.0, STACK_PTR, stack.offset + off);
                reg
            }
            l => unreachable!("{:?}", l),
        }
    }

    fn find_and_declare(&mut self, pos: Pos, file: FileId, name: Result<Ident, &str>) -> TypeKind {
        let f = self.files[file as usize].clone();
        let Some((expr, id)) = f.find_decl(name) else {
            self.report(
                pos,
                match name {
                    Ok(_) => format!("undefined indentifier"),
                    Err("main") => {
                        format!("compilation root is missing main function: {f}")
                    }
                    Err(name) => todo!("somehow we did not handle: {name:?}"),
                },
            );
        };

        let sym = match expr {
            E::BinOp {
                left: &E::Ident { .. },
                op: T::Decl,
                right: E::Closure { args, ret, .. },
            } => {
                let args = args.iter().map(|arg| self.ty(&arg.ty)).collect::<Vec<_>>();
                let ret = self.ty(ret);
                let id = self.declare_fn_label(args.into(), ret);
                self.to_generate.push(ItemId {
                    file,
                    expr: ExprRef::new(expr),
                    id,
                });
                TypeKind::Func(id)
            }
            E::BinOp {
                left: &E::Ident { .. },
                op: T::Decl,
                right: E::Struct { fields, .. },
            } => {
                let fields = fields
                    .iter()
                    .map(|&(name, ty)| (name.into(), self.ty(&ty)))
                    .collect();
                self.structs.push(Struct { fields });
                TypeKind::Struct(self.structs.len() as u32 - 1)
            }
            E::BinOp {
                left: &E::Ident { .. },
                op: T::Decl,
                right,
            } => {
                let gid = self.globals.len() as GlobalId;
                let prev_in_global = std::mem::replace(&mut self.cur_item, TypeKind::Global(gid));

                let prev_gpa = std::mem::replace(&mut *self.gpa.borrow_mut(), Default::default());
                let prev_sa = std::mem::replace(&mut *self.sa.borrow_mut(), Default::default());

                let offset = self.code.code.len();
                let reloc_count = self.code.relocs.len();
                self.globals.push(Global {
                    ty:     bt::UNDECLARED,
                    code:   0,
                    dep:    0,
                    offset: u32::MAX,
                });

                self.gpa.borrow_mut().init_callee();
                let ret = self.gpa.borrow_mut().allocate();
                // TODO: detect is constant does not call anything
                self.code.encode(instrs::cp(ret, 1));

                let ret = self
                    .expr_ctx(right, Ctx::DestUntyped(Loc::DerefRef(ret, None, 0)))
                    .expect("TODO: unreachable constant/global");
                self.code.encode(instrs::tx());
                self.globals[gid as usize].ty = ret.ty;

                self.globals[gid as usize].code = self.data.code.len() as u32;
                self.data.append(&mut self.code, offset, reloc_count);

                *self.sa.borrow_mut() = prev_sa;
                *self.gpa.borrow_mut() = prev_gpa;
                self.cur_item = prev_in_global;

                TypeKind::Global(gid)
            }
            e => unimplemented!("{e:#?}"),
        };
        self.symbols.insert(SymKey { id, file }, sym.encode());
        sym
    }

    fn declare_fn_label(&mut self, args: Rc<[Type]>, ret: Type) -> FuncId {
        self.funcs.push(Func {
            offset: 0,
            relocs: 0,
            args,
            ret,
        });
        self.funcs.len() as u32 - 1
    }

    fn gen_prelude(&mut self) {
        self.code.encode(instrs::addi64(STACK_PTR, STACK_PTR, 0));
        self.code.encode(instrs::st(RET_ADDR, STACK_PTR, 0, 0));
    }

    fn reloc_prelude(&mut self, id: FuncId) {
        let mut cursor = self.funcs[id as usize].offset as usize;
        let mut allocate = |size| (cursor += size, cursor).1;

        let pushed = self.gpa.borrow().pushed_size() as i64;
        let stack = self.sa.borrow().height as i64;

        write_reloc(&mut self.code.code, allocate(3), -(pushed + stack), 8);
        write_reloc(&mut self.code.code, allocate(8 + 3), stack, 8);
        write_reloc(&mut self.code.code, allocate(8), pushed, 2);
    }

    fn ret(&mut self) {
        let pushed = self.gpa.borrow().pushed_size() as u64;
        let stack = self.sa.borrow().height as u64;
        self.code
            .encode(instrs::ld(RET_ADDR, STACK_PTR, stack, pushed as _));
        self.code
            .encode(instrs::addi64(STACK_PTR, STACK_PTR, stack + pushed));
        self.code.ret();
    }

    fn link(&mut self) {
        let mut globals = std::mem::take(&mut self.globals);
        for global in globals.iter_mut().skip(self.linked.globals as usize) {
            let size = self.size_of(global.ty);
            global.offset = self.code.code.len() as u32;
            self.code.code.extend(std::iter::repeat(0).take(size as _));
        }
        self.globals = globals;

        let prev_len = self.code.code.len();
        self.code.append(&mut self.data, 0, 0);

        self.code
            .relocate(&self.funcs, &self.globals, 0, self.linked.relocs);

        let mut var_order = self
            .globals
            .iter()
            .map(|g| g.dep)
            .zip(0u32..)
            .skip(self.linked.globals as usize)
            .collect::<Vec<_>>();
        var_order.sort_unstable();

        for (_, glob_id) in var_order.into_iter().rev() {
            let global = &self.globals[glob_id as usize];
            self.vm.pc = hbvm::mem::Address::new(
                &mut self.code.code[global.code as usize + prev_len] as *mut _ as u64,
            );
            self.vm.write_reg(
                1,
                &mut self.code.code[global.offset as usize] as *mut _ as u64,
            );
            self.vm.run().unwrap();
        }
        self.code.code.truncate(prev_len);

        self.linked.globals = self.globals.len();
        self.linked.relocs = self.code.relocs.len();
    }

    pub fn dump(mut self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        self.code.relocs.push(Reloc {
            offset: 0,
            size: 4,
            instr_offset: 3,
            id: TypeKind::Func(0).encode() as _,
        });
        self.link();
        out.write_all(&self.code.code)
    }

    fn alloc_pointer(&mut self, ty: Type) -> Type {
        let ty = self
            .pointers
            .iter()
            .position(|&p| p == ty)
            .unwrap_or_else(|| {
                self.pointers.push(ty);
                self.pointers.len() - 1
            });

        TypeKind::Pointer(ty as Type).encode()
    }

    fn make_loc_owned(&mut self, loc: Loc, ty: Type) -> Loc {
        match loc {
            Loc::RegRef(rreg) => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::cp(reg.0, rreg));
                Loc::Reg(reg)
            }
            Loc::Imm(imm) => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::li64(reg.0, imm));
                Loc::Reg(reg)
            }
            l @ (Loc::DerefRef(..) | Loc::Deref(..)) => {
                let size = self.size_of(ty);
                let stack = self.alloc_stack(size);
                self.assign(ty, Loc::DerefRef(STACK_PTR, None, stack.offset), l);
                Loc::Stack(stack, 0)
            }
            l => l,
        }
    }

    fn pass_arg(&mut self, value: &Value, parama: &mut Range<u8>) {
        let size = self.size_of(value.ty);
        let p = parama.next().unwrap() as Reg;

        if size > 16 {
            let (ptr, off) = value.loc.ref_to_ptr();
            self.code.addi64(p, ptr, off);
            return;
        }

        match value.loc {
            Loc::Reg(LinReg(reg, ..)) | Loc::RegRef(reg) => self.code.encode(instrs::cp(p, reg)),
            Loc::Deref(LinReg(reg, ..), .., off) | Loc::DerefRef(reg, .., off) => {
                self.code.encode(instrs::ld(p, reg, off, size as _));
            }
            Loc::Imm(imm) => self.code.encode(instrs::li64(p, imm)),
            Loc::Stack(ref stack, off) => {
                self.load_stack(p, stack.offset + off, size as _);
                if size > 8 {
                    parama.next().unwrap();
                }
            }
        }
    }

    fn load_arg(&mut self, flags: parser::IdentFlags, ty: Type, parama: &mut Range<u8>) -> Loc {
        let size = self.size_of(ty);
        match size {
            0 => Loc::Imm(0),
            ..=8 if flags & idfl::REFERENCED == 0 => {
                let reg = self.alloc_reg();
                self.code.encode(instrs::cp(reg.0, parama.next().unwrap()));
                Loc::Reg(reg)
            }
            ..=8 => {
                let stack = self.alloc_stack(size as _);
                self.store_stack(parama.next().unwrap(), stack.offset, size as _);
                Loc::Stack(stack, 0)
            }
            ..=16 => {
                let stack = self.alloc_stack(size);
                self.store_stack(parama.next().unwrap(), stack.offset, size as _);
                parama.next().unwrap();
                Loc::Stack(stack, 0)
            }
            _ if flags & (idfl::MUTABLE | idfl::REFERENCED) == 0 => {
                let ptr = parama.next().unwrap();
                let reg = self.alloc_reg();
                self.code.encode(instrs::cp(reg.0, ptr));
                Loc::Deref(reg, None, 0)
            }
            _ => {
                let ptr = parama.next().unwrap();
                let stack = self.alloc_stack(size);
                self.assign(
                    ty,
                    Loc::DerefRef(STACK_PTR, None, stack.offset),
                    Loc::DerefRef(ptr, None, 0),
                );
                Loc::Stack(stack, 0)
            }
        }
    }

    #[must_use]
    fn assert_ty(&self, pos: Pos, ty: Type, expected: Type) -> Type {
        if let Some(res) = bt::try_upcast(ty, expected) {
            res
        } else {
            let ty = self.display_ty(ty);
            let expected = self.display_ty(expected);
            self.report(pos, format_args!("expected {ty}, got {expected}"));
        }
    }

    fn report(&self, pos: Pos, msg: impl std::fmt::Display) -> ! {
        let (line, col) = self.cf.nlines.line_col(pos);
        println!("{}:{}:{}: {}", self.cf.path, line, col, msg);
        unreachable!();
    }

    fn alloc_ret_loc(&mut self, ret: Type, ctx: Ctx) -> Loc {
        let size = self.size_of(ret);
        match size {
            0 => Loc::Imm(0),
            ..=8 => Loc::RegRef(1),
            ..=16 => match ctx {
                Ctx::Dest(dest) => dest.loc,
                _ => Loc::Stack(self.alloc_stack(size), 0),
            },
            ..=u64::MAX => {
                let val = match ctx {
                    Ctx::Dest(dest) => dest.loc,
                    _ => Loc::Stack(self.alloc_stack(size), 0),
                };
                let (ptr, off) = val.ref_to_ptr();
                self.code.encode(instrs::cp(1, ptr));
                self.code.addi64(1, ptr, off);
                val
            }
        }
    }

    fn post_process_ret_loc(&mut self, ty: Type, loc: &Loc) {
        let size = self.size_of(ty);
        match size {
            0 => {}
            ..=8 => {}
            ..=16 => {
                if let Loc::Stack(ref stack, off) = loc {
                    self.store_stack(1, stack.offset + off, size as _);
                } else {
                    unreachable!()
                }
            }
            ..=u64::MAX => {}
        }
    }

    fn param_alloc(&self, ret: Type) -> Range<u8> {
        2 + (9..=16).contains(&self.size_of(ret)) as u8..12
    }
}

#[derive(Debug)]
pub struct Value {
    ty:  Type,
    loc: Loc,
}

impl Value {
    const VOID: Self = Self {
        ty:  bt::VOID,
        loc: Loc::Imm(0),
    };

    fn new(ty: Type, loc: Loc) -> Self {
        Self { ty, loc }
    }

    fn ty(ty: Type) -> Self {
        Self {
            ty:  bt::TYPE,
            loc: Loc::Imm(ty as _),
        }
    }

    fn imm(imm: u64) -> Value {
        Self {
            ty:  bt::UINT,
            loc: Loc::Imm(imm),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Loc {
    Reg(LinReg),
    RegRef(Reg),
    Deref(LinReg, Option<Rc<Stack>>, u64),
    DerefRef(Reg, Option<Rc<Stack>>, u64),
    Imm(u64),
    Stack(Rc<Stack>, u64),
}
impl Loc {
    fn take_ref(&self) -> Loc {
        match *self {
            Self::Reg(LinReg(reg, ..), ..) | Self::RegRef(reg) => Self::RegRef(reg),
            Self::Deref(LinReg(reg, ..), ref st, off) | Self::DerefRef(reg, ref st, off) => {
                Self::DerefRef(reg, st.clone(), off)
            }
            Self::Stack(ref stack, off) => {
                Self::DerefRef(STACK_PTR, Some(stack.clone()), stack.offset + off)
            }
            ref un => unreachable!("{:?}", un),
        }
    }

    fn ref_to_ptr(&self) -> (Reg, u64) {
        match *self {
            Loc::Deref(LinReg(reg, ..), _, off) => (reg, off),
            Loc::DerefRef(reg, _, off) => (reg, off),
            Loc::Stack(ref stack, off) => (STACK_PTR, stack.offset + off),
            ref l => panic!("expected stack location, got {:?}", l),
        }
    }

    fn offset_ref(&self, offset: u64) -> Loc {
        self.take_ref().offset(offset)
    }

    fn offset(self, offset: u64) -> Loc {
        match self {
            Loc::Deref(r, stack, off) => Loc::Deref(r, stack, off + offset),
            Loc::DerefRef(r, stack, off) => Loc::DerefRef(r, stack, off + offset),
            Loc::Stack(s, off) => Loc::Stack(s, off + offset),
            l => unreachable!("{:?}", l),
        }
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
        log::dbg!(
            "load: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(addr.get() as *const u8, count)
        );
        self.mem.load(addr, target, count)
    }

    unsafe fn store(
        &mut self,
        addr: hbvm::mem::Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), hbvm::mem::StoreError> {
        log::dbg!(
            "store: {:x} {:?}",
            addr.get(),
            core::slice::from_raw_parts(source, count)
        );
        self.mem.store(addr, source, count)
    }

    unsafe fn prog_read<T: Copy>(&mut self, addr: hbvm::mem::Address) -> T {
        log::dbg!(
            "read-typed: {:x} {} {:?}",
            addr.get(),
            std::any::type_name::<T>(),
            if core::mem::size_of::<T>() == 1 {
                instrs::NAMES[std::ptr::read(addr.get() as *const u8) as usize].to_string()
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
    use crate::{codegen::LoggedMem, log};

    use super::parser;

    fn generate(input: &'static str, output: &mut String) {
        let path = "test";
        let mut codegen = super::Codegen::default();
        codegen.files = vec![parser::Ast::new(path, input, &parser::no_loader)];
        codegen.generate();
        let mut out = Vec::new();
        codegen.dump(&mut out).unwrap();

        use std::fmt::Write;

        let mut stack = [0_u64; 128];

        let mut vm = unsafe {
            hbvm::Vm::<_, 0>::new(
                LoggedMem::default(),
                hbvm::mem::Address::new(out.as_ptr() as u64),
            )
        };

        vm.write_reg(
            super::STACK_PTR,
            unsafe { stack.as_mut_ptr().add(stack.len()) } as u64,
        );

        let stat = loop {
            match vm.run() {
                Ok(hbvm::VmRunOk::End) => break Ok(()),
                Ok(ev) => writeln!(output, "ev: {:?}", ev).unwrap(),
                Err(e) => break Err(e),
            }
        };

        writeln!(output, "code size: {}", out.len()).unwrap();
        writeln!(output, "ret: {:?}", vm.read_reg(1).0).unwrap();
        writeln!(output, "status: {:?}", stat).unwrap();
        log::inf!("input lenght: {}", input.len());
    }

    crate::run_tests! { generate:
        example => include_str!("../examples/main_fn.hb");
        arithmetic => include_str!("../examples/arithmetic.hb");
        variables => include_str!("../examples/variables.hb");
        functions => include_str!("../examples/functions.hb");
        if_statements => include_str!("../examples/if_statement.hb");
        loops => include_str!("../examples/loops.hb");
        fb_driver => include_str!("../examples/fb_driver.hb");
        pointers => include_str!("../examples/pointers.hb");
        structs => include_str!("../examples/structs.hb");
        different_types => include_str!("../examples/different_types.hb");
        struct_operators => include_str!("../examples/struct_operators.hb");
        directives => include_str!("../examples/directives.hb");
        global_variables => include_str!("../examples/global_variables.hb");
        geneic_types => include_str!("../examples/generic_types.hb");
    }
}
