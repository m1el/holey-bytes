//! HoleyBytes Virtual Machine
//!
//! All unsafe code here should be sound, if input bytecode passes validation.

// # General safety notice:
// - Validation has to assure there is 256 registers (r0 - r255)
// - Instructions have to be valid as specified (values and sizes)
// - Mapped pages should be at least 4 KiB

use self::mem::HandlePageFault;

pub mod mem;
pub mod value;

use {
    crate::validate,
    core::ops,
    hbbytecode::{OpParam, ParamBB, ParamBBB, ParamBBBB, ParamBBD, ParamBBDH, ParamBD},
    mem::Memory,
    static_assertions::assert_impl_one,
    value::Value,
};

/// Extract a parameter from program
macro_rules! param {
    ($self:expr, $ty:ty) => {{
        assert_impl_one!($ty: OpParam);
        let data = $self
            .program
            .as_ptr()
            .add($self.pc + 1)
            .cast::<$ty>()
            .read();
        $self.pc += 1 + core::mem::size_of::<$ty>();
        data
    }};
}

/// Perform binary operation `#0 ← #1 OP #2`
macro_rules! binary_op {
    ($self:expr, $ty:ident, $handler:expr) => {{
        let ParamBBB(tg, a0, a1) = param!($self, ParamBBB);
        $self.write_reg(
            tg,
            $handler(
                Value::$ty(&$self.read_reg(a0)),
                Value::$ty(&$self.read_reg(a1)),
            ),
        );
    }};

    ($self:expr, $ty:ident, $handler:expr, $con:ty) => {{
        let ParamBBB(tg, a0, a1) = param!($self, ParamBBB);
        $self.write_reg(
            tg,
            $handler(
                Value::$ty(&$self.read_reg(a0)),
                Value::$ty(&$self.read_reg(a1)) as $con,
            ),
        );
    }};
}

/// Perform binary operation with immediate `#0 ← #1 OP imm #2`
macro_rules! binary_op_imm {
    ($self:expr, $ty:ident, $handler:expr) => {{
        let ParamBBD(tg, a0, imm) = param!($self, ParamBBD);
        $self.write_reg(
            tg,
            $handler(Value::$ty(&$self.read_reg(a0)), Value::$ty(&imm.into())),
        );
    }};

    ($self:expr, $ty:ident, $handler:expr, $con:ty) => {{
        let ParamBBD(tg, a0, imm) = param!($self, ParamBBD);
        $self.write_reg(
            tg,
            $handler(Value::$ty(&$self.read_reg(a0)), Value::$ty(&imm.into()) as $con),
        );
    }};
}

/// Jump at `#3` if ordering on `#0 <=> #1` is equal to expected
macro_rules! cond_jump {
    ($self:expr, $ty:ident, $expected:ident) => {{
        let ParamBBD(a0, a1, jt) = param!($self, ParamBBD);
        if core::cmp::Ord::cmp(&$self.read_reg(a0).as_u64(), &$self.read_reg(a1).as_u64())
            == core::cmp::Ordering::$expected
        {
            $self.pc = jt as usize;
        }
    }};
}

/// HoleyBytes Virtual Machine
pub struct Vm<'a, PfHandler, const TIMER_QUOTIENT: usize> {
    /// Holds 256 registers
    ///
    /// Writing to register 0 is considered undefined behaviour
    /// in terms of HoleyBytes program execution
    pub registers: [Value; 256],

    /// Memory implementation
    pub memory: Memory,

    /// Trap handler
    pub pfhandler: PfHandler,

    /// Program counter
    pc: usize,

    /// Program
    program: &'a [u8],

    /// Cached program length (without unreachable end)
    program_len: usize,

    /// Program timer
    timer: usize,
}

impl<'a, PfHandler: HandlePageFault, const TIMER_QUOTIENT: usize>
    Vm<'a, PfHandler, TIMER_QUOTIENT>
{
    /// Create a new VM with program and trap handler
    ///
    /// # Safety
    /// Program code has to be validated
    pub unsafe fn new_unchecked(program: &'a [u8], traph: PfHandler) -> Self {
        Self {
            registers: [Value::from(0_u64); 256],
            memory: Default::default(),
            pfhandler: traph,
            pc: 0,
            program_len: program.len() - 12,
            program,
            timer: 0,
        }
    }

    /// Create a new VM with program and trap handler only if it passes validation
    pub fn new_validated(program: &'a [u8], traph: PfHandler) -> Result<Self, validate::Error> {
        validate::validate(program)?;
        Ok(unsafe { Self::new_unchecked(program, traph) })
    }

    /// Execute program
    ///
    /// Program can return [`VmRunError`] if a trap handling failed
    pub fn run(&mut self) -> Result<VmRunOk, VmRunError> {
        use hbbytecode::opcode::*;
        loop {
            // Check instruction boundary
            if self.pc >= self.program_len {
                return Ok(VmRunOk::End);
            }

            // Big match
            //
            // Contribution guide:
            // - Zero register shall never be overwitten. It's value has to always be 0.
            //     - Prefer `Self::read_reg` and `Self::write_reg` functions
            // - Extract parameters using `param!` macro
            // - Prioritise speed over code size
            //     - Memory is cheap, CPUs not that much
            // - Do not heap allocate at any cost
            //     - Yes, user-provided trap handler may allocate,
            //       but that is not our »fault«.
            // - Unsafe is kinda must, but be sure you have validated everything
            //     - Your contributions have to pass sanitizers and Miri
            // - Strictly follow the spec
            //     - The spec does not specify how you perform actions, in what order,
            //       just that the observable effects have to be performed in order and
            //       correctly.
            // - Yes, we assume you run 64 bit CPU. Else ?conradluget a better CPU
            //   sorry 8 bit fans, HBVM won't run on your Speccy :(
            unsafe {
                match *self.program.get_unchecked(self.pc) {
                    UN => {
                        param!(self, ());
                        return Err(VmRunError::Unreachable);
                    }
                    NOP => param!(self, ()),
                    ADD => binary_op!(self, as_u64, u64::wrapping_add),
                    SUB => binary_op!(self, as_u64, u64::wrapping_sub),
                    MUL => binary_op!(self, as_u64, u64::wrapping_mul),
                    AND => binary_op!(self, as_u64, ops::BitAnd::bitand),
                    OR => binary_op!(self, as_u64, ops::BitOr::bitor),
                    XOR => binary_op!(self, as_u64, ops::BitXor::bitxor),
                    SL => binary_op!(self, as_u64, u64::wrapping_shl, u32),
                    SR => binary_op!(self, as_u64, u64::wrapping_shr, u32),
                    SRS => binary_op!(self, as_i64, i64::wrapping_shr, u32),
                    CMP => {
                        // Compare a0 <=> a1
                        // < → -1
                        // > →  1
                        // = →  0

                        let ParamBBB(tg, a0, a1) = param!(self, ParamBBB);
                        self.write_reg(
                            tg,
                            self.read_reg(a0).as_i64().cmp(&self.read_reg(a1).as_i64()) as i64,
                        );
                    }
                    CMPU => {
                        // Unsigned comparsion
                        let ParamBBB(tg, a0, a1) = param!(self, ParamBBB);
                        self.write_reg(
                            tg,
                            self.read_reg(a0).as_u64().cmp(&self.read_reg(a1).as_u64()) as i64,
                        );
                    }
                    NOT => {
                        // Logical negation
                        let param = param!(self, ParamBB);
                        self.write_reg(param.0, !self.read_reg(param.1).as_u64());
                    }
                    NEG => {
                        // Bitwise negation
                        let param = param!(self, ParamBB);
                        self.write_reg(
                            param.0,
                            match self.read_reg(param.1).as_u64() {
                                0 => 1_u64,
                                _ => 0,
                            },
                        );
                    }
                    DIR => {
                        // Fused Division-Remainder
                        let ParamBBBB(dt, rt, a0, a1) = param!(self, ParamBBBB);
                        let a0 = self.read_reg(a0).as_u64();
                        let a1 = self.read_reg(a1).as_u64();
                        self.write_reg(dt, a0.checked_div(a1).unwrap_or(u64::MAX));
                        self.write_reg(rt, a0.checked_rem(a1).unwrap_or(u64::MAX));
                    }
                    ADDI => binary_op_imm!(self, as_u64, ops::Add::add),
                    MULI => binary_op_imm!(self, as_u64, ops::Mul::mul),
                    ANDI => binary_op_imm!(self, as_u64, ops::BitAnd::bitand),
                    ORI => binary_op_imm!(self, as_u64, ops::BitOr::bitor),
                    XORI => binary_op_imm!(self, as_u64, ops::BitXor::bitxor),
                    SLI => binary_op_imm!(self, as_u64, u64::wrapping_shl, u32),
                    SRI => binary_op_imm!(self, as_u64, u64::wrapping_shr, u32),
                    SRSI => binary_op_imm!(self, as_i64, i64::wrapping_shr, u32),
                    CMPI => {
                        let ParamBBD(tg, a0, imm) = param!(self, ParamBBD);
                        self.write_reg(
                            tg,
                            self.read_reg(a0).as_i64().cmp(&Value::from(imm).as_i64()) as i64,
                        );
                    }
                    CMPUI => {
                        let ParamBBD(tg, a0, imm) = param!(self, ParamBBD);
                        self.write_reg(tg, self.read_reg(a0).as_u64().cmp(&imm) as i64);
                    }
                    CP => {
                        let param = param!(self, ParamBB);
                        self.write_reg(param.0, self.read_reg(param.1));
                    }
                    SWA => {
                        // Swap registers
                        let ParamBB(r0, r1) = param!(self, ParamBB);
                        match (r0, r1) {
                            (0, 0) => (),
                            (dst, 0) | (0, dst) => self.write_reg(dst, 0_u64),
                            (r0, r1) => {
                                core::ptr::swap(
                                    self.registers.get_unchecked_mut(usize::from(r0)),
                                    self.registers.get_unchecked_mut(usize::from(r1)),
                                );
                            }
                        }
                    }
                    LI => {
                        let param = param!(self, ParamBD);
                        self.write_reg(param.0, param.1);
                    }
                    LD => {
                        // Load. If loading more than register size, continue on adjecent registers
                        let ParamBBDH(dst, base, off, count) = param!(self, ParamBBDH);
                        let n: usize = match dst {
                            0 => 1,
                            _ => 0,
                        };

                        self.memory.load(
                            self.read_reg(base).as_u64() + off + n as u64,
                            self.registers.as_mut_ptr().add(usize::from(dst) + n).cast(),
                            usize::from(count).saturating_sub(n),
                            &mut self.pfhandler,
                        )?;
                    }
                    ST => {
                        // Store. Same rules apply as to LD
                        let ParamBBDH(dst, base, off, count) = param!(self, ParamBBDH);
                        self.memory.store(
                            self.read_reg(base).as_u64() + off,
                            self.registers.as_ptr().add(usize::from(dst)).cast(),
                            count.into(),
                            &mut self.pfhandler,
                        )?;
                    }
                    BMC => {
                        // Block memory copy
                        let ParamBBD(src, dst, count) = param!(self, ParamBBD);
                        self.memory.block_copy(
                            self.read_reg(src).as_u64(),
                            self.read_reg(dst).as_u64(),
                            count as _,
                            &mut self.pfhandler,
                        )?;
                    }
                    BRC => {
                        // Block register copy
                        let ParamBBB(src, dst, count) = param!(self, ParamBBB);
                        core::ptr::copy(
                            self.registers.get_unchecked(usize::from(src)),
                            self.registers.get_unchecked_mut(usize::from(dst)),
                            usize::from(count),
                        );
                    }
                    JAL => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg + offset.
                        let ParamBBD(save, reg, offset) = param!(self, ParamBBD);
                        self.write_reg(save, self.pc as u64);
                        self.pc = (self.read_reg(reg).as_u64() + offset) as usize;
                    }
                    // Conditional jumps, jump only to immediates
                    JEQ => cond_jump!(self, int, Equal),
                    JNE => {
                        let ParamBBD(a0, a1, jt) = param!(self, ParamBBD);
                        if self.read_reg(a0).as_u64() != self.read_reg(a1).as_u64() {
                            self.pc = jt as usize;
                        }
                    }
                    JLT => cond_jump!(self, int, Less),
                    JGT => cond_jump!(self, int, Greater),
                    JLTU => cond_jump!(self, sint, Less),
                    JGTU => cond_jump!(self, sint, Greater),
                    ECALL => {
                        param!(self, ());

                        // So we don't get timer interrupt after ECALL
                        if TIMER_QUOTIENT != 0 {
                            self.timer = self.timer.wrapping_add(1);
                        }
                        return Ok(VmRunOk::Ecall);
                    }
                    ADDF => binary_op!(self, as_f64, ops::Add::add),
                    SUBF => binary_op!(self, as_f64, ops::Sub::sub),
                    MULF => binary_op!(self, as_f64, ops::Mul::mul),
                    DIRF => {
                        let ParamBBBB(dt, rt, a0, a1) = param!(self, ParamBBBB);
                        let a0 = self.read_reg(a0).as_f64();
                        let a1 = self.read_reg(a1).as_f64();
                        self.write_reg(dt, a0 / a1);
                        self.write_reg(rt, a0 % a1);
                    }
                    FMAF => {
                        let ParamBBBB(dt, a0, a1, a2) = param!(self, ParamBBBB);
                        self.write_reg(
                            dt,
                            self.read_reg(a0).as_f64() * self.read_reg(a1).as_f64()
                                + self.read_reg(a2).as_f64(),
                        );
                    }
                    NEGF => {
                        let ParamBB(dt, a0) = param!(self, ParamBB);
                        self.write_reg(dt, -self.read_reg(a0).as_f64());
                    }
                    ITF => {
                        let ParamBB(dt, a0) = param!(self, ParamBB);
                        self.write_reg(dt, self.read_reg(a0).as_i64() as f64);
                    }
                    FTI => {
                        let ParamBB(dt, a0) = param!(self, ParamBB);
                        self.write_reg(dt, self.read_reg(a0).as_f64() as i64);
                    }
                    ADDFI => binary_op_imm!(self, as_f64, ops::Add::add),
                    MULFI => binary_op_imm!(self, as_f64, ops::Mul::mul),
                    op => return Err(VmRunError::InvalidOpcode(op)),
                }
            }

            if TIMER_QUOTIENT != 0 {
                self.timer = self.timer.wrapping_add(1);
                if self.timer % TIMER_QUOTIENT == 0 {
                    return Ok(VmRunOk::Timer);
                }
            }
        }
    }

    /// Read register
    #[inline]
    unsafe fn read_reg(&self, n: u8) -> Value {
        *self.registers.get_unchecked(n as usize)
    }

    /// Write a register.
    /// Writing to register 0 is no-op.
    #[inline]
    unsafe fn write_reg(&mut self, n: u8, value: impl Into<Value>) {
        if n != 0 {
            *self.registers.get_unchecked_mut(n as usize) = value.into();
        }
    }
}

/// Virtual machine halt error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum VmRunError {
    /// Tried to execute invalid instruction
    InvalidOpcode(u8),

    /// Unhandled load access exception
    LoadAccessEx(u64),

    /// Unhandled store access exception
    StoreAccessEx(u64),

    /// Reached unreachable code
    Unreachable,
}

/// Virtual machine halt ok
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VmRunOk {
    /// Program has eached its end
    End,

    /// Program was interrupted by a timer
    Timer,

    /// Environment call
    Ecall,
}
