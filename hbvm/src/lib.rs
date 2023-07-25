//! HoleyBytes Virtual Machine
//!
//! # Alloc feature
//! - Enabled by default
//! - Provides [`mem::Memory`] mapping / unmapping, as well as
//!   [`Default`] and [`Drop`] implementation

// # General safety notice:
// - Validation has to assure there is 256 registers (r0 - r255)
// - Instructions have to be valid as specified (values and sizes)
// - Mapped pages should be at least 4 KiB

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod mem;
pub mod value;

use {
    self::{mem::HandlePageFault, value::ValueVariant},
    core::{cmp::Ordering, ops},
    hbbytecode::{
        valider, OpParam, ParamBB, ParamBBB, ParamBBBB, ParamBBD, ParamBBDH, ParamBBW, ParamBD,
    },
    mem::Memory,
    value::Value,
};

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
    pub pc: usize,

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
    pub unsafe fn new_unchecked(program: &'a [u8], traph: PfHandler, memory: Memory) -> Self {
        Self {
            registers: [Value::from(0_u64); 256],
            memory,
            pfhandler: traph,
            pc: 0,
            program_len: program.len() - 12,
            program,
            timer: 0,
        }
    }

    /// Create a new VM with program and trap handler only if it passes validation
    pub fn new_validated(
        program: &'a [u8],
        traph: PfHandler,
        memory: Memory,
    ) -> Result<Self, valider::Error> {
        valider::validate(program)?;
        Ok(unsafe { Self::new_unchecked(program, traph, memory) })
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
                        self.decode::<()>();
                        return Err(VmRunError::Unreachable);
                    }
                    NOP => self.decode::<()>(),
                    ADD => self.binary_op(u64::wrapping_add),
                    SUB => self.binary_op(u64::wrapping_sub),
                    MUL => self.binary_op(u64::wrapping_mul),
                    AND => self.binary_op::<u64>(ops::BitAnd::bitand),
                    OR => self.binary_op::<u64>(ops::BitOr::bitor),
                    XOR => self.binary_op::<u64>(ops::BitXor::bitxor),
                    SL => self.binary_op(|l, r| u64::wrapping_shl(l, r as u32)),
                    SR => self.binary_op(|l, r| u64::wrapping_shr(l, r as u32)),
                    SRS => self.binary_op(|l, r| i64::wrapping_shl(l, r as u32)),
                    CMP => {
                        // Compare a0 <=> a1
                        // < → -1
                        // > →  1
                        // = →  0

                        let ParamBBB(tg, a0, a1) = self.decode();
                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<i64>()
                                .cmp(&self.read_reg(a1).cast::<i64>())
                                as i64,
                        );
                    }
                    CMPU => {
                        // Unsigned comparsion
                        let ParamBBB(tg, a0, a1) = self.decode();
                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<u64>()
                                .cmp(&self.read_reg(a1).cast::<u64>())
                                as i64,
                        );
                    }
                    NOT => {
                        // Logical negation
                        let ParamBB(tg, a0) = self.decode();
                        self.write_reg(tg, !self.read_reg(a0).cast::<u64>());
                    }
                    NEG => {
                        // Bitwise negation
                        let ParamBB(tg, a0) = self.decode();
                        self.write_reg(
                            tg,
                            match self.read_reg(a0).cast::<u64>() {
                                0 => 1_u64,
                                _ => 0,
                            },
                        );
                    }
                    DIR => {
                        // Fused Division-Remainder
                        let ParamBBBB(dt, rt, a0, a1) = self.decode();
                        let a0 = self.read_reg(a0).cast::<u64>();
                        let a1 = self.read_reg(a1).cast::<u64>();
                        self.write_reg(dt, a0.checked_div(a1).unwrap_or(u64::MAX));
                        self.write_reg(rt, a0.checked_rem(a1).unwrap_or(u64::MAX));
                    }
                    ADDI => self.binary_op_imm(u64::wrapping_add),
                    MULI => self.binary_op_imm(u64::wrapping_sub),
                    ANDI => self.binary_op_imm::<u64>(ops::BitAnd::bitand),
                    ORI => self.binary_op_imm::<u64>(ops::BitOr::bitor),
                    XORI => self.binary_op_imm::<u64>(ops::BitXor::bitxor),
                    SLI => self.binary_op_ims(u64::wrapping_shl),
                    SRI => self.binary_op_ims(u64::wrapping_shr),
                    SRSI => self.binary_op_ims(i64::wrapping_shr),
                    CMPI => {
                        let ParamBBD(tg, a0, imm) = self.decode();
                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<i64>()
                                .cmp(&Value::from(imm).cast::<i64>())
                                as i64,
                        );
                    }
                    CMPUI => {
                        let ParamBBD(tg, a0, imm) = self.decode();
                        self.write_reg(tg, self.read_reg(a0).cast::<u64>().cmp(&imm) as i64);
                    }
                    CP => {
                        let ParamBB(tg, a0) = self.decode();
                        self.write_reg(tg, self.read_reg(a0));
                    }
                    SWA => {
                        // Swap registers
                        let ParamBB(r0, r1) = self.decode();
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
                        let ParamBD(tg, imm) = self.decode();
                        self.write_reg(tg, imm);
                    }
                    LD => {
                        // Load. If loading more than register size, continue on adjecent registers
                        let ParamBBDH(dst, base, off, count) = self.decode();
                        ldst_bound_check(dst, count)?;

                        let n: usize = match dst {
                            0 => 1,
                            _ => 0,
                        };

                        self.memory.load(
                            self.read_reg(base).cast::<u64>() + off + n as u64,
                            self.registers.as_mut_ptr().add(usize::from(dst) + n).cast(),
                            usize::from(count).saturating_sub(n),
                            &mut self.pfhandler,
                        )?;
                    }
                    ST => {
                        // Store. Same rules apply as to LD
                        let ParamBBDH(dst, base, off, count) = self.decode();
                        ldst_bound_check(dst, count)?;

                        self.memory.store(
                            self.read_reg(base).cast::<u64>() + off,
                            self.registers.as_ptr().add(usize::from(dst)).cast(),
                            count.into(),
                            &mut self.pfhandler,
                        )?;
                    }
                    BMC => {
                        // Block memory copy
                        let ParamBBD(src, dst, count) = self.decode();
                        self.memory.block_copy(
                            self.read_reg(src).cast::<u64>(),
                            self.read_reg(dst).cast::<u64>(),
                            count as _,
                            &mut self.pfhandler,
                        )?;
                    }
                    BRC => {
                        // Block register copy
                        let ParamBBB(src, dst, count) = self.decode();
                        if src.checked_add(count).is_none() || dst.checked_add(count).is_none() {
                            return Err(VmRunError::RegOutOfBounds);
                        }

                        core::ptr::copy(
                            self.registers.get_unchecked(usize::from(src)),
                            self.registers.get_unchecked_mut(usize::from(dst)),
                            usize::from(count),
                        );
                    }
                    JAL => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg + offset.
                        let ParamBBD(save, reg, offset) = self.decode();
                        self.write_reg(save, self.pc as u64);
                        self.pc = (self.read_reg(reg).cast::<u64>() + offset) as usize;
                    }
                    // Conditional jumps, jump only to immediates
                    JEQ => self.cond_jmp::<u64>(Ordering::Equal),
                    JNE => {
                        let ParamBBD(a0, a1, jt) = self.decode();
                        if self.read_reg(a0).cast::<u64>() != self.read_reg(a1).cast::<u64>() {
                            self.pc = jt as usize;
                        }
                    }
                    JLT => self.cond_jmp::<u64>(Ordering::Less),
                    JGT => self.cond_jmp::<u64>(Ordering::Greater),
                    JLTU => self.cond_jmp::<i64>(Ordering::Less),
                    JGTU => self.cond_jmp::<i64>(Ordering::Greater),
                    ECALL => {
                        self.decode::<()>();

                        // So we don't get timer interrupt after ECALL
                        if TIMER_QUOTIENT != 0 {
                            self.timer = self.timer.wrapping_add(1);
                        }
                        return Ok(VmRunOk::Ecall);
                    }
                    ADDF => self.binary_op::<f64>(ops::Add::add),
                    SUBF => self.binary_op::<f64>(ops::Sub::sub),
                    MULF => self.binary_op::<f64>(ops::Mul::mul),
                    DIRF => {
                        let ParamBBBB(dt, rt, a0, a1) = self.decode();
                        let a0 = self.read_reg(a0).cast::<f64>();
                        let a1 = self.read_reg(a1).cast::<f64>();
                        self.write_reg(dt, a0 / a1);
                        self.write_reg(rt, a0 % a1);
                    }
                    FMAF => {
                        let ParamBBBB(dt, a0, a1, a2) = self.decode();
                        self.write_reg(
                            dt,
                            self.read_reg(a0).cast::<f64>() * self.read_reg(a1).cast::<f64>()
                                + self.read_reg(a2).cast::<f64>(),
                        );
                    }
                    NEGF => {
                        let ParamBB(dt, a0) = self.decode();
                        self.write_reg(dt, -self.read_reg(a0).cast::<f64>());
                    }
                    ITF => {
                        let ParamBB(dt, a0) = self.decode();
                        self.write_reg(dt, self.read_reg(a0).cast::<i64>() as f64);
                    }
                    FTI => {
                        let ParamBB(dt, a0) = self.decode();
                        self.write_reg(dt, self.read_reg(a0).cast::<f64>() as i64);
                    }
                    ADDFI => self.binary_op_imm::<f64>(ops::Add::add),
                    MULFI => self.binary_op_imm::<f64>(ops::Mul::mul),
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

    /// Decode instruction operands
    #[inline]
    unsafe fn decode<T: OpParam>(&mut self) -> T {
        let data = self.program.as_ptr().add(self.pc + 1).cast::<T>().read();
        self.pc += 1 + core::mem::size_of::<T>();
        data
    }

    /// Perform binary operating over two registers
    #[inline]
    unsafe fn binary_op<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let ParamBBB(tg, a0, a1) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(a0).cast::<T>(), self.read_reg(a1).cast::<T>()),
        );
    }

    /// Perform binary operation over register and immediate
    #[inline]
    unsafe fn binary_op_imm<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let ParamBBD(tg, reg, imm) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(reg).cast::<T>(), Value::from(imm).cast::<T>()),
        );
    }

    /// Perform binary operation over register and shift immediate
    #[inline]
    unsafe fn binary_op_ims<T: ValueVariant>(&mut self, op: impl Fn(T, u32) -> T) {
        let ParamBBW(tg, reg, imm) = self.decode();
        self.write_reg(tg, op(self.read_reg(reg).cast::<T>(), imm));
    }

    /// Jump at `#3` if ordering on `#0 <=> #1` is equal to expected
    #[inline]
    unsafe fn cond_jmp<T: ValueVariant + Ord>(&mut self, expected: Ordering) {
        let ParamBBD(a0, a1, ja) = self.decode();
        if self
            .read_reg(a0)
            .cast::<T>()
            .cmp(&self.read_reg(a1).cast::<T>())
            == expected
        {
            self.pc = ja as usize;
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

/// Load/Store target/source register range bound checking
#[inline]
fn ldst_bound_check(reg: u8, size: u16) -> Result<(), VmRunError> {
    if usize::from(reg) * 8 + usize::from(size) > 2048 {
        Err(VmRunError::RegOutOfBounds)
    } else {
        Ok(())
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

    /// Register out-of-bounds access
    RegOutOfBounds,

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
