//! Welcome to the land of The Great Dispatch Loop
//!
//! Have fun

use {
    super::{bmc::BlockCopier, mem::Memory, value::ValueVariant, Vm, VmRunError, VmRunOk},
    crate::{
        mem::{addr::AddressOp, Address},
        value::CheckedDivRem,
    },
    core::{cmp::Ordering, ops},
    hbbytecode::{
        OpsN, OpsO, OpsP, OpsRB, OpsRD, OpsRH, OpsRR, OpsRRA, OpsRRAH, OpsRRB, OpsRRD, OpsRRH,
        OpsRRO, OpsRROH, OpsRRP, OpsRRPH, OpsRRR, OpsRRRR, OpsRW, RoundingMode,
    },
};

macro_rules! handler {
    ($self:expr, |$ty:ident ($($ident:pat),* $(,)?)| $expr:expr) => {{
        #[allow(unused_unsafe)]
        let $ty($($ident),*) = unsafe { $self.decode::<$ty>() };
        #[allow(clippy::no_effect)] let e = $expr;
        $self.bump_pc::<$ty>();
        e
    }};
}

impl<Mem, const TIMER_QUOTIENT: usize> Vm<Mem, TIMER_QUOTIENT>
where
    Mem: Memory,
{
    /// Execute program
    ///
    /// Program can return [`VmRunError`] if a trap handling failed
    #[cfg_attr(feature = "nightly", repr(align(4096)))]
    pub fn run(&mut self) -> Result<VmRunOk, VmRunError> {
        use hbbytecode::opcode::*;
        loop {
            // Big match
            //
            // Contribution guide:
            // - Zero register shall never be overwitten. It's value has to always be 0.
            //     - Prefer `Self::read_reg` and `Self::write_reg` functions
            // - Try to use `handler!` macro for decoding and then bumping program counter
            // - Prioritise speed over code size
            //     - Memory is cheap, CPUs not that much
            // - Do not heap allocate at any cost
            //     - Yes, user-provided trap handler may allocate,
            //       but that is not our »fault«.
            // - Unsafe is kinda must, but be sure you have validated everything
            //     - Your contributions have to pass sanitizers, fuzzer and Miri
            // - Strictly follow the spec
            //     - The spec does not specify how you perform actions, in what order,
            //       just that the observable effects have to be performed in order and
            //       correctly.
            // - Yes, we assume you run 64 bit CPU. Else ?conradluget a better CPU
            //   sorry 8 bit fans, HBVM won't run on your Speccy :(
            unsafe {
                match self.memory.prog_read::<u8>(self.pc as _) {
                    UN => {
                        self.bump_pc::<OpsN>();
                        return Err(VmRunError::Unreachable);
                    }
                    TX => {
                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::End);
                    }
                    NOP => handler!(self, |OpsN()| ()),
                    ADD8 => self.binary_op(u8::wrapping_add),
                    ADD16 => self.binary_op(u16::wrapping_add),
                    ADD32 => self.binary_op(u32::wrapping_add),
                    ADD64 => self.binary_op(u64::wrapping_add),
                    SUB8 => self.binary_op(u8::wrapping_sub),
                    SUB16 => self.binary_op(u16::wrapping_sub),
                    SUB32 => self.binary_op(u32::wrapping_sub),
                    SUB64 => self.binary_op(u64::wrapping_sub),
                    MUL8 => self.binary_op(u8::wrapping_mul),
                    MUL16 => self.binary_op(u16::wrapping_mul),
                    MUL32 => self.binary_op(u32::wrapping_mul),
                    MUL64 => self.binary_op(u64::wrapping_mul),
                    AND => self.binary_op::<u64>(ops::BitAnd::bitand),
                    OR => self.binary_op::<u64>(ops::BitOr::bitor),
                    XOR => self.binary_op::<u64>(ops::BitXor::bitxor),
                    SLU8 => self.binary_op_shift::<u8>(u8::wrapping_shl),
                    SLU16 => self.binary_op_shift::<u16>(u16::wrapping_shl),
                    SLU32 => self.binary_op_shift::<u32>(u32::wrapping_shl),
                    SLU64 => self.binary_op_shift::<u64>(u64::wrapping_shl),
                    SRU8 => self.binary_op_shift::<u8>(u8::wrapping_shr),
                    SRU16 => self.binary_op_shift::<u16>(u16::wrapping_shr),
                    SRU32 => self.binary_op_shift::<u32>(u32::wrapping_shr),
                    SRU64 => self.binary_op_shift::<u64>(u64::wrapping_shr),
                    SRS8 => self.binary_op_shift::<i8>(i8::wrapping_shr),
                    SRS16 => self.binary_op_shift::<i16>(i16::wrapping_shr),
                    SRS32 => self.binary_op_shift::<i32>(i32::wrapping_shr),
                    SRS64 => self.binary_op_shift::<i64>(i64::wrapping_shr),
                    CMPU => handler!(self, |OpsRRR(tg, a0, a1)| self.cmp(
                        tg,
                        a0,
                        self.read_reg(a1).cast::<u64>()
                    )),
                    CMPS => handler!(self, |OpsRRR(tg, a0, a1)| self.cmp(
                        tg,
                        a0,
                        self.read_reg(a1).cast::<i64>()
                    )),
                    DIRU8 => self.dir::<u8>(),
                    DIRU16 => self.dir::<u16>(),
                    DIRU32 => self.dir::<u32>(),
                    DIRU64 => self.dir::<u64>(),
                    DIRS8 => self.dir::<i8>(),
                    DIRS16 => self.dir::<i16>(),
                    DIRS32 => self.dir::<i32>(),
                    DIRS64 => self.dir::<i64>(),
                    NEG => handler!(self, |OpsRR(tg, a0)| {
                        // Bit negation
                        self.write_reg(tg, !self.read_reg(a0).cast::<u64>())
                    }),
                    NOT => handler!(self, |OpsRR(tg, a0)| {
                        // Logical negation
                        self.write_reg(tg, u64::from(self.read_reg(a0).cast::<u64>() == 0));
                    }),
                    SXT8 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i8>() as i64)
                    }),
                    SXT16 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i16>() as i64)
                    }),
                    SXT32 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i32>() as i64)
                    }),
                    ADDI8 => self.binary_op_imm(u8::wrapping_add),
                    ADDI16 => self.binary_op_imm(u16::wrapping_add),
                    ADDI32 => self.binary_op_imm(u32::wrapping_add),
                    ADDI64 => self.binary_op_imm(u64::wrapping_add),
                    MULI8 => self.binary_op_imm(u8::wrapping_mul),
                    MULI16 => self.binary_op_imm(u16::wrapping_mul),
                    MULI32 => self.binary_op_imm(u32::wrapping_mul),
                    MULI64 => self.binary_op_imm(u64::wrapping_mul),
                    ANDI => self.binary_op_imm::<u64>(ops::BitAnd::bitand),
                    ORI => self.binary_op_imm::<u64>(ops::BitOr::bitor),
                    XORI => self.binary_op_imm::<u64>(ops::BitXor::bitxor),
                    SLUI8 => self.binary_op_ims::<u8>(u8::wrapping_shl),
                    SLUI16 => self.binary_op_ims::<u16>(u16::wrapping_shl),
                    SLUI32 => self.binary_op_ims::<u32>(u32::wrapping_shl),
                    SLUI64 => self.binary_op_ims::<u64>(u64::wrapping_shl),
                    SRUI8 => self.binary_op_ims::<u8>(u8::wrapping_shr),
                    SRUI16 => self.binary_op_ims::<u16>(u16::wrapping_shr),
                    SRUI32 => self.binary_op_ims::<u32>(u32::wrapping_shr),
                    SRUI64 => self.binary_op_ims::<u64>(u64::wrapping_shr),
                    SRSI8 => self.binary_op_ims::<i8>(i8::wrapping_shr),
                    SRSI16 => self.binary_op_ims::<i16>(i16::wrapping_shr),
                    SRSI32 => self.binary_op_ims::<i32>(i32::wrapping_shr),
                    SRSI64 => self.binary_op_ims::<i64>(i64::wrapping_shr),
                    CMPUI => handler!(self, |OpsRRD(tg, a0, imm)| { self.cmp(tg, a0, imm) }),
                    CMPSI => handler!(self, |OpsRRD(tg, a0, imm)| { self.cmp(tg, a0, imm as i64) }),
                    CP => handler!(self, |OpsRR(tg, a0)| self.write_reg(tg, self.read_reg(a0))),
                    SWA => handler!(self, |OpsRR(r0, r1)| {
                        // Swap registers
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
                    }),
                    LI8 => handler!(self, |OpsRB(tg, imm)| self.write_reg(tg, imm)),
                    LI16 => handler!(self, |OpsRH(tg, imm)| self.write_reg(tg, imm)),
                    LI32 => handler!(self, |OpsRW(tg, imm)| self.write_reg(tg, imm)),
                    LI64 => handler!(self, |OpsRD(tg, imm)| self.write_reg(tg, imm)),
                    LRA => handler!(self, |OpsRRO(tg, reg, off)| self.write_reg(
                        tg,
                        self.pcrel(off).wrapping_add(self.read_reg(reg).cast::<i64>()).get(),
                    )),
                    // Load. If loading more than register size, continue on adjecent registers
                    LD => handler!(self, |OpsRRAH(dst, base, off, count)| self
                        .load(dst, base, off, count)?),
                    // Store. Same rules apply as to LD
                    ST => handler!(self, |OpsRRAH(dst, base, off, count)| self
                        .store(dst, base, off, count)?),
                    LDR => handler!(self, |OpsRROH(dst, base, off, count)| self.load(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    STR => handler!(self, |OpsRROH(dst, base, off, count)| self.store(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    BMC => {
                        // Block memory copy
                        match if let Some(copier) = &mut self.copier {
                            // There is some copier, poll.
                            copier.poll(&mut self.memory)
                        } else {
                            // There is none, make one!
                            let OpsRRH(src, dst, count) = self.decode();

                            self.copier = Some(BlockCopier::new(
                                Address::new(self.read_reg(src).cast()),
                                Address::new(self.read_reg(dst).cast()),
                                count as _,
                            ));

                            self.copier
                                .as_mut()
                                .unwrap_unchecked() // SAFETY: We just assigned there
                                .poll(&mut self.memory)
                        } {
                            // We are done, shift program counter
                            core::task::Poll::Ready(Ok(())) => {
                                self.copier = None;
                                self.bump_pc::<OpsRRH>();
                            }
                            // Error, shift program counter (for consistency)
                            // and yield error
                            core::task::Poll::Ready(Err(e)) => {
                                return Err(e.into());
                            }
                            // Not done yet, proceed to next cycle
                            core::task::Poll::Pending => (),
                        }
                    }
                    BRC => handler!(self, |OpsRRB(src, dst, count)| {
                        // Block register copy
                        if src.checked_add(count).is_none() || dst.checked_add(count).is_none() {
                            return Err(VmRunError::RegOutOfBounds);
                        }

                        core::ptr::copy(
                            self.registers.get_unchecked(usize::from(src)),
                            self.registers.get_unchecked_mut(usize::from(dst)),
                            usize::from(count),
                        );
                    }),
                    JMP => {
                        let OpsO(off) = self.decode();
                        self.pc = self.pc.wrapping_add(off);
                    }
                    JAL => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg + relative offset.
                        let OpsRRO(save, reg, offset) = self.decode();

                        self.write_reg(save, self.pc.next::<OpsRRO>());
                        self.pc = self.pcrel(offset).wrapping_add(self.read_reg(reg).cast::<i64>());
                    }
                    JALA => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg
                        let OpsRRA(save, reg, offset) = self.decode();

                        self.write_reg(save, self.pc.next::<OpsRRA>());
                        self.pc =
                            Address::new(self.read_reg(reg).cast::<u64>().wrapping_add(offset));
                    }
                    // Conditional jumps, jump only to immediates
                    JEQ => self.cond_jmp::<u64>(Ordering::Equal),
                    JNE => {
                        let OpsRRP(a0, a1, ja) = self.decode();
                        if self.read_reg(a0).cast::<u64>() != self.read_reg(a1).cast::<u64>() {
                            self.pc = self.pcrel(ja);
                        } else {
                            self.bump_pc::<OpsRRP>();
                        }
                    }
                    JLTS => self.cond_jmp::<u64>(Ordering::Less),
                    JGTS => self.cond_jmp::<u64>(Ordering::Greater),
                    JLTU => self.cond_jmp::<i64>(Ordering::Less),
                    JGTU => self.cond_jmp::<i64>(Ordering::Greater),
                    ECA => {
                        // So we don't get timer interrupt after ECALL
                        if TIMER_QUOTIENT != 0 {
                            self.timer = self.timer.wrapping_add(1);
                        }

                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::Ecall);
                    }
                    EBP => {
                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::Breakpoint);
                    }
                    FADD32 => self.binary_op::<f32>(ops::Add::add),
                    FADD64 => self.binary_op::<f64>(ops::Add::add),
                    FSUB32 => self.binary_op::<f32>(ops::Sub::sub),
                    FSUB64 => self.binary_op::<f64>(ops::Sub::sub),
                    FMUL32 => self.binary_op::<f32>(ops::Mul::mul),
                    FMUL64 => self.binary_op::<f64>(ops::Mul::mul),
                    FDIV32 => self.binary_op::<f32>(ops::Div::div),
                    FDIV64 => self.binary_op::<f64>(ops::Div::div),
                    FMA32 => self.fma::<f32>(),
                    FMA64 => self.fma::<f64>(),
                    FINV32 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, 1. / self.read_reg(reg).cast::<f32>())),
                    FINV64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, 1. / self.read_reg(reg).cast::<f64>())),
                    FCMPLT32 => self.fcmp::<f32>(Ordering::Less),
                    FCMPLT64 => self.fcmp::<f64>(Ordering::Less),
                    FCMPGT32 => self.fcmp::<f32>(Ordering::Greater),
                    FCMPGT64 => self.fcmp::<f64>(Ordering::Greater),
                    ITF32 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<i64>() as f32)),
                    ITF64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<i64>() as f64)),
                    FTI32 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::f32toint(
                            self.read_reg(reg).cast::<f32>(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    FTI64 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::f64toint(
                            self.read_reg(reg).cast::<f64>(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    FC32T64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<f32>() as f64)),
                    FC64T32 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::conv64to32(
                            self.read_reg(reg).cast(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    LRA16 => handler!(self, |OpsRRP(tg, reg, imm)| self.write_reg(
                        tg,
                        (self.pc + self.read_reg(reg).cast::<u64>() + imm + 3_u16).get(),
                    )),
                    LDR16 => handler!(self, |OpsRRPH(dst, base, off, count)| self.load(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    STR16 => handler!(self, |OpsRRPH(dst, base, off, count)| self.store(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    JMP16 => {
                        let OpsP(off) = self.decode();
                        self.pc = self.pcrel(off);
                    }
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

    /// Bump instruction pointer
    #[inline(always)]
    fn bump_pc<T: Copy>(&mut self) {
        self.pc = self.pc.wrapping_add(core::mem::size_of::<T>()).wrapping_add(1);
    }

    /// Decode instruction operands
    #[inline(always)]
    unsafe fn decode<T: Copy>(&mut self) -> T {
        unsafe { self.memory.prog_read::<T>(self.pc + 1_u64) }
    }

    /// Load
    #[inline(always)]
    unsafe fn load(
        &mut self,
        dst: u8,
        base: u8,
        offset: u64,
        count: u16,
    ) -> Result<(), VmRunError> {
        let n: u8 = match dst {
            0 => 1,
            _ => 0,
        };

        unsafe {
            self.memory.load(
                self.ldst_addr_uber(dst, base, offset, count, n)?,
                self.registers.as_mut_ptr().add(usize::from(dst) + usize::from(n)).cast(),
                usize::from(count).saturating_sub(n.into()),
            )
        }?;

        Ok(())
    }

    /// Store
    #[inline(always)]
    unsafe fn store(
        &mut self,
        dst: u8,
        base: u8,
        offset: u64,
        count: u16,
    ) -> Result<(), VmRunError> {
        unsafe {
            self.memory.store(
                self.ldst_addr_uber(dst, base, offset, count, 0)?,
                self.registers.as_ptr().add(usize::from(dst)).cast(),
                count.into(),
            )
        }?;
        Ok(())
    }

    /// Three-way comparsion
    #[inline(always)]
    unsafe fn cmp<T: ValueVariant + Ord>(&mut self, to: u8, reg: u8, val: T) {
        self.write_reg(to, self.read_reg(reg).cast::<T>().cmp(&val) as i64);
    }

    /// Perform binary operating over two registers
    #[inline(always)]
    unsafe fn binary_op<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let OpsRRR(tg, a0, a1) = unsafe { self.decode() };
        self.write_reg(tg, op(self.read_reg(a0).cast::<T>(), self.read_reg(a1).cast::<T>()));
        self.bump_pc::<OpsRRR>();
    }

    /// Perform binary operation over register and immediate
    #[inline(always)]
    unsafe fn binary_op_imm<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        #[derive(Clone, Copy)]
        #[repr(packed)]
        struct OpsRRImm<I>(OpsRR, I);

        let OpsRRImm::<T>(OpsRR(tg, reg), imm) = unsafe { self.decode() };
        self.write_reg(tg, op(self.read_reg(reg).cast::<T>(), imm));
        self.bump_pc::<OpsRRImm<T>>();
    }

    /// Perform binary operation over register and shift immediate
    #[inline(always)]
    unsafe fn binary_op_shift<T: ValueVariant>(&mut self, op: impl Fn(T, u32) -> T) {
        let OpsRRR(tg, a0, a1) = unsafe { self.decode() };
        self.write_reg(tg, op(self.read_reg(a0).cast::<T>(), self.read_reg(a1).cast::<u32>()));
        self.bump_pc::<OpsRRR>();
    }

    /// Perform binary operation over register and shift immediate
    #[inline(always)]
    unsafe fn binary_op_ims<T: ValueVariant>(&mut self, op: impl Fn(T, u32) -> T) {
        let OpsRRB(tg, reg, imm) = unsafe { self.decode() };
        self.write_reg(tg, op(self.read_reg(reg).cast::<T>(), imm.into()));
        self.bump_pc::<OpsRRB>();
    }

    /// Fused division-remainder
    #[inline(always)]
    unsafe fn dir<T: ValueVariant + CheckedDivRem>(&mut self) {
        handler!(self, |OpsRRRR(td, tr, a0, a1)| {
            let a0 = self.read_reg(a0).cast::<T>();
            let a1 = self.read_reg(a1).cast::<T>();

            if let Some(div) = a0.checked_div(a1) {
                self.write_reg(td, div);
            } else {
                self.write_reg(td, -1_i64);
            }

            if let Some(rem) = a0.checked_rem(a1) {
                self.write_reg(tr, rem);
            } else {
                self.write_reg(tr, a0);
            }
        });
    }

    /// Fused multiply-add
    #[inline(always)]
    unsafe fn fma<T>(&mut self)
    where
        T: ValueVariant + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
    {
        handler!(self, |OpsRRRR(tg, a0, a1, a2)| {
            let a0 = self.read_reg(a0).cast::<T>();
            let a1 = self.read_reg(a1).cast::<T>();
            let a2 = self.read_reg(a2).cast::<T>();
            self.write_reg(tg, a0 * a1 + a2)
        });
    }

    /// Float comparsion
    #[inline(always)]
    unsafe fn fcmp<T: PartialOrd + ValueVariant>(&mut self, nan: Ordering) {
        handler!(self, |OpsRRR(tg, a0, a1)| {
            let a0 = self.read_reg(a0).cast::<T>();
            let a1 = self.read_reg(a1).cast::<T>();
            self.write_reg(tg, (a0.partial_cmp(&a1).unwrap_or(nan) as i8 + 1) as u8)
        });
    }

    /// Calculate pc-relative address
    #[inline(always)]
    fn pcrel(&self, offset: impl AddressOp) -> Address {
        self.pc.wrapping_add(offset)
    }

    /// Jump at `PC + #3` if ordering on `#0 <=> #1` is equal to expected
    #[inline(always)]
    unsafe fn cond_jmp<T: ValueVariant + Ord>(&mut self, expected: Ordering) {
        let OpsRRP(a0, a1, ja) = unsafe { self.decode() };
        if self.read_reg(a0).cast::<T>().cmp(&self.read_reg(a1).cast::<T>()) == expected {
            self.pc = self.pcrel(ja);
        } else {
            self.bump_pc::<OpsRRP>();
        }
    }

    /// Load / Store Address check-computation überfunction
    #[inline(always)]
    unsafe fn ldst_addr_uber(
        &self,
        dst: u8,
        base: u8,
        offset: u64,
        size: u16,
        adder: u8,
    ) -> Result<Address, VmRunError> {
        let reg = dst.checked_add(adder).ok_or(VmRunError::RegOutOfBounds)?;

        if usize::from(reg) * 8 + usize::from(size) > 2048 {
            Err(VmRunError::RegOutOfBounds)
        } else {
            self.read_reg(base)
                .cast::<u64>()
                .checked_add(offset)
                .and_then(|x| x.checked_add(adder.into()))
                .ok_or(VmRunError::AddrOutOfBounds)
                .map(Address::new)
        }
    }
}
