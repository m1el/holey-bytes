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
        use hbbytecode::Instr as I;
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
                match self
                    .memory
                    .prog_read::<u8>(self.pc as _)
                    .try_into()
                    .map_err(VmRunError::InvalidOpcode)?
                {
                    I::UN => {
                        self.bump_pc::<OpsN>();
                        return Err(VmRunError::Unreachable);
                    }
                    I::TX => {
                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::End);
                    }
                    I::NOP => handler!(self, |OpsN()| ()),
                    I::ADD8 => self.binary_op(u8::wrapping_add),
                    I::ADD16 => self.binary_op(u16::wrapping_add),
                    I::ADD32 => self.binary_op(u32::wrapping_add),
                    I::ADD64 => self.binary_op(u64::wrapping_add),
                    I::SUB8 => self.binary_op(u8::wrapping_sub),
                    I::SUB16 => self.binary_op(u16::wrapping_sub),
                    I::SUB32 => self.binary_op(u32::wrapping_sub),
                    I::SUB64 => self.binary_op(u64::wrapping_sub),
                    I::MUL8 => self.binary_op(u8::wrapping_mul),
                    I::MUL16 => self.binary_op(u16::wrapping_mul),
                    I::MUL32 => self.binary_op(u32::wrapping_mul),
                    I::MUL64 => self.binary_op(u64::wrapping_mul),
                    I::AND => self.binary_op::<u64>(ops::BitAnd::bitand),
                    I::OR => self.binary_op::<u64>(ops::BitOr::bitor),
                    I::XOR => self.binary_op::<u64>(ops::BitXor::bitxor),
                    I::SLU8 => self.binary_op_shift::<u8>(u8::wrapping_shl),
                    I::SLU16 => self.binary_op_shift::<u16>(u16::wrapping_shl),
                    I::SLU32 => self.binary_op_shift::<u32>(u32::wrapping_shl),
                    I::SLU64 => self.binary_op_shift::<u64>(u64::wrapping_shl),
                    I::SRU8 => self.binary_op_shift::<u8>(u8::wrapping_shr),
                    I::SRU16 => self.binary_op_shift::<u16>(u16::wrapping_shr),
                    I::SRU32 => self.binary_op_shift::<u32>(u32::wrapping_shr),
                    I::SRU64 => self.binary_op_shift::<u64>(u64::wrapping_shr),
                    I::SRS8 => self.binary_op_shift::<i8>(i8::wrapping_shr),
                    I::SRS16 => self.binary_op_shift::<i16>(i16::wrapping_shr),
                    I::SRS32 => self.binary_op_shift::<i32>(i32::wrapping_shr),
                    I::SRS64 => self.binary_op_shift::<i64>(i64::wrapping_shr),
                    I::CMPU => handler!(self, |OpsRRR(tg, a0, a1)| self.cmp(
                        tg,
                        a0,
                        self.read_reg(a1).cast::<u64>()
                    )),
                    I::CMPS => handler!(self, |OpsRRR(tg, a0, a1)| self.cmp(
                        tg,
                        a0,
                        self.read_reg(a1).cast::<i64>()
                    )),
                    I::DIRU8 => self.dir::<u8>(),
                    I::DIRU16 => self.dir::<u16>(),
                    I::DIRU32 => self.dir::<u32>(),
                    I::DIRU64 => self.dir::<u64>(),
                    I::DIRS8 => self.dir::<i8>(),
                    I::DIRS16 => self.dir::<i16>(),
                    I::DIRS32 => self.dir::<i32>(),
                    I::DIRS64 => self.dir::<i64>(),
                    I::NEG => handler!(self, |OpsRR(tg, a0)| {
                        // Bit negation
                        self.write_reg(tg, self.read_reg(a0).cast::<u64>().wrapping_neg())
                    }),
                    I::NOT => handler!(self, |OpsRR(tg, a0)| {
                        // Logical negation
                        self.write_reg(tg, u64::from(self.read_reg(a0).cast::<u64>() == 0));
                    }),
                    I::SXT8 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i8>() as i64)
                    }),
                    I::SXT16 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i16>() as i64)
                    }),
                    I::SXT32 => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<i32>() as i64)
                    }),
                    I::ADDI8 => self.binary_op_imm(u8::wrapping_add),
                    I::ADDI16 => self.binary_op_imm(u16::wrapping_add),
                    I::ADDI32 => self.binary_op_imm(u32::wrapping_add),
                    I::ADDI64 => self.binary_op_imm(u64::wrapping_add),
                    I::MULI8 => self.binary_op_imm(u8::wrapping_mul),
                    I::MULI16 => self.binary_op_imm(u16::wrapping_mul),
                    I::MULI32 => self.binary_op_imm(u32::wrapping_mul),
                    I::MULI64 => self.binary_op_imm(u64::wrapping_mul),
                    I::ANDI => self.binary_op_imm::<u64>(ops::BitAnd::bitand),
                    I::ORI => self.binary_op_imm::<u64>(ops::BitOr::bitor),
                    I::XORI => self.binary_op_imm::<u64>(ops::BitXor::bitxor),
                    I::SLUI8 => self.binary_op_ims::<u8>(u8::wrapping_shl),
                    I::SLUI16 => self.binary_op_ims::<u16>(u16::wrapping_shl),
                    I::SLUI32 => self.binary_op_ims::<u32>(u32::wrapping_shl),
                    I::SLUI64 => self.binary_op_ims::<u64>(u64::wrapping_shl),
                    I::SRUI8 => self.binary_op_ims::<u8>(u8::wrapping_shr),
                    I::SRUI16 => self.binary_op_ims::<u16>(u16::wrapping_shr),
                    I::SRUI32 => self.binary_op_ims::<u32>(u32::wrapping_shr),
                    I::SRUI64 => self.binary_op_ims::<u64>(u64::wrapping_shr),
                    I::SRSI8 => self.binary_op_ims::<i8>(i8::wrapping_shr),
                    I::SRSI16 => self.binary_op_ims::<i16>(i16::wrapping_shr),
                    I::SRSI32 => self.binary_op_ims::<i32>(i32::wrapping_shr),
                    I::SRSI64 => self.binary_op_ims::<i64>(i64::wrapping_shr),
                    I::CMPUI => handler!(self, |OpsRRD(tg, a0, imm)| { self.cmp(tg, a0, imm) }),
                    I::CMPSI => {
                        handler!(self, |OpsRRD(tg, a0, imm)| { self.cmp(tg, a0, imm as i64) })
                    }
                    I::CP => handler!(self, |OpsRR(tg, a0)| self.write_reg(tg, self.read_reg(a0))),
                    I::SWA => handler!(self, |OpsRR(r0, r1)| {
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
                    I::LI8 => handler!(self, |OpsRB(tg, imm)| self.write_reg(tg, imm)),
                    I::LI16 => handler!(self, |OpsRH(tg, imm)| self.write_reg(tg, imm)),
                    I::LI32 => handler!(self, |OpsRW(tg, imm)| self.write_reg(tg, imm)),
                    I::LI64 => handler!(self, |OpsRD(tg, imm)| self.write_reg(tg, imm)),
                    I::LRA => handler!(self, |OpsRRO(tg, reg, off)| self.write_reg(
                        tg,
                        self.pcrel(off).wrapping_add(self.read_reg(reg).cast::<i64>()).get(),
                    )),
                    // Load. If loading more than register size, continue on adjecent registers
                    I::LD => handler!(self, |OpsRRAH(dst, base, off, count)| self
                        .load(dst, base, off, count)?),
                    // Store. Same rules apply as to LD
                    I::ST => handler!(self, |OpsRRAH(dst, base, off, count)| self
                        .store(dst, base, off, count)?),
                    I::LDR => handler!(self, |OpsRROH(dst, base, off, count)| self.load(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    I::STR => handler!(self, |OpsRROH(dst, base, off, count)| self.store(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    I::BMC => {
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
                    I::BRC => handler!(self, |OpsRRB(src, dst, count)| {
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
                    I::JMP => {
                        let OpsO(off) = self.decode();
                        self.pc = self.pc.wrapping_add(off);
                    }
                    I::JAL => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg + relative offset.
                        let OpsRRO(save, reg, offset) = self.decode();

                        self.write_reg(save, self.pc.next::<OpsRRO>());
                        self.pc = self.pcrel(offset).wrapping_add(self.read_reg(reg).cast::<i64>());
                    }
                    I::JALA => {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg
                        let OpsRRA(save, reg, offset) = self.decode();

                        self.write_reg(save, self.pc.next::<OpsRRA>());
                        self.pc =
                            Address::new(self.read_reg(reg).cast::<u64>().wrapping_add(offset));
                    }
                    // Conditional jumps, jump only to immediates
                    I::JEQ => self.cond_jmp::<u64>(Ordering::Equal),
                    I::JNE => {
                        let OpsRRP(a0, a1, ja) = self.decode();
                        if self.read_reg(a0).cast::<u64>() != self.read_reg(a1).cast::<u64>() {
                            self.pc = self.pcrel(ja);
                        } else {
                            self.bump_pc::<OpsRRP>();
                        }
                    }
                    I::JLTS => self.cond_jmp::<i64>(Ordering::Less),
                    I::JGTS => self.cond_jmp::<i64>(Ordering::Greater),
                    I::JLTU => self.cond_jmp::<u64>(Ordering::Less),
                    I::JGTU => self.cond_jmp::<u64>(Ordering::Greater),
                    I::ECA => {
                        // So we don't get timer interrupt after ECALL
                        if TIMER_QUOTIENT != 0 {
                            self.timer = self.timer.wrapping_add(1);
                        }

                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::Ecall);
                    }
                    I::EBP => {
                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::Breakpoint);
                    }
                    I::FADD32 => self.binary_op::<f32>(ops::Add::add),
                    I::FADD64 => self.binary_op::<f64>(ops::Add::add),
                    I::FSUB32 => self.binary_op::<f32>(ops::Sub::sub),
                    I::FSUB64 => self.binary_op::<f64>(ops::Sub::sub),
                    I::FMUL32 => self.binary_op::<f32>(ops::Mul::mul),
                    I::FMUL64 => self.binary_op::<f64>(ops::Mul::mul),
                    I::FDIV32 => self.binary_op::<f32>(ops::Div::div),
                    I::FDIV64 => self.binary_op::<f64>(ops::Div::div),
                    I::FMA32 => self.fma::<f32>(),
                    I::FMA64 => self.fma::<f64>(),
                    I::FINV32 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, 1. / self.read_reg(reg).cast::<f32>())),
                    I::FINV64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, 1. / self.read_reg(reg).cast::<f64>())),
                    I::FCMPLT32 => self.fcmp::<f32>(Ordering::Less),
                    I::FCMPLT64 => self.fcmp::<f64>(Ordering::Less),
                    I::FCMPGT32 => self.fcmp::<f32>(Ordering::Greater),
                    I::FCMPGT64 => self.fcmp::<f64>(Ordering::Greater),
                    I::ITF32 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<i64>() as f32)),
                    I::ITF64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<i64>() as f64)),
                    I::FTI32 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::f32toint(
                            self.read_reg(reg).cast::<f32>(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    I::FTI64 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::f64toint(
                            self.read_reg(reg).cast::<f64>(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    I::FC32T64 => handler!(self, |OpsRR(tg, reg)| self
                        .write_reg(tg, self.read_reg(reg).cast::<f32>() as f64)),
                    I::FC64T32 => handler!(self, |OpsRRB(tg, reg, mode)| self.write_reg(
                        tg,
                        crate::float::conv64to32(
                            self.read_reg(reg).cast(),
                            RoundingMode::try_from(mode)
                                .map_err(|()| VmRunError::InvalidOperand)?,
                        ),
                    )),
                    I::LRA16 => handler!(self, |OpsRRP(tg, reg, imm)| self.write_reg(
                        tg,
                        (self.pc + self.read_reg(reg).cast::<u64>() + imm + 3_u16).get(),
                    )),
                    I::LDR16 => handler!(self, |OpsRRPH(dst, base, off, count)| self.load(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    I::STR16 => handler!(self, |OpsRRPH(dst, base, off, count)| self.store(
                        dst,
                        base,
                        self.pcrel(off).get(),
                        count
                    )?),
                    I::JMP16 => {
                        let OpsP(off) = self.decode();
                        self.pc = self.pcrel(off);
                    }
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
