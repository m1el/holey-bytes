//! Welcome to the land of The Great Dispatch Loop
//!
//! Have fun

use hbbytecode::OpsN;

use {
    super::{
        bmc::BlockCopier,
        mem::Memory,
        value::{Value, ValueVariant},
        Vm, VmRunError, VmRunOk,
    },
    crate::mem::Address,
    core::{cmp::Ordering, ops},
    hbbytecode::{
        BytecodeItem, OpsO, OpsP, OpsRD, OpsRR, OpsRRAH, OpsRRB, OpsRRD, OpsRRH, OpsRRO, OpsRROH,
        OpsRRP, OpsRRPH, OpsRRR, OpsRRRR, OpsRRW,
    },
};

macro_rules! handler {
    ($self:expr, |$ty:ident ($($ident:pat),* $(,)?)| $expr:expr) => {{
        let $ty($($ident),*) = $self.decode::<$ty>();
        #[allow(clippy::no_effect)] $expr;
        $self.bump_pc::<$ty>();
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
                match self
                    .memory
                    .prog_read::<u8>(self.pc as _)
                    .ok_or(VmRunError::ProgramFetchLoadEx(self.pc as _))?
                {
                    UN => {
                        self.bump_pc::<OpsN>();
                        return Err(VmRunError::Unreachable);
                    }
                    TX => {
                        self.bump_pc::<OpsN>();
                        return Ok(VmRunOk::End);
                    }
                    NOP => handler!(self, |OpsN()| ()),
                    ADD => self.binary_op(u64::wrapping_add),
                    SUB => self.binary_op(u64::wrapping_sub),
                    MUL => self.binary_op(u64::wrapping_mul),
                    AND => self.binary_op::<u64>(ops::BitAnd::bitand),
                    OR => self.binary_op::<u64>(ops::BitOr::bitor),
                    XOR => self.binary_op::<u64>(ops::BitXor::bitxor),
                    SL => self.binary_op(|l, r| u64::wrapping_shl(l, r as u32)),
                    SR => self.binary_op(|l, r| u64::wrapping_shr(l, r as u32)),
                    SRS => self.binary_op(|l: u64, r| i64::wrapping_shl(l as i64, r as u32) as u64),
                    CMP => handler!(self, |OpsRRR(tg, a0, a1)| {
                        // Compare a0 <=> a1
                        // < →  0
                        // > →  1
                        // = →  2

                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<i64>()
                                .cmp(&self.read_reg(a1).cast::<i64>())
                                as i64
                                + 1,
                        );
                    }),
                    CMPU => handler!(self, |OpsRRR(tg, a0, a1)| {
                        // Unsigned comparsion
                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<u64>()
                                .cmp(&self.read_reg(a1).cast::<u64>())
                                as i64
                                + 1,
                        );
                    }),
                    NEG => handler!(self, |OpsRR(tg, a0)| {
                        // Bit negation
                        self.write_reg(tg, !self.read_reg(a0).cast::<u64>())
                    }),
                    NOT => handler!(self, |OpsRR(tg, a0)| {
                        // Logical negation
                        self.write_reg(tg, u64::from(self.read_reg(a0).cast::<u64>() == 0));
                    }),
                    DIR => handler!(self, |OpsRRRR(dt, rt, a0, a1)| {
                        // Fused Division-Remainder
                        let a0 = self.read_reg(a0).cast::<u64>();
                        let a1 = self.read_reg(a1).cast::<u64>();
                        self.write_reg(dt, a0.checked_div(a1).unwrap_or(u64::MAX));
                        self.write_reg(rt, a0.checked_rem(a1).unwrap_or(u64::MAX));
                    }),
                    ADDI => self.binary_op_imm(u64::wrapping_add),
                    MULI => self.binary_op_imm(u64::wrapping_sub),
                    ANDI => self.binary_op_imm::<u64>(ops::BitAnd::bitand),
                    ORI => self.binary_op_imm::<u64>(ops::BitOr::bitor),
                    XORI => self.binary_op_imm::<u64>(ops::BitXor::bitxor),
                    SLI => self.binary_op_ims(u64::wrapping_shl),
                    SRI => self.binary_op_ims(u64::wrapping_shr),
                    SRSI => self.binary_op_ims(i64::wrapping_shr),
                    CMPI => handler!(self, |OpsRRD(tg, a0, imm)| {
                        self.write_reg(
                            tg,
                            self.read_reg(a0)
                                .cast::<i64>()
                                .cmp(&Value::from(imm).cast::<i64>())
                                as i64,
                        );
                    }),
                    CMPUI => handler!(self, |OpsRRD(tg, a0, imm)| {
                        self.write_reg(tg, self.read_reg(a0).cast::<u64>().cmp(&imm) as i64);
                    }),
                    CP => handler!(self, |OpsRR(tg, a0)| {
                        self.write_reg(tg, self.read_reg(a0));
                    }),
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
                    LI => handler!(self, |OpsRD(tg, imm)| {
                        self.write_reg(tg, imm);
                    }),
                    LRA => handler!(self, |OpsRRO(tg, reg, imm)| {
                        self.write_reg(
                            tg,
                            (self.pc + self.read_reg(reg).cast::<u64>() + imm + 3_u16).get(),
                        );
                    }),
                    LD => handler!(self, |OpsRRAH(dst, base, off, count)| {
                        // Load. If loading more than register size, continue on adjecent registers
                        self.load(dst, base, off, count)?;
                    }),
                    ST => handler!(self, |OpsRRAH(dst, base, off, count)| {
                        // Store. Same rules apply as to LD
                        self.store(dst, base, off, count)?;
                    }),
                    LDR => handler!(self, |OpsRROH(dst, base, off, count)| {
                        self.load(
                            dst,
                            base,
                            u64::from(off).wrapping_add((self.pc + 3_u64).get()),
                            count,
                        )?;
                    }),
                    STR => handler!(self, |OpsRROH(dst, base, off, count)| {
                        self.store(
                            dst,
                            base,
                            u64::from(off).wrapping_add((self.pc + 3_u64).get()),
                            count,
                        )?;
                    }),
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
                    JMP => handler!(self, |OpsO(off)| self.pc = self.pc.wrapping_add(off)),
                    JAL => handler!(self, |OpsRRW(save, reg, offset)| {
                        // Jump and link. Save PC after this instruction to
                        // specified register and jump to reg + offset.
                        self.write_reg(save, self.pc.get());
                        self.pc = Address::new(
                            self.read_reg(reg).cast::<u64>().wrapping_add(offset.into()),
                        );
                    }),
                    // Conditional jumps, jump only to immediates
                    JEQ => self.cond_jmp::<u64>(Ordering::Equal),
                    JNE => handler!(self, |OpsRRP(a0, a1, ja)| {
                        if self.read_reg(a0).cast::<u64>() != self.read_reg(a1).cast::<u64>() {
                            self.pc = Address::new(
                                ((self.pc.get() as i64).wrapping_add(ja as i64)) as u64,
                            )
                        }
                    }),
                    JLT => self.cond_jmp::<u64>(Ordering::Less),
                    JGT => self.cond_jmp::<u64>(Ordering::Greater),
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
                    ADDF => self.binary_op::<f64>(ops::Add::add),
                    SUBF => self.binary_op::<f64>(ops::Sub::sub),
                    MULF => self.binary_op::<f64>(ops::Mul::mul),
                    DIRF => handler!(self, |OpsRRRR(dt, rt, a0, a1)| {
                        let a0 = self.read_reg(a0).cast::<f64>();
                        let a1 = self.read_reg(a1).cast::<f64>();
                        self.write_reg(dt, a0 / a1);
                        self.write_reg(rt, a0 % a1);
                    }),
                    FMAF => handler!(self, |OpsRRRR(dt, a0, a1, a2)| {
                        self.write_reg(
                            dt,
                            self.read_reg(a0).cast::<f64>() * self.read_reg(a1).cast::<f64>()
                                + self.read_reg(a2).cast::<f64>(),
                        );
                    }),
                    NEGF => handler!(self, |OpsRR(dt, a0)| {
                        self.write_reg(dt, -self.read_reg(a0).cast::<f64>());
                    }),
                    ITF => handler!(self, |OpsRR(dt, a0)| {
                        self.write_reg(dt, self.read_reg(a0).cast::<i64>() as f64);
                    }),
                    FTI => {
                        let OpsRR(dt, a0) = self.decode();
                        self.write_reg(dt, self.read_reg(a0).cast::<f64>() as i64);
                    }
                    ADDFI => self.binary_op_imm::<f64>(ops::Add::add),
                    MULFI => self.binary_op_imm::<f64>(ops::Mul::mul),
                    LRA16 => handler!(self, |OpsRRP(tg, reg, imm)| {
                        self.write_reg(
                            tg,
                            (self.pc + self.read_reg(reg).cast::<u64>() + imm + 3_u16).get(),
                        );
                    }),
                    LDR16 => handler!(self, |OpsRRPH(dst, base, off, count)| {
                        self.load(dst, base, u64::from(off).wrapping_add(self.pc.get()), count)?;
                    }),
                    STR16 => handler!(self, |OpsRRPH(dst, base, off, count)| {
                        self.store(dst, base, u64::from(off).wrapping_add(self.pc.get()), count)?;
                    }),
                    JMPR16 => handler!(self, |OpsP(off)| self.pc = self.pc.wrapping_add(off)),
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
    fn bump_pc<T: BytecodeItem>(&mut self) {
        self.pc = self.pc.wrapping_add(core::mem::size_of::<T>() + 1);
    }

    /// Decode instruction operands
    #[inline(always)]
    unsafe fn decode<T: BytecodeItem>(&mut self) -> T {
        self.memory.prog_read_unchecked::<T>(self.pc + 1_u64)
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

        self.memory.load(
            self.ldst_addr_uber(dst, base, offset, count, n)?,
            self.registers
                .as_mut_ptr()
                .add(usize::from(dst) + usize::from(n))
                .cast(),
            usize::from(count).wrapping_sub(n.into()),
        )?;

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
        self.memory.store(
            self.ldst_addr_uber(dst, base, offset, count, 0)?,
            self.registers.as_ptr().add(usize::from(dst)).cast(),
            count.into(),
        )?;
        Ok(())
    }

    /// Perform binary operating over two registers
    #[inline(always)]
    unsafe fn binary_op<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let OpsRRR(tg, a0, a1) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(a0).cast::<T>(), self.read_reg(a1).cast::<T>()),
        );
        self.bump_pc::<OpsRRR>();
    }

    /// Perform binary operation over register and immediate
    #[inline(always)]
    unsafe fn binary_op_imm<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let OpsRRD(tg, reg, imm) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(reg).cast::<T>(), Value::from(imm).cast::<T>()),
        );
        self.bump_pc::<OpsRRD>();
    }

    /// Perform binary operation over register and shift immediate
    #[inline(always)]
    unsafe fn binary_op_ims<T: ValueVariant>(&mut self, op: impl Fn(T, u32) -> T) {
        let OpsRRW(tg, reg, imm) = self.decode();
        self.write_reg(tg, op(self.read_reg(reg).cast::<T>(), imm));
        self.bump_pc::<OpsRRW>();
    }

    /// Jump at `PC + #3` if ordering on `#0 <=> #1` is equal to expected
    #[inline(always)]
    unsafe fn cond_jmp<T: ValueVariant + Ord>(&mut self, expected: Ordering) {
        let OpsRRP(a0, a1, ja) = self.decode();
        if self
            .read_reg(a0)
            .cast::<T>()
            .cmp(&self.read_reg(a1).cast::<T>())
            == expected
        {
            self.pc = Address::new(((self.pc.get() as i64).wrapping_add(ja as i64)) as u64);
        }

        self.bump_pc::<OpsRRP>();
    }

    /// Read register
    #[inline(always)]
    fn read_reg(&self, n: u8) -> Value {
        unsafe { *self.registers.get_unchecked(n as usize) }
    }

    /// Write a register.
    /// Writing to register 0 is no-op.
    #[inline(always)]
    fn write_reg(&mut self, n: u8, value: impl Into<Value>) {
        if n != 0 {
            unsafe { *self.registers.get_unchecked_mut(n as usize) = value.into() };
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
