//! Welcome to the land of The Great Dispatch Loop
//!
//! Have fun

use crate::mem::Address;

use {
    super::{
        bmc::BlockCopier,
        mem::Memory,
        value::{Value, ValueVariant},
        Vm, VmRunError, VmRunOk,
    },
    core::{cmp::Ordering, mem::size_of, ops},
    hbbytecode::{
        ParamBB, ParamBBB, ParamBBBB, ParamBBD, ParamBBDH, ParamBBW, ParamBD, ProgramVal,
    },
};

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
                        self.decode::<()>();
                        return Err(VmRunError::Unreachable);
                    }
                    TX => {
                        self.decode::<()>();
                        return Ok(VmRunOk::End);
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
                        let n: u8 = match dst {
                            0 => 1,
                            _ => 0,
                        };

                        self.memory.load(
                            self.ldst_addr_uber(dst, base, off, count, n)?,
                            self.registers
                                .as_mut_ptr()
                                .add(usize::from(dst) + usize::from(n))
                                .cast(),
                            usize::from(count).saturating_sub(n.into()),
                        )?;
                    }
                    ST => {
                        // Store. Same rules apply as to LD
                        let ParamBBDH(dst, base, off, count) = self.decode();
                        self.memory.store(
                            self.ldst_addr_uber(dst, base, off, count, 0)?,
                            self.registers.as_ptr().add(usize::from(dst)).cast(),
                            count.into(),
                        )?;
                    }
                    BMC => {
                        // Block memory copy
                        match if let Some(copier) = &mut self.copier {
                            // There is some copier, poll.
                            copier.poll(&mut self.memory)
                        } else {
                            // There is none, make one!
                            let ParamBBD(src, dst, count) = self.decode();

                            // So we are still on BMC on next cycle
                            self.pc -= size_of::<ParamBBD>() + 1;

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
                                self.pc += size_of::<ParamBBD>() + 1;
                            }
                            // Error, shift program counter (for consistency)
                            // and yield error
                            core::task::Poll::Ready(Err(e)) => {
                                self.pc += size_of::<ParamBBD>() + 1;
                                return Err(e.into());
                            }
                            // Not done yet, proceed to next cycle
                            core::task::Poll::Pending => (),
                        }
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
                        self.write_reg(save, self.pc.get());
                        self.pc =
                            Address::new(self.read_reg(reg).cast::<u64>().saturating_add(offset));
                    }
                    // Conditional jumps, jump only to immediates
                    JEQ => self.cond_jmp::<u64>(Ordering::Equal),
                    JNE => {
                        let ParamBBD(a0, a1, jt) = self.decode();
                        if self.read_reg(a0).cast::<u64>() != self.read_reg(a1).cast::<u64>() {
                            self.pc = Address::new(jt);
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
    #[inline(always)]
    unsafe fn decode<T: ProgramVal>(&mut self) -> T {
        let pc1 = self.pc + 1_u64;
        let data = self.memory.prog_read_unchecked::<T>(pc1 as _);
        self.pc += 1 + size_of::<T>();
        data
    }

    /// Perform binary operating over two registers
    #[inline(always)]
    unsafe fn binary_op<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let ParamBBB(tg, a0, a1) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(a0).cast::<T>(), self.read_reg(a1).cast::<T>()),
        );
    }

    /// Perform binary operation over register and immediate
    #[inline(always)]
    unsafe fn binary_op_imm<T: ValueVariant>(&mut self, op: impl Fn(T, T) -> T) {
        let ParamBBD(tg, reg, imm) = self.decode();
        self.write_reg(
            tg,
            op(self.read_reg(reg).cast::<T>(), Value::from(imm).cast::<T>()),
        );
    }

    /// Perform binary operation over register and shift immediate
    #[inline(always)]
    unsafe fn binary_op_ims<T: ValueVariant>(&mut self, op: impl Fn(T, u32) -> T) {
        let ParamBBW(tg, reg, imm) = self.decode();
        self.write_reg(tg, op(self.read_reg(reg).cast::<T>(), imm));
    }

    /// Jump at `#3` if ordering on `#0 <=> #1` is equal to expected
    #[inline(always)]
    unsafe fn cond_jmp<T: ValueVariant + Ord>(&mut self, expected: Ordering) {
        let ParamBBD(a0, a1, ja) = self.decode();
        if self
            .read_reg(a0)
            .cast::<T>()
            .cmp(&self.read_reg(a1).cast::<T>())
            == expected
        {
            self.pc = Address::new(ja);
        }
    }

    /// Read register
    #[inline(always)]
    unsafe fn read_reg(&self, n: u8) -> Value {
        *self.registers.get_unchecked(n as usize)
    }

    /// Write a register.
    /// Writing to register 0 is no-op.
    #[inline(always)]
    unsafe fn write_reg(&mut self, n: u8, value: impl Into<Value>) {
        if n != 0 {
            *self.registers.get_unchecked_mut(n as usize) = value.into();
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
