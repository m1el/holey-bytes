//! HoleyBytes Virtual Machine
//!
//! All unsafe code here should be sound, if input bytecode passes validation.

// # General safety notice:
// - Validation has to assure there is 256 registers (r0 - r255)
// - Instructions have to be valid as specified (values and sizes)
// - Mapped pages should be at least 4 KiB
// - Yes, I am aware of the UB when jumping in-mid of instruction where
//   the read byte corresponds to an instruction whose lenght exceets the
//   program size. If you are (rightfully) worried about the UB, for now just
//   append your program with 11 zeroes.

mod mem;
mod value;

use {
    crate::validate,
    core::ops,
    hbbytecode::{OpParam, ParamBB, ParamBBB, ParamBBBB, ParamBBD, ParamBBDH, ParamBD},
    mem::Memory,
    static_assertions::assert_impl_one,
    value::Value,
};

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

macro_rules! binary_op {
    ($self:expr, $ty:ident, $handler:expr) => {{
        let ParamBBB(tg, a0, a1) = param!($self, ParamBBB);
        $self.write_reg(
            tg,
            $handler(
                Value::$ty(&$self.read_reg(a0)),
                Value::$ty(&$self.read_reg(a1)),
            )
            .into(),
        );
    }};
}

macro_rules! binary_op_imm {
    ($self:expr, $ty:ident, $handler:expr) => {{
        let ParamBBD(tg, a0, imm) = param!($self, ParamBBD);
        $self.write_reg(
            tg,
            $handler(Value::$ty(&$self.read_reg(a0)), Value::$ty(&imm.into())).into(),
        );
    }};
}

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

pub struct Vm<'a> {
    pub registers: [Value; 256],
    pub memory: Memory,
    pc: usize,
    program: &'a [u8],
}

impl<'a> Vm<'a> {
    /// # Safety
    /// Program code has to be validated
    pub unsafe fn new_unchecked(program: &'a [u8]) -> Self {
        Self {
            registers: [Value::from(0_u64); 256],
            memory: Default::default(),
            pc: 0,
            program,
        }
    }

    pub fn new_validated(program: &'a [u8]) -> Result<Self, validate::Error> {
        validate::validate(program)?;
        Ok(unsafe { Self::new_unchecked(program) })
    }

    pub fn run(&mut self) -> HaltReason {
        use hbbytecode::opcode::*;
        loop {
            let Some(&opcode) = self.program.get(self.pc)
                else { return HaltReason::ProgramEnd };

            unsafe {
                match opcode {
                    NOP => param!(self, ()),
                    ADD => binary_op!(self, as_u64, u64::wrapping_add),
                    SUB => binary_op!(self, as_u64, u64::wrapping_sub),
                    MUL => binary_op!(self, as_u64, u64::wrapping_mul),
                    AND => binary_op!(self, as_u64, ops::BitAnd::bitand),
                    OR => binary_op!(self, as_u64, ops::BitOr::bitor),
                    XOR => binary_op!(self, as_u64, ops::BitXor::bitxor),
                    SL => binary_op!(self, as_u64, ops::Shl::shl),
                    SR => binary_op!(self, as_u64, ops::Shr::shr),
                    SRS => binary_op!(self, as_i64, ops::Shr::shr),
                    CMP => {
                        let ParamBBB(tg, a0, a1) = param!(self, ParamBBB);
                        self.write_reg(
                            tg,
                            (self.read_reg(a0).as_i64().cmp(&self.read_reg(a1).as_i64()) as i64)
                                .into(),
                        );
                    }
                    CMPU => {
                        let ParamBBB(tg, a0, a1) = param!(self, ParamBBB);
                        self.write_reg(
                            tg,
                            (self.read_reg(a0).as_u64().cmp(&self.read_reg(a1).as_u64()) as i64)
                                .into(),
                        );
                    }
                    NOT => {
                        let param = param!(self, ParamBB);
                        self.write_reg(param.0, (!self.read_reg(param.1).as_u64()).into());
                    }
                    NEG => {
                        let param = param!(self, ParamBB);
                        self.write_reg(
                            param.0,
                            match self.read_reg(param.1).as_u64() {
                                0 => 1_u64,
                                _ => 0,
                            }
                            .into(),
                        );
                    }
                    DIR => {
                        let ParamBBBB(dt, rt, a0, a1) = param!(self, ParamBBBB);
                        let a0 = self.read_reg(a0).as_u64();
                        let a1 = self.read_reg(a1).as_u64();
                        self.write_reg(dt, (a0.checked_div(a1).unwrap_or(u64::MAX)).into());
                        self.write_reg(rt, (a0.checked_rem(a1).unwrap_or(u64::MAX)).into());
                    }
                    ADDI => binary_op_imm!(self, as_u64, ops::Add::add),
                    MULI => binary_op_imm!(self, as_u64, ops::Mul::mul),
                    ANDI => binary_op_imm!(self, as_u64, ops::BitAnd::bitand),
                    ORI => binary_op_imm!(self, as_u64, ops::BitOr::bitor),
                    XORI => binary_op_imm!(self, as_u64, ops::BitXor::bitxor),
                    SLI => binary_op_imm!(self, as_u64, ops::Shl::shl),
                    SRI => binary_op_imm!(self, as_u64, ops::Shr::shr),
                    SRSI => binary_op_imm!(self, as_i64, ops::Shr::shr),
                    CMPI => {
                        let ParamBBD(tg, a0, imm) = param!(self, ParamBBD);
                        self.write_reg(
                            tg,
                            (self.read_reg(a0).as_i64().cmp(&Value::from(imm).as_i64()) as i64)
                                .into(),
                        );
                    }
                    CMPUI => {
                        let ParamBBD(tg, a0, imm) = param!(self, ParamBBD);
                        self.write_reg(tg, (self.read_reg(a0).as_u64().cmp(&imm) as i64).into());
                    }
                    CP => {
                        let param = param!(self, ParamBB);
                        self.write_reg(param.0, self.read_reg(param.1));
                    }
                    SWA => {
                        let ParamBB(src, dst) = param!(self, ParamBB);
                        if src + dst != 0 {
                            core::ptr::swap(
                                self.registers.get_unchecked_mut(usize::from(src)),
                                self.registers.get_unchecked_mut(usize::from(dst)),
                            );
                        }
                    }
                    LI => {
                        let param = param!(self, ParamBD);
                        self.write_reg(param.0, param.1.into());
                    }
                    LD => {
                        let ParamBBDH(dst, base, off, count) = param!(self, ParamBBDH);
                        let n: usize = match dst {
                            0 => 1,
                            _ => 0,
                        };

                        if self
                            .memory
                            .load(
                                self.read_reg(base).as_u64() + off + n as u64,
                                self.registers.as_mut_ptr().add(usize::from(dst) + n).cast(),
                                usize::from(count).saturating_sub(n),
                            )
                            .is_err()
                        {
                            return HaltReason::LoadAccessEx;
                        }
                    }
                    ST => {
                        let ParamBBDH(dst, base, off, count) = param!(self, ParamBBDH);
                        if self
                            .memory
                            .store(
                                self.read_reg(base).as_u64() + off,
                                self.registers.as_ptr().add(usize::from(dst)).cast(),
                                count.into(),
                            )
                            .is_err()
                        {
                            return HaltReason::LoadAccessEx;
                        }
                    }
                    BMC => {
                        let ParamBBD(src, dst, count) = param!(self, ParamBBD);
                        if self
                            .memory
                            .block_copy(
                                self.read_reg(src).as_u64(),
                                self.read_reg(dst).as_u64(),
                                count,
                            )
                            .is_err()
                        {
                            return HaltReason::LoadAccessEx;
                        }
                    }
                    BRC => {
                        let ParamBBB(src, dst, count) = param!(self, ParamBBB);
                        core::ptr::copy(
                            self.registers.get_unchecked(usize::from(src)),
                            self.registers.get_unchecked_mut(usize::from(dst)),
                            usize::from(count * 8),
                        );
                    }
                    JMP => {
                        let ParamBD(reg, offset) = param!(self, ParamBD);
                        self.pc = (self.read_reg(reg).as_u64() + offset) as usize;
                    }
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
                        return HaltReason::Ecall;
                    }
                    ADDF => binary_op!(self, as_f64, ops::Add::add),
                    MULF => binary_op!(self, as_f64, ops::Mul::mul),
                    DIRF => {
                        let ParamBBBB(dt, rt, a0, a1) = param!(self, ParamBBBB);
                        let a0 = self.read_reg(a0).as_f64();
                        let a1 = self.read_reg(a1).as_f64();
                        self.write_reg(dt, (a0 / a1).into());
                        self.write_reg(rt, (a0 % a1).into());
                    }
                    ADDFI => binary_op_imm!(self, as_f64, ops::Add::add),
                    MULFI => binary_op_imm!(self, as_f64, ops::Mul::mul),
                    _ => return HaltReason::InvalidOpcode,
                }
            }
        }
    }

    #[inline]
    unsafe fn read_reg(&self, n: u8) -> Value {
        if n == 0 {
            0_u64.into()
        } else {
            *self.registers.get_unchecked(n as usize)
        }
    }

    #[inline]
    unsafe fn write_reg(&mut self, n: u8, value: Value) {
        if n != 0 {
            *self.registers.get_unchecked_mut(n as usize) = value;
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HaltReason {
    ProgramEnd,
    Ecall,
    InvalidOpcode,
    LoadAccessEx,
    StoreAccessEx,
}
