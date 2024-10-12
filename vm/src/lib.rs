//! HoleyBytes Virtual Machine
//!
//! # Alloc feature
//! - Enabled by default
//! - Provides mapping / unmapping, as well as [`Default`] and [`Drop`]
//!   implementations for soft-paged memory implementation

// # General safety notice:
// - Validation has to assure there is 256 registers (r0 - r255)
// - Instructions have to be valid as specified (values and sizes)
// - Mapped pages should be at least 4 KiB

#![no_std]
#![cfg_attr(feature = "nightly", feature(fn_align))]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod mem;
pub mod value;

mod bmc;
mod float;
mod utils;
mod vmrun;

pub use float::FL_ARCH_SPECIFIC_SUPPORTED;
use {
    bmc::BlockCopier,
    mem::{Address, Memory},
    value::{Value, ValueVariant},
};

/// HoleyBytes Virtual Machine
pub struct Vm<Mem, const TIMER_QUOTIENT: usize> {
    /// Holds 256 registers
    ///
    /// Writing to register 0 is considered idk behaviour
    /// in terms of HoleyBytes program execution
    pub registers: [Value; 256],

    /// Memory implementation
    pub memory: Mem,

    /// Program counter
    pub pc: Address,

    /// Program timer
    timer: usize,

    /// Saved block copier
    copier: Option<BlockCopier>,
}

impl<Mem: Default, const TIMER_QUOTIENT: usize> Default for Vm<Mem, TIMER_QUOTIENT> {
    fn default() -> Self {
        Self {
            registers: [Value::from(0_u64); 256],
            memory: Mem::default(),
            pc: Address::default(),
            timer: 0,
            copier: None,
        }
    }
}

impl<Mem, const TIMER_QUOTIENT: usize> Vm<Mem, TIMER_QUOTIENT>
where
    Mem: Memory,
{
    /// Create a new VM with program and trap handler
    ///
    /// # Safety
    /// Program code has to be validated
    pub unsafe fn new(memory: Mem, entry: Address) -> Self {
        Self { registers: [Value::from(0_u64); 256], memory, pc: entry, timer: 0, copier: None }
    }

    /// Read register
    #[inline(always)]
    pub fn read_reg(&self, n: u8) -> Value {
        unsafe { *self.registers.get_unchecked(n as usize) }
    }

    /// Write a register.
    /// Writing to register 0 is no-op.
    #[inline(always)]
    pub fn write_reg<T: ValueVariant>(&mut self, n: u8, value: T) {
        if n != 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    (&value as *const T).cast::<u8>(),
                    self.registers.as_mut_ptr().add(n.into()).cast::<u8>(),
                    core::mem::size_of::<T>(),
                );
            };
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
    LoadAccessEx(Address),

    /// Unhandled instruction load access exception
    ProgramFetchLoadEx(Address),

    /// Unhandled store access exception
    StoreAccessEx(Address),

    /// Register out-of-bounds access
    RegOutOfBounds,

    /// Address out-of-bounds
    AddrOutOfBounds,

    /// Reached unreachable code
    Unreachable,

    /// Invalid operand
    InvalidOperand,
}

impl core::fmt::Display for VmRunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VmRunError::InvalidOpcode(op) => {
                f.write_str("invalid op code: ").and_then(|_| op.fmt(f))
            }
            VmRunError::LoadAccessEx(address) => {
                f.write_str("falied to load at ").and_then(|_| address.fmt(f))
            }
            VmRunError::ProgramFetchLoadEx(address) => {
                f.write_str("falied to load instruction at ").and_then(|_| address.fmt(f))
            }
            VmRunError::StoreAccessEx(address) => {
                f.write_str("falied to store at ").and_then(|_| address.fmt(f))
            }
            VmRunError::RegOutOfBounds => f.write_str("reg out of bounds"),
            VmRunError::AddrOutOfBounds => f.write_str("addr out-of-bounds"),
            VmRunError::Unreachable => f.write_str("unreachable"),
            VmRunError::InvalidOperand => f.write_str("invalud operand"),
        }
    }
}

impl core::error::Error for VmRunError {}

/// Virtual machine halt ok
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VmRunOk {
    /// Program has eached its end
    End,

    /// Program was interrupted by a timer
    Timer,

    /// Environment call
    Ecall,

    /// Breakpoint
    Breakpoint,
}
