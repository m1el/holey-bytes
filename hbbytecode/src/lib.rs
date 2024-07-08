#![no_std]

pub use crate::ops::*;
use core::convert::TryFrom;

pub mod opcode;
mod ops;

type OpR = u8;

type OpA = u64;
type OpO = i32;
type OpP = i16;

type OpB = u8;
type OpH = u16;
type OpW = u32;
type OpD = u64;

/// # Safety
/// Has to be valid to be decoded from bytecode.
pub unsafe trait BytecodeItem {}
unsafe impl BytecodeItem for u8 {}

/// Rounding mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    NearestEven = 0,
    Truncate = 1,
    Up = 2,
    Down = 3,
}

impl TryFrom<u8> for RoundingMode {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        (value <= 3).then(|| unsafe { core::mem::transmute(value) }).ok_or(())
    }
}
