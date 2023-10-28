#![no_std]

use core::convert::TryFrom;

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
macro_rules! define_items {
    ($($name:ident ($($item:ident),* $(,)?)),* $(,)?) => {
        $(
            #[derive(Clone, Copy, Debug)]
            #[repr(packed)]
            pub struct $name($(pub $item),*);
            unsafe impl BytecodeItem for $name {}
        )*
    };
}

define_items! {
    OpsRR   (OpR, OpR          ),
    OpsRRR  (OpR, OpR, OpR     ),
    OpsRRRR (OpR, OpR, OpR, OpR),
    OpsRRB  (OpR, OpR, OpB     ),
    OpsRRH  (OpR, OpR, OpH     ),
    OpsRRW  (OpR, OpR, OpW     ),
    OpsRRD  (OpR, OpR, OpD     ),
    OpsRB   (OpR, OpB          ),
    OpsRH   (OpR, OpH          ),
    OpsRW   (OpR, OpW          ),
    OpsRD   (OpR, OpD          ),
    OpsRRA  (OpR, OpR, OpA     ),
    OpsRRAH (OpR, OpR, OpA, OpH),
    OpsRROH (OpR, OpR, OpO, OpH),
    OpsRRPH (OpR, OpR, OpP, OpH),
    OpsRRO  (OpR, OpR, OpO     ),
    OpsRRP  (OpR, OpR, OpP     ),
    OpsO    (OpO,              ),
    OpsP    (OpP,              ),
    OpsN    (                  ),
}

unsafe impl BytecodeItem for u8 {}

::with_builtin_macros::with_builtin! {
    let $spec = include_from_root!("instructions.in") in {
        /// Invoke macro with bytecode definition
        ///
        /// # Format
        /// ```text
        /// Opcode, Mnemonic, Type, Docstring;
        /// ```
        ///
        /// # Type
        /// ```text
        /// Types consist of letters meaning a single field
        /// | Type | Size (B) | Meaning                 |
        /// |:-----|:---------|:------------------------|
        /// | N    | 0        | Empty                   |
        /// | R    | 1        | Register                |
        /// | A    | 8        | Absolute address        |
        /// | O    | 4        | Relative address offset |
        /// | P    | 2        | Relative address offset |
        /// | B    | 1        | Immediate               |
        /// | H    | 2        | Immediate               |
        /// | W    | 4        | Immediate               |
        /// | D    | 8        | Immediate               |
        /// ```
        #[macro_export]
            macro_rules! invoke_with_def {
                ($macro:path) => {
                    $macro! { $spec }
                };
            }
    }
}

macro_rules! gen_opcodes {
    ($($opcode:expr, $mnemonic:ident, $_ty:ident, $doc:literal;)*) => {
        pub mod opcode {
            $(
                #[doc = $doc]
                pub const $mnemonic: u8 = $opcode;
            )*
        }
    };
}

/// Rounding mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundingMode {
    NearestEven = 0,
    Truncate = 1,
    Up       = 2,
    Down     = 3,
}

impl TryFrom<u8> for RoundingMode {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        (value >= 3)
            .then(|| unsafe { core::mem::transmute(value) })
            .ok_or(())
    }
}

invoke_with_def!(gen_opcodes);
