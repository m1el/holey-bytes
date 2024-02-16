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
    ($($name:ident ($($nm:ident: $item:ident),* $(,)?)),* $(,)?) => {
        $(
            #[derive(Clone, Copy, Debug)]
            #[repr(packed)]
            pub struct $name($(pub $item),*);
            unsafe impl BytecodeItem for $name {}

            impl Encodable for $name {
                fn encode(self, _buffer: &mut impl Buffer) {
                    let Self($($nm),*) = self;
                    $(
                        for byte in $nm.to_le_bytes() {
                            unsafe { _buffer.write(byte) };
                        }
                    )*
                }

                fn encode_len(self) -> usize {
                    core::mem::size_of::<Self>()
                }
            }
        )*
    };
}

define_items! {
    OpsRR   (a: OpR, b: OpR                ),
    OpsRRR  (a: OpR, b: OpR, c: OpR        ),
    OpsRRRR (a: OpR, b: OpR, c: OpR, d: OpR),
    OpsRRB  (a: OpR, b: OpR, c: OpB        ),
    OpsRRH  (a: OpR, b: OpR, c: OpH        ),
    OpsRRW  (a: OpR, b: OpR, c: OpW        ),
    OpsRRD  (a: OpR, b: OpR, c: OpD        ),
    OpsRB   (a: OpR, b: OpB                ),
    OpsRH   (a: OpR, b: OpH                ),
    OpsRW   (a: OpR, b: OpW                ),
    OpsRD   (a: OpR, b: OpD                ),
    OpsRRA  (a: OpR, b: OpR, c: OpA        ),
    OpsRRAH (a: OpR, b: OpR, c: OpA, d: OpH),
    OpsRROH (a: OpR, b: OpR, c: OpO, d: OpH),
    OpsRRPH (a: OpR, b: OpR, c: OpP, d: OpH),
    OpsRRO  (a: OpR, b: OpR, c: OpO        ),
    OpsRRP  (a: OpR, b: OpR, c: OpP        ),
    OpsO    (a: OpO,                       ),
    OpsP    (a: OpP,                       ),
    OpsN    (                              ),
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
            ($($macro:tt)*) => {
                $($macro)*! { $spec }
            };
        }
    }
}

pub trait Buffer {
    fn reserve(&mut self, bytes: usize);
    /// # Safety
    /// Reserve needs to be called before this function, and only reserved amount can be written.
    unsafe fn write(&mut self, byte: u8);
}

pub trait Encodable {
    fn encode(self, buffer: &mut impl Buffer);
    fn encode_len(self) -> usize;
}

macro_rules! gen_opcodes {
    ($($opcode:expr, $mnemonic:ident, $ty:ident, $doc:literal;)*) => {
        pub mod opcode {
            $(
                #[doc = $doc]
                pub const $mnemonic: u8 = $opcode;
            )*

            paste::paste! {
                #[derive(Clone, Copy, Debug)]
                #[repr(u8)]
                pub enum Op { $(
                    [< $mnemonic:lower:camel >](super::[<Ops $ty>]),
                )* }

                impl crate::Encodable for Op {
                    fn encode(self, buffer: &mut impl crate::Buffer) {
                        match self {
                            $(
                                Self::[< $mnemonic:lower:camel >](op) => {
                                    unsafe { buffer.write($opcode) };
                                    op.encode(buffer);
                                }
                            )*
                        }
                    }

                    fn encode_len(self) -> usize {
                        match self {
                            $(
                                Self::[< $mnemonic:lower:camel >](op) => {
                                    1 + crate::Encodable::encode_len(op)
                                }
                            )*
                        }
                    }
                }
            }
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
        (value <= 3)
            .then(|| unsafe { core::mem::transmute(value) })
            .ok_or(())
    }
}

invoke_with_def!(gen_opcodes);
