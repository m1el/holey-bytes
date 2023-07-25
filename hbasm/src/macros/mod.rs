//! And here the land of macros begin.
//!
//! They do not bite, really. Have you seen what Yandros is writing?

pub mod asm;
pub mod text;

#[allow(rustdoc::invalid_rust_codeblocks)]
/// Generate code for both programmatic-interface assembler and
/// textural interface.
///
/// Some people claim:
/// > Write programs to handle text streams, because that is a universal interface.
///
/// We at AbleCorp believe that nice programatic API is nicer than piping some text
/// into a program. It's less error-prone and faster.
///
/// # Syntax
/// ```no_run
/// impl_all!(
///     INSTRUCTION_TYPE(p0: TYPE, p1: TYPE, …)
///         => [INSTRUCTION_A, INSTRUCTION_B, …],
///     …
/// );
/// ```
/// - Instruction type determines opcode-generic, instruction-type-specific
///   function. Name: `i_param_INSTRUCTION_TYPE`
/// - Per-instructions there will be generated opcode-specific functions calling the generic ones
/// - Operand types
///     - R: Register (u8)
///     - I: Immediate (implements [`crate::Imm`] trait)
///     - Other types are identity-mapped
///
/// # Text assembler
/// Text assembler generated simply calls methods in the [`crate::Assembler`] type.
/// # Syntax
/// ```text
/// instruction op1, op2, …
/// …
/// ```
/// - Opcode names are lowercase
/// - Registers are prefixed with `r` followed by number
/// - Operands are separated by `,`
/// - Instructions are separated by either line feed or `;` (αυτό δεν είναι ερωτηματικό!)
/// - Labels are defined by their names followed by colon `label:`
/// - Labels are referenced simply by their names
/// - Immediates are numbers, can be negative, floats are not yet supported
macro_rules! impl_all {
    ($($tt:tt)*) => {
        impl Assembler {
            $crate::macros::asm::impl_asm!($($tt)*);
        }

        $crate::macros::text::gen_text!($($tt)*);
    };
}

pub(crate) use impl_all;
