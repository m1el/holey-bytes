pub mod asm;
pub mod text;

macro_rules! impl_both {
    ($($tt:tt)*) => {
        impl Assembler {
            $crate::macros::asm::impl_asm!($($tt)*);
        }

        $crate::macros::text::gen_text!($($tt)*);
    };
}

pub(crate) use impl_both;
