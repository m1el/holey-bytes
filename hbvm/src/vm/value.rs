//! HoleyBytes register value definition

use core::fmt::Debug;

/// Define [`Value`] union
///
/// # Safety
/// Union variants have to be sound to byte-reinterpretate
/// between each other. Otherwise the behaviour is undefined.
macro_rules! value_def {
    ($($ty:ident),* $(,)?) => {
        /// HBVM register value
        #[derive(Copy, Clone)]
        #[repr(packed)]
        pub union Value {
            $(pub $ty: $ty),*
        }

        paste::paste! {
            impl Value {$(
                #[doc = "Byte-reinterpret [`Value`] as [`" $ty "`]"]
                #[inline]
                pub fn [<as_ $ty>](&self) -> $ty {
                    unsafe { self.$ty }
                }
            )*}
        }

        $(
            impl From<$ty> for Value {
                #[inline]
                fn from(value: $ty) -> Self {
                    Self { $ty: value }
                }
            }
        )*
    };
}

value_def!(u64, i64, f64);
static_assertions::const_assert_eq!(core::mem::size_of::<Value>(), 8);

impl Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Print formatted as hexadecimal, unsigned integer
        write!(f, "{:x}", self.as_u64())
    }
}
