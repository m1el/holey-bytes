//! HoleyBytes register value definition

use sealed::sealed;

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


        $(
            impl From<$ty> for Value {
                #[inline]
                fn from(value: $ty) -> Self {
                    Self { $ty: value }
                }
            }

            static_assertions::const_assert_eq!(
                core::mem::size_of::<$ty>(),
                core::mem::size_of::<Value>(),
            );

            #[sealed]
            unsafe impl ValueVariant for $ty {}
        )*
    };
}

impl Value {
    #[inline]
    pub fn cast<Variant: ValueVariant>(self) -> Variant {
        union Transmute<Variant: ValueVariant> {
            src:     Value,
            variant: Variant,
        }

        unsafe { Transmute { src: self }.variant }
    }
}

/// # Safety
/// - N/A, not to be implemented manually
#[sealed]
pub unsafe trait ValueVariant: Copy + Into<Value> {}

value_def!(u64, i64, f64);
static_assertions::const_assert_eq!(core::mem::size_of::<Value>(), 8);

impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Print formatted as hexadecimal, unsigned integer
        write!(f, "{:x}", self.cast::<u64>())
    }
}
