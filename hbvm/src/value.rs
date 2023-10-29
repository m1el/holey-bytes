//! HoleyBytes register value definition

use crate::utils::static_assert;

/// Define [`Value`] »union« (it's fake)
///
/// # Safety
/// Its variants have to be sound to byte-reinterpretate
/// between each other. Otherwise the behaviour is undefined.
macro_rules! value_def {
    ($($ty:ident),* $(,)?) => {
        /// HBVM register value
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct Value(pub u64);

        $(
            impl From<$ty> for Value {
                #[inline]
                fn from(value: $ty) -> Self {
                    let mut new = core::mem::MaybeUninit::<u64>::zeroed();
                    unsafe {
                        new.as_mut_ptr().cast::<$ty>().write(value);
                        Self(new.assume_init())
                    }
                }
            }

            static_assert!(core::mem::size_of::<$ty>() <= core::mem::size_of::<Value>());

            impl private::Sealed for $ty {}
            unsafe impl ValueVariant for $ty {}
        )*
    };
}

impl Value {
    /// Byte reinterpret value to target variant
    #[inline]
    pub fn cast<V: ValueVariant>(self) -> V {
        unsafe { core::mem::transmute_copy(&self.0) }
    }
}

/// # Safety
/// - N/A, not to be implemented manually
pub unsafe trait ValueVariant: private::Sealed + Copy + Into<Value> {}
impl private::Sealed for Value {}
unsafe impl ValueVariant for Value {}

mod private {
    pub trait Sealed {}
}

value_def!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
static_assert!(core::mem::size_of::<Value>() == 8);

impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Print formatted as hexadecimal, unsigned integer
        write!(f, "{:x}", self.cast::<u64>())
    }
}

pub(crate) trait CheckedDivRem {
    fn checked_div(self, other: Self) -> Option<Self>
    where
        Self: Sized;
    fn checked_rem(self, other: Self) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! impl_checked_div_rem {
    ($($ty:ty),* $(,)?) => {
        $(impl CheckedDivRem for $ty {
            #[inline(always)]
            fn checked_div(self, another: Self) -> Option<Self>
                { self.checked_div(another) }

            #[inline(always)]
            fn checked_rem(self, another: Self) -> Option<Self>
                { self.checked_rem(another) }
        })*
    };
}

impl_checked_div_rem!(u8, u16, u32, u64, i8, i16, i32, i64);
