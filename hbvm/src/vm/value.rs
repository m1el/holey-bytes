use core::fmt::Debug;

macro_rules! value_def {
    ($($ty:ident),* $(,)?) => {
        #[derive(Copy, Clone)]
        #[repr(packed)]
        pub union Value {
            $(pub $ty: $ty),*
        }

        paste::paste! {
            impl Value {$(
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

impl Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_u64().fmt(f)
    }
}
