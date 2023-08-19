macro_rules! impl_display {
    (for $ty:ty => $(|$selfty:pat_param|)? $fmt:literal $(, $($param:expr),+)? $(,)?) => {
        impl ::core::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                $(let $selfty = self;)?
                write!(f, $fmt, $($param),*)
            }
        }
    };

    (for $ty:ty => $str:literal) => {
        impl ::core::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str($str)
            }
        }
    };

    (for $ty:ty => match {$(
        $bind:pat => $($const:ident)? $fmt:literal $(,$($params:tt)*)?;
    )*}) => {
        impl ::core::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self {
                    $(
                        $bind => $crate::utils::internal::impl_display_match_fragment!($($const,)? f, $fmt $(, $($params)*)?)
                    ),*
                }
            }
        }
    }
}

#[doc(hidden)]
pub(crate) mod internal {
    macro_rules! impl_display_match_fragment {
        (const, $f:expr, $lit:literal) => {
            $f.write_str($lit)
        };

        ($f:expr, $fmt:literal $(, $($params:tt)*)?) => {
            write!($f, $fmt, $($($params)*)?)
        };
    }

    pub(crate) use impl_display_match_fragment;
}

macro_rules! static_assert_eq(($l:expr, $r:expr $(,)?) => {
    const _: [(); ($l != $r) as usize] = [];
});

pub(crate) use {impl_display, static_assert_eq};
