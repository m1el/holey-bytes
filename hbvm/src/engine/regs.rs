use core::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, Copy)]
pub struct Registers([Value; 60]);

impl Index<u8> for Registers {
    type Output = Value;

    #[inline]
    fn index(&self, index: u8) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<u8> for Registers {
    #[inline]
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl Default for Registers {
    fn default() -> Self {
        Self([Value { u: 0 }; 60])
    }
}

/// # Safety
/// The macro invoker shall make sure that byte reinterpret-cast
/// won't cause undefined behaviour.
macro_rules! value_def {
    ($($fname:ident : $fty:ident, $getter:ident);* $(;)?) => {
        #[derive(Clone, Copy)]
        pub union Value {
            $($fname: $fty),*
        }

        impl Value {$(
            #[inline]
            pub fn $getter(&self) -> $fty {
                unsafe { self.$fname }
            }
        )*}

        $(impl From<$fty> for Value {
            #[inline]
            fn from($fname: $fty) -> Self {
                Self { $fname }
            }
        })*
    }
}

value_def! {
    u: u64, unsigned;
    s: i64, signed;
    f: f64, float;
}

impl Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.unsigned().fmt(f)
    }
}
