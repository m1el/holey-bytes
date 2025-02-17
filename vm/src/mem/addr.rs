//! Virtual(?) memory address

use {
    crate::utils::impl_display,
    core::{fmt::Debug, ops},
};

/// Memory address
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address(u64);
impl Address {
    /// A null address
    pub const NULL: Self = Self(0);

    /// Saturating integer addition. Computes self + rhs, saturating at the numeric bounds instead of overflowing.
    #[inline]
    pub fn saturating_add<T: AddressOp>(self, rhs: T) -> Self {
        Self(self.0.saturating_add(rhs.cast_u64()))
    }

    /// Saturating integer subtraction. Computes self - rhs, saturating at the numeric bounds instead of overflowing.
    #[inline]
    pub fn saturating_sub<T: AddressOp>(self, rhs: T) -> Self {
        Self(self.0.saturating_sub(rhs.cast_u64()))
    }

    /// Wrapping integer addition. Computes self + rhs, wrapping the numeric bounds.
    #[inline]
    pub fn wrapping_add<T: AddressOp>(self, rhs: T) -> Self {
        Self(self.0.wrapping_add(rhs.cast_u64()))
    }

    /// Wrapping integer subtraction. Computes self + rhs, wrapping the numeric bounds.
    #[inline]
    pub fn wrapping_sub<T: AddressOp>(self, rhs: T) -> Self {
        Self(self.0.wrapping_sub(rhs.cast_u64()))
    }

    /// Cast or if smaller, truncate to [`usize`]
    pub fn truncate_usize(self) -> usize {
        self.0 as _
    }

    /// Get inner value
    #[inline(always)]
    pub fn get(self) -> u64 {
        self.0
    }

    /// Get ptr to the next instruction
    #[inline(always)]
    pub fn next<A>(self) -> u64 {
        self.0.wrapping_add(core::mem::size_of::<A>() as u64 + 1)
    }

    /// Construct new address
    #[inline(always)]
    pub fn new(val: u64) -> Self {
        Self(val)
    }

    /// Do something with inner value
    #[inline(always)]
    pub fn map(self, f: impl Fn(u64) -> u64) -> Self {
        Self(f(self.0))
    }
}

impl_display!(for Address =>
    |Address(a)| "{a:0x}"
);

impl<T: AddressOp> ops::Add<T> for Address {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self(self.0.wrapping_add(rhs.cast_u64()))
    }
}

impl<T: AddressOp> ops::Sub<T> for Address {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self(self.0.wrapping_sub(rhs.cast_u64()))
    }
}

impl<T: AddressOp> ops::AddAssign<T> for Address {
    fn add_assign(&mut self, rhs: T) {
        self.0 = self.0.wrapping_add(rhs.cast_u64())
    }
}

impl<T: AddressOp> ops::SubAssign<T> for Address {
    fn sub_assign(&mut self, rhs: T) {
        self.0 = self.0.wrapping_sub(rhs.cast_u64())
    }
}

impl From<Address> for u64 {
    #[inline(always)]
    fn from(value: Address) -> Self {
        value.0
    }
}

impl Debug for Address {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{:0x}]", self.0)
    }
}

/// Can perform address operations with
pub trait AddressOp {
    /// Cast to u64, truncating or extending
    fn cast_u64(self) -> u64;
}

macro_rules! impl_address_ops_u(($($ty:ty),* $(,)?) => {
    $(impl AddressOp for $ty {
        #[inline(always)]
        fn cast_u64(self) -> u64 { self as _ }
    })*
});

macro_rules! impl_address_ops_i(($($ty:ty),* $(,)?) => {
    $(impl AddressOp for $ty {
        #[inline(always)]
        fn cast_u64(self) -> u64 { self as i64 as u64 }
    })*
});

impl_address_ops_u!(u8, u16, u32, u64, usize);
impl_address_ops_i!(i8, i16, i32, i64, isize);
