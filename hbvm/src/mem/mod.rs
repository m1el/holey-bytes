//! Memory implementations

pub mod softpaging;

pub(crate) mod addr;

use crate::utils::impl_display;
pub use addr::Address;

/// Load-store memory access
pub trait Memory {
    /// Load data from memory on address
    ///
    /// # Safety
    /// - Shall not overrun the buffer
    unsafe fn load(
        &mut self,
        addr: Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), LoadError>;

    /// Store data to memory on address
    ///
    /// # Safety
    /// - Shall not overrun the buffer
    unsafe fn store(
        &mut self,
        addr: Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), StoreError>;

    /// Read from program memory to execute
    ///
    /// # Safety
    /// - Data read have to be valid
    unsafe fn prog_read<T: Copy + 'static>(&mut self, addr: Address) -> T;
}

/// Unhandled load access trap
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadError(pub Address);
impl_display!(for LoadError =>
    |LoadError(a)| "Load access error at address {a}",
);

/// Unhandled store access trap
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreError(pub Address);
impl_display!(for StoreError =>
    |StoreError(a)| "Load access error at address {a}",
);

/// Reason to access memory
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAccessReason {
    /// Memory was accessed for load (read)
    Load,
    /// Memory was accessed for store (write)
    Store,
}

impl_display!(for MemoryAccessReason => match {
    Self::Load  => const "Load";
    Self::Store => const "Store";
});

impl From<LoadError> for crate::VmRunError {
    fn from(value: LoadError) -> Self {
        Self::LoadAccessEx(value.0)
    }
}

impl From<StoreError> for crate::VmRunError {
    fn from(value: StoreError) -> Self {
        Self::StoreAccessEx(value.0)
    }
}

#[derive(Default)]
pub struct HostMemory;
impl Memory for HostMemory {
    #[inline]
    unsafe fn load(
        &mut self,
        addr: Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), LoadError> {
        unsafe { core::ptr::copy(addr.get() as *const u8, target, count) }
        Ok(())
    }

    #[inline]
    unsafe fn store(
        &mut self,
        addr: Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), StoreError> {
        debug_assert!(addr.get() != 0);
        debug_assert!(!source.is_null());
        unsafe { core::ptr::copy(source, addr.get() as *mut u8, count) }
        Ok(())
    }

    #[inline]
    unsafe fn prog_read<T: Copy>(&mut self, addr: Address) -> T {
        debug_assert!(addr.get() != 0);
        unsafe { core::ptr::read(addr.get() as *const T) }
    }
}
