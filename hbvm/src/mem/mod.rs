//! Memory implementations

pub mod softpaging;

pub(crate) mod addr;

pub use addr::Address;

use {crate::utils::impl_display, hbbytecode::BytecodeItem};

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
    unsafe fn prog_read<T: BytecodeItem>(&mut self, addr: Address) -> Option<T>;

    /// Read from program memory to exectue
    ///
    /// # Safety
    /// - You have to be really sure that these bytes are there, understand?
    unsafe fn prog_read_unchecked<T: BytecodeItem>(&mut self, addr: Address) -> T;
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
