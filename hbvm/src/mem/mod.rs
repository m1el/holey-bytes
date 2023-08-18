//! Memory implementations

pub mod softpaging;

mod addr;

pub use addr::Address;
use {derive_more::Display, hbbytecode::ProgramVal};

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
    unsafe fn prog_read<T: ProgramVal>(&mut self, addr: Address) -> Option<T>;

    /// Read from program memory to exectue
    ///
    /// # Safety
    /// - You have to be really sure that these bytes are there, understand?
    unsafe fn prog_read_unchecked<T: ProgramVal>(&mut self, addr: Address) -> T;
}

/// Unhandled load access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
#[display(fmt = "Load access error at address {_0}")]
pub struct LoadError(pub Address);

/// Unhandled store access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
#[display(fmt = "Store access error at address {_0}")]
pub struct StoreError(pub Address);

/// Reason to access memory
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub enum MemoryAccessReason {
    /// Memory was accessed for load (read)
    Load,
    /// Memory was accessed for store (write)
    Store,
}

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
