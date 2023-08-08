//! Program memory implementation

pub mod bmc;
pub mod softpaged;

use {super::VmRunError, derive_more::Display};

pub trait Memory {
    /// Load data from memory on address
    ///
    /// # Safety
    /// - Shall not overrun the buffer
    unsafe fn load(&mut self, addr: u64, target: *mut u8, count: usize) -> Result<(), LoadError>;

    /// Store data to memory on address
    ///
    /// # Safety
    /// - Shall not overrun the buffer
    unsafe fn store(
        &mut self,
        addr: u64,
        source: *const u8,
        count: usize,
    ) -> Result<(), StoreError>;
}

/// Unhandled load access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
#[display(fmt = "Load access error at address {_0:#x}")]
pub struct LoadError(pub u64);

/// Unhandled store access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
#[display(fmt = "Store access error at address {_0:#x}")]
pub struct StoreError(pub u64);

/// Reason to access memory
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub enum MemoryAccessReason {
    Load,
    Store,
}

impl From<LoadError> for VmRunError {
    fn from(value: LoadError) -> Self {
        Self::LoadAccessEx(value.0)
    }
}

impl From<StoreError> for VmRunError {
    fn from(value: StoreError) -> Self {
        Self::StoreAccessEx(value.0)
    }
}
