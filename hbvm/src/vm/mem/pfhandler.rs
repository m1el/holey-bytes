//! Program trap handling interfaces

use super::{Memory, MemoryAccessReason, PageSize};

/// Handle VM traps
pub trait HandlePageFault {
    /// Handle page fault
    /// 
    /// Return true if handling was sucessful,
    /// otherwise the program will be interrupted and will
    /// yield an error.
    fn page_fault(
        &mut self,
        reason: MemoryAccessReason,
        memory: &mut Memory,
        vaddr: u64,
        size: PageSize,
        dataptr: *mut u8,
    ) -> bool;
}
