//! Program trap handling interfaces

use super::{
    mem::{Memory, MemoryAccessReason, PageSize},
    value::Value,
};

/// Handle VM traps
pub trait HandleTrap {
    /// Handle page fault
    fn page_fault(
        &mut self,
        reason: MemoryAccessReason,
        memory: &mut Memory,
        vaddr: u64,
        size: PageSize,
        dataptr: *mut u8,
    ) -> bool;

    /// Handle invalid opcode exception
    fn invalid_op(
        &mut self,
        regs: &mut [Value; 256],
        pc: &mut usize,
        memory: &mut Memory,
        op: u8,
    ) -> bool
    where
        Self: Sized;

    /// Handle environment calls
    fn ecall(&mut self, regs: &mut [Value; 256], pc: &mut usize, memory: &mut Memory)
    where
        Self: Sized;
}
