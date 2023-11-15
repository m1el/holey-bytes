use hbvm::mem::{Address, LoadError, Memory, StoreError};

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
        unsafe { core::ptr::copy(source, addr.get() as *mut u8, count) }
        Ok(())
    }

    #[inline]
    unsafe fn prog_read<T: Copy>(&mut self, addr: Address) -> T {
        unsafe { core::ptr::read(addr.get() as *const T) }
    }
}
