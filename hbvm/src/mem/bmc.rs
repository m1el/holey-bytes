use {
    super::MemoryAccessReason,
    crate::{
        mem::{perm_check, HandlePageFault, Memory},
        VmRunError,
    },
    core::{mem::MaybeUninit, task::Poll},
};

// Buffer size (defaults to 4 KiB, a smallest page size on most platforms)
const BUF_SIZE: usize = 4096;

// This should be equal to `BUF_SIZE`
#[repr(align(4096))]
struct AlignedBuf([MaybeUninit<u8>; BUF_SIZE]);

pub struct BlockCopier {
    /// Source address
    src:       u64,
    /// Destination address
    dst:       u64,
    /// How many buffer sizes to copy?
    n_buffers: usize,
    /// …and what remainds after?
    rem:       usize,
}

impl BlockCopier {
    pub fn new(src: u64, dst: u64, count: usize) -> Self {
        Self {
            src,
            dst,
            n_buffers: count / BUF_SIZE,
            rem: count % BUF_SIZE,
        }
    }

    /// Copy one block
    /// 
    /// # Safety
    /// - Same as for [`Memory::load`] and [`Memory::store`]
    pub unsafe fn poll(
        &mut self,
        memory: &mut Memory,
        traph: &mut impl HandlePageFault,
    ) -> Poll<Result<(), BlkCopyError>> {
        // Safety: Assuming uninit of array of MaybeUninit is sound
        let mut buf = AlignedBuf(MaybeUninit::uninit().assume_init());

        if self.n_buffers != 0 {
            if let Err(e) = act(
                memory,
                self.src,
                self.dst,
                buf.0.as_mut_ptr().cast(),
                BUF_SIZE,
                traph,
            ) {
                return Poll::Ready(Err(e));
            }

            self.src += BUF_SIZE as u64;
            self.dst += BUF_SIZE as u64;
            self.n_buffers -= 1;

            return if self.n_buffers + self.rem == 0 {
                // If there is nothing left, we are done
                Poll::Ready(Ok(()))
            } else {
                // Otherwise let's advice to run it again
                Poll::Pending
            };
        }

        if self.rem != 0 {
            if let Err(e) = act(
                memory,
                self.src,
                self.dst,
                buf.0.as_mut_ptr().cast(),
                self.rem,
                traph,
            ) {
                return Poll::Ready(Err(e));
            }
        }

        Poll::Ready(Ok(()))
    }
}

#[inline]
unsafe fn act(
    memory: &mut Memory,
    src: u64,
    dst: u64,
    buf: *mut u8,
    count: usize,
    traph: &mut impl HandlePageFault,
) -> Result<(), BlkCopyError> {
    // Load to buffer
    memory
        .memory_access(
            MemoryAccessReason::Load,
            src,
            buf,
            count,
            perm_check::readable,
            |src, dst, count| core::ptr::copy(src, dst, count),
            traph,
        )
        .map_err(|addr| BlkCopyError {
            access_reason: MemoryAccessReason::Load,
            addr,
        })?;

    // Store from buffer
    memory
        .memory_access(
            MemoryAccessReason::Store,
            dst,
            buf,
            count,
            perm_check::writable,
            |dst, src, count| core::ptr::copy(src, dst, count),
            traph,
        )
        .map_err(|addr| BlkCopyError {
            access_reason: MemoryAccessReason::Store,
            addr,
        })?;

    Ok(())
}

/// Error occured when copying a block of memory
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlkCopyError {
    /// Kind of access
    access_reason: MemoryAccessReason,
    /// VM Address
    addr: u64,
}

impl From<BlkCopyError> for VmRunError {
    fn from(value: BlkCopyError) -> Self {
        match value.access_reason {
            MemoryAccessReason::Load => Self::LoadAccessEx(value.addr),
            MemoryAccessReason::Store => Self::StoreAccessEx(value.addr),
        }
    }
}
