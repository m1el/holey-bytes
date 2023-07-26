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

        // We have at least one buffer size to copy
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

            // Bump source and destination address
            //
            // If we are over the address space, bail.
            match self.src.checked_add(BUF_SIZE as u64) {
                Some(n) => self.src = n,
                None => return Poll::Ready(Err(BlkCopyError::OutOfBounds)),
            };
            
            match self.dst.checked_add(BUF_SIZE as u64) {
                Some(n) => self.dst = n,
                None => return Poll::Ready(Err(BlkCopyError::OutOfBounds)),
            };

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
        .map_err(|addr| BlkCopyError::Access {
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
        .map_err(|addr| BlkCopyError::Access {
            access_reason: MemoryAccessReason::Store,
            addr,
        })?;

    Ok(())
}

/// Error occured when copying a block of memory
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlkCopyError {
    /// Memory access error
    Access {
        /// Kind of access
        access_reason: MemoryAccessReason,
        /// VM Address
        addr: u64,
    },
    /// Address out of bounds
    OutOfBounds,
}

impl From<BlkCopyError> for VmRunError {
    fn from(value: BlkCopyError) -> Self {
        match value {
            BlkCopyError::Access {
                access_reason: MemoryAccessReason::Load,
                addr,
            } => Self::LoadAccessEx(addr),
            BlkCopyError::Access {
                access_reason: MemoryAccessReason::Store,
                addr,
            } => Self::StoreAccessEx(addr),
            BlkCopyError::OutOfBounds => Self::AddrOutOfBounds,
        }
    }
}
