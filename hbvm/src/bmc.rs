//! Block memory copier state machine

use {
    super::{mem::MemoryAccessReason, Memory, VmRunError},
    crate::mem::Address,
    core::{mem::MaybeUninit, task::Poll},
};

/// Buffer size (defaults to 4 KiB, a smallest page size on most platforms)
const BUF_SIZE: usize = 4096;

/// Buffer of possibly uninitialised bytes, aligned to [`BUF_SIZE`]
#[repr(align(4096))]
struct AlignedBuf([MaybeUninit<u8>; BUF_SIZE]);

/// State for block memory copy
pub struct BlockCopier {
    /// Source address
    src:       Address,
    /// Destination address
    dst:       Address,
    /// How many buffer sizes to copy?
    n_buffers: usize,
    /// …and what remainds after?
    rem:       usize,
}

impl BlockCopier {
    /// Construct a new one
    #[inline]
    pub fn new(src: Address, dst: Address, count: usize) -> Self {
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
    pub unsafe fn poll(&mut self, memory: &mut impl Memory) -> Poll<Result<(), BlkCopyError>> {
        // Safety: Assuming uninit of array of MaybeUninit is sound
        let mut buf = AlignedBuf(unsafe { MaybeUninit::uninit().assume_init() });

        // We have at least one buffer size to copy
        if self.n_buffers != 0 {
            if let Err(e) = unsafe {
                act(
                    memory,
                    self.src,
                    self.dst,
                    buf.0.as_mut_ptr().cast(),
                    BUF_SIZE,
                )
            } {
                return Poll::Ready(Err(e));
            }

            // Bump source and destination address
            self.src += BUF_SIZE;
            self.dst += BUF_SIZE;

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
            if let Err(e) = unsafe {
                act(
                    memory,
                    self.src,
                    self.dst,
                    buf.0.as_mut_ptr().cast(),
                    self.rem,
                )
            } {
                return Poll::Ready(Err(e));
            }
        }

        Poll::Ready(Ok(()))
    }
}

/// Load to buffer and store from buffer
#[inline]
unsafe fn act(
    memory: &mut impl Memory,
    src: Address,
    dst: Address,
    buf: *mut u8,
    count: usize,
) -> Result<(), BlkCopyError> {
    unsafe {
        // Load to buffer
        memory
            .load(src, buf, count)
            .map_err(|super::mem::LoadError(addr)| BlkCopyError {
                access_reason: MemoryAccessReason::Load,
                addr,
            })?;

        // Store from buffer
        memory
            .store(dst, buf, count)
            .map_err(|super::mem::StoreError(addr)| BlkCopyError {
                access_reason: MemoryAccessReason::Store,
                addr,
            })?;
    }

    Ok(())
}

/// Error occured when copying a block of memory
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlkCopyError {
    /// Kind of access
    access_reason: MemoryAccessReason,
    /// VM Address
    addr: Address,
}

impl From<BlkCopyError> for VmRunError {
    fn from(value: BlkCopyError) -> Self {
        match value.access_reason {
            MemoryAccessReason::Load => Self::LoadAccessEx(value.addr),
            MemoryAccessReason::Store => Self::StoreAccessEx(value.addr),
        }
    }
}
