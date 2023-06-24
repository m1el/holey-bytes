mod paging;

use core::mem::MaybeUninit;

use self::paging::{PageTable, Permission, PtEntry};
use super::{trap::HandleTrap, VmRunError};
use alloc::boxed::Box;
use derive_more::Display;

/// HoleyBytes virtual memory
#[derive(Clone, Debug)]
pub struct Memory {
    /// Root page table
    root_pt: *mut PageTable,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            root_pt: Box::into_raw(Box::default()),
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        let _ = unsafe { Box::from_raw(self.root_pt) };
    }
}

impl Memory {
    // HACK: Just for allocation testing, will be removed when proper memory interfaces
    // implemented.
    pub fn insert_test_page(&mut self) {
        unsafe {
            let mut entry = PtEntry::new(
                {
                    let layout = alloc::alloc::Layout::from_size_align_unchecked(4096, 4096);
                    let ptr = alloc::alloc::alloc_zeroed(layout);
                    if ptr.is_null() {
                        alloc::alloc::handle_alloc_error(layout);
                    }

                    core::ptr::write_bytes(ptr, 69, 10);
                    ptr.cast()
                },
                Permission::Write,
            );

            for _ in 0..4 {
                let mut pt = Box::<PageTable>::default();
                pt[0] = entry;
                entry = PtEntry::new(Box::into_raw(pt) as _, Permission::Node);
            }

            (*self.root_pt)[0] = entry;
        }
    }

    /// Load value from an address
    ///
    /// # Safety
    /// Applies same conditions as for [`core::ptr::copy_nonoverlapping`]
    pub unsafe fn load(
        &mut self,
        addr: u64,
        target: *mut u8,
        count: usize,
        traph: &mut impl HandleTrap,
    ) -> Result<(), LoadError> {
        self.memory_access(
            MemoryAccessReason::Load,
            addr,
            target,
            count,
            |perm| {
                matches!(
                    perm,
                    Permission::Readonly | Permission::Write | Permission::Exec
                )
            },
            |src, dst, count| core::ptr::copy_nonoverlapping(src, dst, count),
            traph,
        )
        .map_err(|_| LoadError)
    }

    /// Store value to an address
    ///
    /// # Safety
    /// Applies same conditions as for [`core::ptr::copy_nonoverlapping`]
    pub unsafe fn store(
        &mut self,
        addr: u64,
        source: *const u8,
        count: usize,
        traph: &mut impl HandleTrap,
    ) -> Result<(), StoreError> {
        self.memory_access(
            MemoryAccessReason::Store,
            addr,
            source.cast_mut(),
            count,
            |perm| perm == Permission::Write,
            |dst, src, count| core::ptr::copy_nonoverlapping(src, dst, count),
            traph,
        )
        .map_err(|_| StoreError)
    }

    /// Copy a block of memory
    ///
    /// # Safety
    /// - Same as for [`Self::load`] and [`Self::store`]
    /// - Your faith in the gods of UB
    ///     - Addr-san claims it's fine but who knows is she isn't lying :ferrisSus:
    pub unsafe fn block_copy(
        &mut self,
        src: u64,
        dst: u64,
        count: usize,
        traph: &mut impl HandleTrap,
    ) -> Result<(), MemoryAccessReason> {
        // Yea, i know it is possible to do this more efficiently, but I am too lazy.

        const STACK_BUFFER_SIZE: usize = 512;

        // Decide if to use stack-allocated buffer or to heap allocate
        // Deallocation is again decided on size at the end of the function
        let mut buf = MaybeUninit::<[u8; STACK_BUFFER_SIZE]>::uninit();
        let buf = if count <= STACK_BUFFER_SIZE {
            buf.as_mut_ptr().cast()
        } else {
            unsafe {
                let layout = core::alloc::Layout::from_size_align_unchecked(count, 1);
                let ptr = alloc::alloc::alloc(layout);
                if ptr.is_null() {
                    alloc::alloc::handle_alloc_error(layout);
                }

                ptr
            }
        };

        // Perform memory block transfer
        let status = (|| {
            // Load to buffer
            self.memory_access(
                MemoryAccessReason::Load,
                src,
                buf,
                count,
                |perm| {
                    matches!(
                        perm,
                        Permission::Readonly | Permission::Write | Permission::Exec
                    )
                },
                |src, dst, count| core::ptr::copy(src, dst, count),
                traph,
            )
            .map_err(|_| MemoryAccessReason::Load)?;

            // Store from buffer
            self.memory_access(
                MemoryAccessReason::Store,
                dst,
                buf,
                count,
                |perm| perm == Permission::Write,
                |dst, src, count| core::ptr::copy(src, dst, count),
                traph,
            )
            .map_err(|_| MemoryAccessReason::Store)?;

            Ok::<_, MemoryAccessReason>(())
        })();

        // Deallocate if used heap-allocated array
        if count > STACK_BUFFER_SIZE {
            alloc::alloc::dealloc(
                buf,
                core::alloc::Layout::from_size_align_unchecked(count, 1),
            );
        }

        status
    }

    /// Split address to pages, check their permissions and feed pointers with offset
    /// to a specified function.
    ///
    /// If page is not found, execute page fault trap handler.
    #[allow(clippy::too_many_arguments)] // Silence peasant
    fn memory_access(
        &mut self,
        reason: MemoryAccessReason,
        src: u64,
        mut dst: *mut u8,
        len: usize,
        permission_check: fn(Permission) -> bool,
        action: fn(*mut u8, *mut u8, usize),
        traph: &mut impl HandleTrap,
    ) -> Result<(), ()> {
        let mut pspl = AddrSplitter::new(src, len, self.root_pt);
        loop {
            match pspl.next() {
                // Page found
                Some(Ok(AddrSplitOk { ptr, size, perm })) => {
                    if !permission_check(perm) {
                        return Err(());
                    }

                    // Perform memory action and bump dst pointer
                    action(ptr, dst, size);
                    dst = unsafe { dst.add(size) };
                }
                Some(Err(AddrSplitError { addr, size })) => {
                    // Execute page fault handler
                    if traph.page_fault(reason, self, addr, size, dst) {
                        // Shift the splitter address
                        pspl.bump(size);

                        // Bump dst pointer
                        dst = unsafe { dst.add(size as _) };
                    } else {
                        return Err(()); // Unhandleable
                    }
                }
                None => return Ok(()),
            }
        }
    }
}

/// Result from address split
struct AddrSplitOk {
    /// Pointer to the start for perform operation
    ptr: *mut u8,

    /// Size to the end of page / end of desired size
    size: usize,

    /// Page permission
    perm: Permission,
}

struct AddrSplitError {
    /// Address of failure
    addr: u64,

    /// Requested page size
    size: PageSize,
}

/// Address splitter into pages
struct AddrSplitter {
    /// Current address
    addr: u64,

    /// Size left
    size: usize,

    /// Page table
    pagetable: *const PageTable,
}

impl AddrSplitter {
    /// Create a new page splitter
    pub const fn new(addr: u64, size: usize, pagetable: *const PageTable) -> Self {
        Self {
            addr,
            size,
            pagetable,
        }
    }

    /// Bump address by size X
    fn bump(&mut self, page_size: PageSize) {
        self.addr += page_size as u64;
        self.size = self.size.saturating_sub(page_size as _);
    }
}

impl Iterator for AddrSplitter {
    type Item = Result<AddrSplitOk, AddrSplitError>;

    fn next(&mut self) -> Option<Self::Item> {
        // The end, everything is fine
        if self.size == 0 {
            return None;
        }

        let (base, perm, size, offset) = 'a: {
            let mut current_pt = self.pagetable;

            // Walk the page table
            for lvl in (0..5).rev() {
                // Get an entry
                unsafe {
                    let entry = (*current_pt).get_unchecked(
                        usize::try_from((self.addr >> (lvl * 9 + 12)) & ((1 << 9) - 1))
                            .expect("?conradluget a better CPU"),
                    );

                    let ptr = entry.ptr();
                    match entry.permission() {
                        // No page → page fault
                        Permission::Empty => {
                            return Some(Err(AddrSplitError {
                                addr: self.addr,
                                size: PageSize::from_lvl(lvl)?,
                            }))
                        }

                        // Node → proceed waking
                        Permission::Node => current_pt = ptr as _,

                        // Leaft → return relevant data
                        perm => {
                            break 'a (
                                // Pointer in host memory
                                ptr as *mut u8,
                                perm,
                                PageSize::from_lvl(lvl)?,
                                // In-page offset
                                self.addr as usize & ((1 << (lvl * 9 + 12)) - 1),
                            );
                        }
                    }
                }
            }
            return None; // Reached the end (should not happen)
        };

        // Get available byte count in the selected page with offset
        let avail = (size as usize - offset).clamp(0, self.size);
        self.bump(size);

        Some(Ok(AddrSplitOk {
            ptr: unsafe { base.add(offset) }, // Return pointer to the start of region
            size: avail,
            perm,
        }))
    }
}

/// Page size
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSize {
    /// 4 KiB page (on level 0)
    Size4K = 4096,

    /// 2 MiB page (on level 1)
    Size2M = 1024 * 1024 * 2,

    /// 1 GiB page (on level 2)
    Size1G = 1024 * 1024 * 1024,
}

impl PageSize {
    /// Convert page table level to size of page
    fn from_lvl(lvl: u8) -> Option<Self> {
        match lvl {
            0 => Some(PageSize::Size4K),
            1 => Some(PageSize::Size2M),
            2 => Some(PageSize::Size1G),
            _ => None,
        }
    }
}

/// Unhandled load access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub struct LoadError;

/// Unhandled store access trap
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub struct StoreError;

#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub enum MemoryAccessReason {
    Load,
    Store,
}

impl From<MemoryAccessReason> for VmRunError {
    fn from(value: MemoryAccessReason) -> Self {
        match value {
            MemoryAccessReason::Load => Self::LoadAccessEx,
            MemoryAccessReason::Store => Self::StoreAccessEx,
        }
    }
}

impl From<LoadError> for VmRunError {
    fn from(_: LoadError) -> Self {
        Self::LoadAccessEx
    }
}

impl From<StoreError> for VmRunError {
    fn from(_: StoreError) -> Self {
        Self::StoreAccessEx
    }
}
