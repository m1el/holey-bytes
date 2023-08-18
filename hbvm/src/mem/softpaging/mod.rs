//! Platform independent, software paged memory implementation

pub mod icache;
pub mod lookup;
pub mod paging;

#[cfg(feature = "alloc")]
pub mod mapping;

use {
    super::{addr::Address, LoadError, Memory, MemoryAccessReason, StoreError},
    core::mem::size_of,
    icache::ICache,
    lookup::{AddrPageLookupError, AddrPageLookupOk, AddrPageLookuper},
    paging::{PageTable, Permission},
};

/// HoleyBytes software paged memory
///
/// - `OUT_PROG_EXEC`: set to `false` to disable executing program
///   not contained in initially provided program, even the pages
///   are executable
#[derive(Clone, Debug)]
pub struct SoftPagedMem<'p, PfH, const OUT_PROG_EXEC: bool = true> {
    /// Root page table
    pub root_pt:    *mut PageTable,
    /// Page fault handler
    pub pf_handler: PfH,
    /// Program memory segment
    pub program:    &'p [u8],
    /// Program instruction cache
    pub icache:     ICache,
}

impl<'p, PfH: HandlePageFault, const OUT_PROG_EXEC: bool> Memory
    for SoftPagedMem<'p, PfH, OUT_PROG_EXEC>
{
    /// Load value from an address
    ///
    /// # Safety
    /// Applies same conditions as for [`core::ptr::copy_nonoverlapping`]
    unsafe fn load(
        &mut self,
        addr: Address,
        target: *mut u8,
        count: usize,
    ) -> Result<(), LoadError> {
        self.memory_access(
            MemoryAccessReason::Load,
            addr,
            target,
            count,
            perm_check::readable,
            |src, dst, count| core::ptr::copy_nonoverlapping(src, dst, count),
        )
        .map_err(LoadError)
    }

    /// Store value to an address
    ///
    /// # Safety
    /// Applies same conditions as for [`core::ptr::copy_nonoverlapping`]
    unsafe fn store(
        &mut self,
        addr: Address,
        source: *const u8,
        count: usize,
    ) -> Result<(), StoreError> {
        self.memory_access(
            MemoryAccessReason::Store,
            addr,
            source.cast_mut(),
            count,
            perm_check::writable,
            |dst, src, count| core::ptr::copy_nonoverlapping(src, dst, count),
        )
        .map_err(StoreError)
    }

    #[inline(always)]
    unsafe fn prog_read<T>(&mut self, addr: Address) -> Option<T> {
        if OUT_PROG_EXEC && addr.truncate_usize() > self.program.len() {
            return self.icache.fetch::<T>(addr, self.root_pt);
        }

        let addr = addr.truncate_usize();
        self.program
            .get(addr..addr + size_of::<T>())
            .map(|x| x.as_ptr().cast::<T>().read())
    }

    #[inline(always)]
    unsafe fn prog_read_unchecked<T>(&mut self, addr: Address) -> T {
        if OUT_PROG_EXEC && addr.truncate_usize() > self.program.len() {
            return self
                .icache
                .fetch::<T>(addr, self.root_pt)
                .unwrap_or_else(|| core::mem::zeroed());
        }

        self.program
            .as_ptr()
            .add(addr.truncate_usize())
            .cast::<T>()
            .read()
    }
}

impl<'p, PfH: HandlePageFault, const OUT_PROG_EXEC: bool> SoftPagedMem<'p, PfH, OUT_PROG_EXEC> {
    // Everyone behold, the holy function, the god of HBVM memory accesses!

    /// Split address to pages, check their permissions and feed pointers with offset
    /// to a specified function.
    ///
    /// If page is not found, execute page fault trap handler.
    #[allow(clippy::too_many_arguments)] // Silence peasant
    fn memory_access(
        &mut self,
        reason: MemoryAccessReason,
        src: Address,
        mut dst: *mut u8,
        len: usize,
        permission_check: fn(Permission) -> bool,
        action: fn(*mut u8, *mut u8, usize),
    ) -> Result<(), Address> {
        // Memory load from program section
        let (src, len) = if src.truncate_usize() < self.program.len() as _ {
            // Allow only loads
            if reason != MemoryAccessReason::Load {
                return Err(src);
            }

            // Determine how much data to copy from here
            let to_copy = len.clamp(0, self.program.len().saturating_sub(src.truncate_usize()));

            // Perform action
            action(
                unsafe { self.program.as_ptr().add(src.truncate_usize()).cast_mut() },
                dst,
                to_copy,
            );

            // Return shifted from what we've already copied
            (
                src.saturating_add(to_copy as u64),
                len.saturating_sub(to_copy),
            )
        } else {
            (src, len) // Nothing weird!
        };

        // Nothing to copy? Don't bother doing anything, bail.
        if len == 0 {
            return Ok(());
        }

        // Create new splitter
        let mut pspl = AddrPageLookuper::new(src, len, self.root_pt);
        loop {
            match pspl.next() {
                // Page is found
                Some(Ok(AddrPageLookupOk {
                    vaddr,
                    ptr,
                    size,
                    perm,
                })) => {
                    if !permission_check(perm) {
                        return Err(vaddr);
                    }

                    // Perform specified memory action and bump destination pointer
                    action(ptr, dst, size);
                    dst = unsafe { dst.add(size) };
                }
                // No page found
                Some(Err(AddrPageLookupError { addr, size })) => {
                    // Attempt to execute page fault handler
                    if self.pf_handler.page_fault(
                        reason,
                        unsafe { &mut *self.root_pt },
                        addr,
                        size,
                        dst,
                    ) {
                        // Shift the splitter address
                        pspl.bump(size);

                        // Bump dst pointer
                        dst = unsafe { dst.add(size as _) };
                    } else {
                        return Err(addr); // Unhandleable, VM will yield.
                    }
                }
                // No remaining pages, we are done!
                None => return Ok(()),
            }
        }
    }
}

/// Extract index in page table on specified level
///
/// The level shall not be larger than 4, otherwise
/// the output of the function is unspecified (yes, it can also panic :)
pub fn addr_extract_index(addr: Address, lvl: u8) -> usize {
    debug_assert!(lvl <= 4);
    let addr = addr.get();
    usize::try_from((addr >> (lvl * 8 + 12)) & ((1 << 8) - 1)).expect("?conradluget a better CPU")
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
    const fn from_lvl(lvl: u8) -> Option<Self> {
        match lvl {
            0 => Some(PageSize::Size4K),
            1 => Some(PageSize::Size2M),
            2 => Some(PageSize::Size1G),
            _ => None,
        }
    }
}

impl core::ops::Add<PageSize> for Address {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: PageSize) -> Self::Output {
        self + (rhs as u64)
    }
}

impl core::ops::AddAssign<PageSize> for Address {
    #[inline(always)]
    fn add_assign(&mut self, rhs: PageSize) {
        *self = Self::new(self.get().wrapping_add(rhs as u64));
    }
}

/// Permisison checks
pub mod perm_check {
    use super::paging::Permission;

    /// Page is readable
    #[inline(always)]
    pub const fn readable(perm: Permission) -> bool {
        matches!(
            perm,
            Permission::Readonly | Permission::Write | Permission::Exec
        )
    }

    /// Page is writable
    #[inline(always)]
    pub const fn writable(perm: Permission) -> bool {
        matches!(perm, Permission::Write)
    }

    /// Page is executable
    #[inline(always)]
    pub const fn executable(perm: Permission) -> bool {
        matches!(perm, Permission::Exec)
    }
}

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
        pagetable: &mut PageTable,
        vaddr: Address,
        size: PageSize,
        dataptr: *mut u8,
    ) -> bool
    where
        Self: Sized;
}
