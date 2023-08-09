//! Platform independent, software paged memory implementation

pub mod paging;

use {
    super::{LoadError, Memory, MemoryAccessReason, StoreError},
    core::slice::SliceIndex,
    derive_more::Display,
    paging::{PageTable, Permission},
};

#[cfg(feature = "alloc")]
use {alloc::boxed::Box, paging::PtEntry};

/// HoleyBytes software paged memory
#[derive(Clone, Debug)]
pub struct SoftPagedMem<'p, PfH> {
    /// Root page table
    pub root_pt:    *mut PageTable,
    /// Page fault handler
    pub pf_handler: PfH,
    /// Program memory segment
    pub program:    &'p [u8],
}

impl<'p, PfH: HandlePageFault> Memory for SoftPagedMem<'p, PfH> {
    /// Load value from an address
    ///
    /// # Safety
    /// Applies same conditions as for [`core::ptr::copy_nonoverlapping`]
    unsafe fn load(&mut self, addr: u64, target: *mut u8, count: usize) -> Result<(), LoadError> {
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
        addr: u64,
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

    /// Fetch slice from program memory section
    #[inline(always)]
    fn load_prog<I>(&mut self, index: I) -> Option<&I::Output>
    where
        I: SliceIndex<[u8]>,
    {
        self.program.get(index)
    }

    /// Fetch slice from program memory section, unchecked!
    #[inline(always)]
    unsafe fn load_prog_unchecked<I>(&mut self, index: I) -> &I::Output
    where
        I: SliceIndex<[u8]>,
    {
        self.program.get_unchecked(index)
    }
}

impl<'p, PfH: HandlePageFault> SoftPagedMem<'p, PfH> {
    // Everyone behold, the holy function, the god of HBVM memory accesses!

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
    ) -> Result<(), u64> {
        // Memory load from program section
        let (src, len) = if src < self.program.len() as _ {
            // Allow only loads
            if reason != MemoryAccessReason::Load {
                return Err(src);
            }

            // Determine how much data to copy from here
            let to_copy = len.clamp(0, self.program.len().saturating_sub(src as _));

            // Perform action
            action(
                unsafe { self.program.as_ptr().add(src as _).cast_mut() },
                dst,
                to_copy,
            );

            // Return shifted from what we've already copied
            (
                src.saturating_add(to_copy as _),
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

/// Good result from address split
struct AddrPageLookupOk {
    /// Virtual address
    vaddr: u64,

    /// Pointer to the start for perform operation
    ptr: *mut u8,

    /// Size to the end of page / end of desired size
    size: usize,

    /// Page permission
    perm: Permission,
}

/// Errornous address split result
struct AddrPageLookupError {
    /// Address of failure
    addr: u64,

    /// Requested page size
    size: PageSize,
}

/// Address splitter into pages
struct AddrPageLookuper {
    /// Current address
    addr: u64,

    /// Size left
    size: usize,

    /// Page table
    pagetable: *const PageTable,
}

impl AddrPageLookuper {
    /// Create a new page lookuper
    #[inline]
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

impl Iterator for AddrPageLookuper {
    type Item = Result<AddrPageLookupOk, AddrPageLookupError>;

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
                    let entry = (*current_pt)
                        .table
                        .get_unchecked(addr_extract_index(self.addr, lvl));

                    let ptr = entry.ptr();
                    match entry.permission() {
                        // No page → page fault
                        Permission::Empty => {
                            return Some(Err(AddrPageLookupError {
                                addr: self.addr,
                                size: PageSize::from_lvl(lvl)?,
                            }))
                        }

                        // Node → proceed waking
                        Permission::Node => current_pt = ptr as _,

                        // Leaf → return relevant data
                        perm => {
                            break 'a (
                                // Pointer in host memory
                                ptr as *mut u8,
                                perm,
                                PageSize::from_lvl(lvl)?,
                                // In-page offset
                                addr_extract_index(self.addr, lvl),
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

        Some(Ok(AddrPageLookupOk {
            vaddr: self.addr,
            ptr: unsafe { base.add(offset) }, // Return pointer to the start of region
            size: avail,
            perm,
        }))
    }
}

#[cfg(feature = "alloc")]
impl<'p, A> Drop for SoftPagedMem<'p, A> {
    fn drop(&mut self) {
        let _ = unsafe { Box::from_raw(self.root_pt) };
    }
}

#[cfg(feature = "alloc")]
impl<'p, A> SoftPagedMem<'p, A> {
    /// Maps host's memory into VM's memory
    ///
    /// # Safety
    /// - Your faith in the gods of UB
    ///     - Addr-san claims it's fine but who knows is she isn't lying :ferrisSus:
    ///     - Alright, Miri-sama is also fine with this, who knows why
    pub unsafe fn map(
        &mut self,
        host: *mut u8,
        target: u64,
        perm: Permission,
        pagesize: PageSize,
    ) -> Result<(), MapError> {
        let mut current_pt = self.root_pt;

        // Decide on what level depth are we going
        let lookup_depth = match pagesize {
            PageSize::Size4K => 0,
            PageSize::Size2M => 1,
            PageSize::Size1G => 2,
        };

        // Walk pagetable levels
        for lvl in (lookup_depth + 1..5).rev() {
            let entry = (*current_pt)
                .table
                .get_unchecked_mut(addr_extract_index(target, lvl));

            let ptr = entry.ptr();
            match entry.permission() {
                // Still not on target and already seeing empty entry?
                // No worries! Let's create one (allocates).
                Permission::Empty => {
                    // Increase children count
                    (*current_pt).childen += 1;

                    let table = Box::into_raw(Box::new(paging::PtPointedData {
                        pt: PageTable::default(),
                    }));

                    core::ptr::write(entry, PtEntry::new(table, Permission::Node));
                    current_pt = table as _;
                }
                // Continue walking
                Permission::Node => current_pt = ptr as _,

                // There is some entry on place of node
                _ => return Err(MapError::PageOnNode),
            }
        }

        let node = (*current_pt)
            .table
            .get_unchecked_mut(addr_extract_index(target, lookup_depth));

        // Check if node is not mapped
        if node.permission() != Permission::Empty {
            return Err(MapError::AlreadyMapped);
        }

        // Write entry
        (*current_pt).childen += 1;
        core::ptr::write(node, PtEntry::new(host.cast(), perm));

        Ok(())
    }

    /// Unmaps pages from VM's memory
    ///
    /// If errors, it only means there is no entry to unmap and in most cases
    /// just should be ignored.
    pub fn unmap(&mut self, addr: u64) -> Result<(), NothingToUnmap> {
        let mut current_pt = self.root_pt;
        let mut page_tables = [core::ptr::null_mut(); 5];

        // Walk page table in reverse
        for lvl in (0..5).rev() {
            let entry = unsafe {
                (*current_pt)
                    .table
                    .get_unchecked_mut(addr_extract_index(addr, lvl))
            };

            let ptr = entry.ptr();
            match entry.permission() {
                // Nothing is there, throw an error, not critical!
                Permission::Empty => return Err(NothingToUnmap),
                // Node – Save to visited pagetables and continue walking
                Permission::Node => {
                    page_tables[lvl as usize] = entry;
                    current_pt = ptr as _
                }
                // Page entry – zero it out!
                // Zero page entry is completely valid entry with
                // empty permission - no UB here!
                _ => unsafe {
                    core::ptr::write_bytes(entry, 0, 1);
                    break;
                },
            }
        }

        // Now walk in order visited page tables
        for entry in page_tables.into_iter() {
            // Level not visited, skip.
            if entry.is_null() {
                continue;
            }

            unsafe {
                let children = &mut (*(*entry).ptr()).pt.childen;
                *children -= 1; // Decrease children count

                // If there are no children, deallocate.
                if *children == 0 {
                    let _ = Box::from_raw((*entry).ptr() as *mut PageTable);

                    // Zero visited entry
                    core::ptr::write_bytes(entry, 0, 1);
                } else {
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Extract index in page table on specified level
///
/// The level shall not be larger than 4, otherwise
/// the output of the function is unspecified (yes, it can also panic :)
pub fn addr_extract_index(addr: u64, lvl: u8) -> usize {
    debug_assert!(lvl <= 4);
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

/// Error mapping
#[derive(Clone, Copy, Display, Debug, PartialEq, Eq)]
pub enum MapError {
    /// Entry was already mapped
    #[display(fmt = "There is already a page mapped on specified address")]
    AlreadyMapped,
    /// When walking a page entry was
    /// encounterd.
    #[display(fmt = "There was a page mapped on the way instead of node")]
    PageOnNode,
}

/// There was no entry in page table to unmap
///
/// No worry, don't panic, nothing bad has happened,
/// but if you are 120% sure there should be something,
/// double-check your addresses.
#[derive(Clone, Copy, Display, Debug)]
#[display(fmt = "There was no entry to unmap")]
pub struct NothingToUnmap;

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
        vaddr: u64,
        size: PageSize,
        dataptr: *mut u8,
    ) -> bool
    where
        Self: Sized;
}
