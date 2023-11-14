//! Address lookup

use crate::mem::addr::Address;

use super::{
    addr_extract_index,
    paging::{PageTable, Permission},
    PageSize,
};

/// Good result from address split
pub struct AddrPageLookupOk {
    /// Virtual address
    pub vaddr: Address,

    /// Pointer to the start for perform operation
    pub ptr: *mut u8,

    /// Size to the end of page / end of desired size
    pub size: usize,

    /// Page permission
    pub perm: Permission,
}

/// Errornous address split result
pub struct AddrPageLookupError {
    /// Address of failure
    pub addr: Address,

    /// Requested page size
    pub size: PageSize,
}

/// Address splitter into pages
pub struct AddrPageLookuper {
    /// Current address
    addr: Address,

    /// Size left
    size: usize,

    /// Page table
    pagetable: *const PageTable,
}

impl AddrPageLookuper {
    /// Create a new page lookuper
    #[inline]
    pub const fn new(addr: Address, size: usize, pagetable: *const PageTable) -> Self {
        Self {
            addr,
            size,
            pagetable,
        }
    }

    /// Bump address by size X
    pub fn bump(&mut self, page_size: PageSize) {
        self.addr += page_size;
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
        let avail = (size as usize).saturating_sub(offset).clamp(0, self.size);
        self.bump(size);

        Some(Ok(AddrPageLookupOk {
            vaddr: self.addr,
            ptr: unsafe { base.add(offset) }, // Return pointer to the start of region
            size: avail,
            perm,
        }))
    }
}
