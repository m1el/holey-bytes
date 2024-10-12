//! Automatic memory mapping

use {
    super::{
        addr_extract_index,
        paging::{PageTable, Permission, PtEntry, PtPointedData},
        PageSize, SoftPagedMem,
    },
    crate::{mem::addr::Address, utils::impl_display},
    alloc::boxed::Box,
};

impl<A, const OUT_PROG_EXEC: bool> SoftPagedMem<'_, A, OUT_PROG_EXEC> {
    /// Maps host's memory into VM's memory
    ///
    /// # Safety
    /// - Your faith in the gods of UB
    ///     - Addr-san claims it's fine but who knows is she isn't lying :ferrisSus:
    ///     - Alright, Miri-sama is also fine with this, who knows why
    pub unsafe fn map(
        &mut self,
        host: *mut u8,
        target: Address,
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
            let entry =
                unsafe { (*current_pt).table.get_unchecked_mut(addr_extract_index(target, lvl)) };

            let ptr = entry.ptr();
            match entry.permission() {
                // Still not on target and already seeing empty entry?
                // No worries! Let's create one (allocates).
                Permission::Empty => {
                    // Increase children count
                    unsafe { *current_pt }.childen += 1;

                    let table = Box::into_raw(Box::new(PtPointedData { pt: PageTable::default() }));

                    unsafe { core::ptr::write(entry, PtEntry::new(table, Permission::Node)) };
                    current_pt = table as _;
                }
                // Continue walking
                Permission::Node => current_pt = ptr as _,

                // There is some entry on place of node
                _ => return Err(MapError::PageOnNode),
            }
        }

        let node = unsafe {
            (*current_pt).table.get_unchecked_mut(addr_extract_index(target, lookup_depth))
        };

        // Check if node is not mapped
        if node.permission() != Permission::Empty {
            return Err(MapError::AlreadyMapped);
        }

        // Write entry
        unsafe {
            (*current_pt).childen += 1;
            core::ptr::write(node, PtEntry::new(host.cast(), perm));
        }

        Ok(())
    }

    /// Unmaps pages from VM's memory
    ///
    /// If errors, it only means there is no entry to unmap and in most cases
    /// just should be ignored.
    pub fn unmap(&mut self, addr: Address) -> Result<(), NothingToUnmap> {
        let mut current_pt = self.root_pt;
        let mut page_tables = [core::ptr::null_mut(); 5];

        // Walk page table in reverse
        for lvl in (0..5).rev() {
            let entry =
                unsafe { (*current_pt).table.get_unchecked_mut(addr_extract_index(addr, lvl)) };

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

/// Error mapping
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError {
    /// Entry was already mapped
    AlreadyMapped,
    /// When walking a page entry was
    /// encounterd.
    PageOnNode,
}

impl_display!(for MapError => match {
    Self::AlreadyMapped => "There is already a page mapped on specified address";
    Self::PageOnNode    => "There was a page mapped on the way instead of node";
});

/// There was no entry in page table to unmap
///
/// No worry, don't panic, nothing bad has happened,
/// but if you are 120% sure there should be something,
/// double-check your addresses.
#[derive(Clone, Copy, Debug)]
pub struct NothingToUnmap;
impl_display!(for NothingToUnmap => "There is no entry to unmap");
