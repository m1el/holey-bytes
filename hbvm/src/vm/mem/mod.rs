mod paging;

use self::paging::{PageTable, Permission, PtEntry};
use alloc::boxed::Box;

#[derive(Clone, Debug)]
pub struct Memory {
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

            self.root_pt_mut()[0] = entry;
        }
    }

    /// Load value from an address
    pub unsafe fn load(&self, addr: u64, target: *mut u8, count: usize) -> Result<(), ()> {
        self.memory_access(
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
        )
    }

    /// Store value to an address
    pub unsafe fn store(&mut self, addr: u64, source: *const u8, count: usize) -> Result<(), ()> {
        self.memory_access(
            addr,
            source.cast_mut(),
            count,
            |perm| perm == Permission::Write,
            |dst, src, count| core::ptr::copy_nonoverlapping(src, dst, count),
        )
    }

    /// Copy a block of memory
    pub unsafe fn block_copy(&mut self, src: u64, dst: u64, count: u64) -> Result<(), ()> {
        let count = usize::try_from(count).expect("?conradluget a better CPU");

        let mut srcs = PageSplitter::new(src, count, self.root_pt);
        let mut dsts = PageSplitter::new(dst, count, self.root_pt);
        let mut c_src = srcs.next().ok_or(())?;
        let mut c_dst = dsts.next().ok_or(())?;

        loop {
            let min_size = c_src.size.min(c_dst.size);
            unsafe {
                core::ptr::copy(c_src.ptr, c_dst.ptr, min_size);
            }

            match (
                match c_src.size.saturating_sub(min_size) {
                    0 => srcs.next(),
                    size => Some(PageSplitResult { size, ..c_src }),
                },
                match c_dst.size.saturating_sub(min_size) {
                    0 => dsts.next(),
                    size => Some(PageSplitResult { size, ..c_dst }),
                },
            ) {
                (None, None) => return Ok(()),
                (Some(src), Some(dst)) => (c_src, c_dst) = (src, dst),
                _ => return Err(()),
            }
        }
    }

    #[inline]
    pub fn root_pt(&self) -> &PageTable {
        unsafe { &*self.root_pt }
    }

    #[inline]
    pub fn root_pt_mut(&mut self) -> &mut PageTable {
        unsafe { &mut *self.root_pt }
    }

    fn memory_access(
        &self,
        src: u64,
        mut dst: *mut u8,
        len: usize,
        permission_check: impl Fn(Permission) -> bool,
        action: impl Fn(*mut u8, *mut u8, usize),
    ) -> Result<(), ()> {
        for PageSplitResult { ptr, size, perm } in PageSplitter::new(src, len, self.root_pt) {
            if !permission_check(perm) {
                return Err(());
            }

            action(ptr, dst, size);
            dst = unsafe { dst.add(size) };
        }

        Ok(())
    }
}

struct PageSplitResult {
    ptr: *mut u8,
    size: usize,
    perm: Permission,
}

struct PageSplitter {
    addr: u64,
    size: usize,
    pagetable: *const PageTable,
}

impl PageSplitter {
    pub const fn new(addr: u64, size: usize, pagetable: *const PageTable) -> Self {
        Self {
            addr,
            size,
            pagetable,
        }
    }
}

impl Iterator for PageSplitter {
    type Item = PageSplitResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.size == 0 {
            return None;
        }

        let (base, perm, size, offset) = 'a: {
            let mut current_pt = self.pagetable;
            for lvl in (0..5).rev() {
                unsafe {
                    let entry = (*current_pt).get_unchecked(
                        usize::try_from((self.addr >> (lvl * 9 + 12)) & ((1 << 9) - 1))
                            .expect("?conradluget a better CPU"),
                    );

                    let ptr = entry.ptr();
                    match entry.permission() {
                        Permission::Empty => return None,
                        Permission::Node => current_pt = ptr as _,
                        perm => {
                            break 'a (
                                ptr as *mut u8,
                                perm,
                                match lvl {
                                    0 => 4096,
                                    1 => 1024_usize.pow(2) * 2,
                                    2 => 1024_usize.pow(3),
                                    _ => return None,
                                },
                                self.addr as usize & ((1 << (lvl * 9 + 12)) - 1),
                            )
                        }
                    }
                }
            }
            return None;
        };

        let avail = (size - offset).clamp(0, self.size);
        self.addr += size as u64;
        self.size = self.size.saturating_sub(size);
        Some(PageSplitResult {
            ptr: unsafe { base.add(offset) },
            size: avail,
            perm,
        })
    }
}
