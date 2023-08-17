//! Program instruction cache

use {
    super::{lookup::AddrPageLookuper, paging::PageTable, PageSize},
    core::{
        mem::{size_of, MaybeUninit},
        ptr::{copy_nonoverlapping, NonNull},
    },
};

/// Instruction cache
#[derive(Clone, Debug)]
pub struct ICache {
    /// Current page address base
    base: u64,
    /// Curent page pointer
    data: Option<NonNull<u8>>,
    /// Current page size
    size: PageSize,
    /// Address mask
    mask: u64,
}

impl Default for ICache {
    fn default() -> Self {
        Self {
            base: Default::default(),
            data: Default::default(),
            size: PageSize::Size4K,
            mask: Default::default(),
        }
    }
}

impl ICache {
    /// Fetch instruction from cache
    ///
    /// # Safety
    /// `T` should be valid to read from instruction memory
    pub(super) unsafe fn fetch<T>(&mut self, addr: u64, root_pt: *const PageTable) -> Option<T> {
        let mut ret = MaybeUninit::<T>::uninit();

        let pbase = self
            .data
            .or_else(|| self.fetch_page(self.base.checked_add(self.size as _)?, root_pt))?;

        // Get address base
        let base = addr & self.mask;

        // Base not matching, fetch anew
        if base != self.base {
            self.fetch_page(base, root_pt)?;
        };

        let offset = addr & !self.mask;
        let requ_size = size_of::<T>();

        // Page overflow
        let rem = (offset as usize)
            .saturating_add(requ_size)
            .saturating_sub(self.size as _);
        let first_copy = requ_size.saturating_sub(rem);

        // Copy non-overflowing part
        copy_nonoverlapping(pbase.as_ptr(), ret.as_mut_ptr().cast::<u8>(), first_copy);

        // Copy overflow
        if rem != 0 {
            let pbase = self.fetch_page(self.base.checked_add(self.size as _)?, root_pt)?;

            // Unlikely, unsupported scenario
            if rem > self.size as _ {
                return None;
            }

            copy_nonoverlapping(
                pbase.as_ptr(),
                ret.as_mut_ptr().cast::<u8>().add(first_copy),
                rem,
            );
        }

        Some(ret.assume_init())
    }

    /// Fetch a page
    unsafe fn fetch_page(&mut self, addr: u64, pt: *const PageTable) -> Option<NonNull<u8>> {
        let res = AddrPageLookuper::new(addr, 0, pt).next()?.ok()?;
        if !super::perm_check::executable(res.perm) {
            return None;
        }

        (self.size, self.mask) = match res.size {
            4096 => (PageSize::Size4K, !((1 << 8) - 1)),
            2097152 => (PageSize::Size2M, !((1 << (8 * 2)) - 1)),
            1073741824 => (PageSize::Size1G, !((1 << (8 * 3)) - 1)),
            _ => return None,
        };
        self.data = Some(NonNull::new(res.ptr)?);
        self.base = addr & self.mask;
        self.data
    }
}
