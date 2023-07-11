//! Page table and associated structures implementation

use {
    core::{
        fmt::Debug,
        mem::MaybeUninit,
        ops::{Index, IndexMut},
        slice::SliceIndex,
    },
    delegate::delegate,
};

/// Page entry permission
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Permission {
    /// No page present
    #[default]
    Empty,
    /// Points to another pagetable
    Node,
    /// Page is read only
    Readonly,
    /// Page is readable and writable
    Write,
    /// Page is readable and executable
    Exec,
}

/// Page table entry
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct PtEntry(u64);
impl PtEntry {
    /// Create new
    ///
    /// # Safety
    /// - `ptr` has to point to valid data and shall not be deallocated
    ///    troughout the entry lifetime
    #[inline]
    pub unsafe fn new(ptr: *mut PtPointedData, permission: Permission) -> Self {
        Self(ptr as u64 | permission as u64)
    }

    /// Get permission
    #[inline]
    pub fn permission(&self) -> Permission {
        unsafe { core::mem::transmute(self.0 as u8 & 0b111) }
    }

    /// Get pointer to the data (leaf) or next page table (node)
    #[inline]
    pub fn ptr(&self) -> *mut PtPointedData {
        (self.0 & !((1 << 12) - 1)) as _
    }
}

impl Debug for PtEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtEntry")
            .field("ptr", &self.ptr())
            .field("permission", &self.permission())
            .finish()
    }
}

/// Page table
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(align(4096))]
pub struct PageTable([PtEntry; 512]);

impl PageTable {
    delegate!(to self.0 {
        /// Returns a reference to an element or subslice depending on the type of
        /// index.
        ///
        /// - If given a position, returns a reference to the element at that
        ///   position or `None` if out of bounds.
        /// - If given a range, returns the subslice corresponding to that range,
        ///   or `None` if out of bounds.
        ///
        pub fn get<I>(&self, ix: I) -> Option<&I::Output>
            where I: SliceIndex<[PtEntry]>;

        /// Returns a mutable reference to an element or subslice depending on the
        /// type of index (see [`get`]) or `None` if the index is out of bounds.
        pub fn get_mut<I>(&mut self, ix: I) -> Option<&mut I::Output>
            where I: SliceIndex<[PtEntry]>;

        /// Returns a reference to an element or subslice, without doing bounds
        /// checking.
        ///
        /// For a safe alternative see [`get`].
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is *[undefined behavior]*
        /// even if the resulting reference is not used.
        pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
            where I: SliceIndex<[PtEntry]>;

        /// Returns a mutable reference to an element or subslice, without doing
        /// bounds checking.
        ///
        /// For a safe alternative see [`get_mut`].
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is *[undefined behavior]*
        /// even if the resulting reference is not used.
        pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut I::Output
            where I: SliceIndex<[PtEntry]>;
    });
}

impl<Idx> Index<Idx> for PageTable
where
    Idx: SliceIndex<[PtEntry]>,
{
    type Output = Idx::Output;

    #[inline(always)]
    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

impl<Idx> IndexMut<Idx> for PageTable
where
    Idx: SliceIndex<[PtEntry]>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Default for PageTable {
    fn default() -> Self {
        // SAFETY: It's fine, zeroed page table entry is valid (= empty)
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

/// Data page table entry can possibly point to
#[derive(Clone, Copy)]
#[repr(C, align(4096))]
pub union PtPointedData {
    /// Node - next page table
    pub pt:   PageTable,
    /// Leaf
    pub page: u8,
}
