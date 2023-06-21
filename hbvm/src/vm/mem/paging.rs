use core::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    slice::SliceIndex,
};
use delegate::delegate;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Permission {
    #[default]
    Empty,
    Node,
    Readonly,
    Write,
    Exec,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct PtEntry(u64);
impl PtEntry {
    #[inline]
    pub unsafe fn new(ptr: *mut PtPointedData, permission: Permission) -> Self {
        Self(ptr as u64 | permission as u64)
    }

    #[inline]
    pub fn permission(&self) -> Permission {
        unsafe { core::mem::transmute(self.0 as u8 & 0b111) }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(align(4096))]
pub struct PageTable([PtEntry; 512]);

impl PageTable {
    delegate!(to self.0 {
        pub unsafe fn get<I>(&self, ix: I) -> Option<&I::Output>
            where I: SliceIndex<[PtEntry]>;

        pub unsafe fn get_mut<I>(&mut self, ix: I) -> Option<&mut I::Output>
            where I: SliceIndex<[PtEntry]>;

        pub unsafe fn get_unchecked<I>(&self, index: I) -> &I::Output
            where I: SliceIndex<[PtEntry]>;

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
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(4096))]
pub union PtPointedData {
    pub pt: PageTable,
    pub page: u8,
}
