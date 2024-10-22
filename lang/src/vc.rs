use {
    alloc::vec::Vec,
    core::{
        fmt::Debug,
        mem::MaybeUninit,
        ops::{Deref, DerefMut, Not},
        ptr::Unique,
    },
};

type Nid = u16;

const VC_SIZE: usize = 16;
const INLINE_ELEMS: usize = VC_SIZE / 2 - 1;

pub union Vc {
    inline: InlineVc,
    alloced: AllocedVc,
}

impl Default for Vc {
    fn default() -> Self {
        Vc { inline: InlineVc { elems: MaybeUninit::uninit(), cap: Default::default() } }
    }
}

impl Debug for Vc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl FromIterator<Nid> for Vc {
    fn from_iter<T: IntoIterator<Item = Nid>>(iter: T) -> Self {
        let mut slf = Self::default();
        for i in iter {
            slf.push(i);
        }
        slf
    }
}

impl Vc {
    fn is_inline(&self) -> bool {
        unsafe { self.inline.cap <= INLINE_ELEMS as Nid }
    }

    fn layout(&self) -> Option<core::alloc::Layout> {
        unsafe {
            self.is_inline().not().then(|| {
                core::alloc::Layout::array::<Nid>(self.alloced.cap as _).unwrap_unchecked()
            })
        }
    }

    pub fn len(&self) -> usize {
        unsafe {
            if self.is_inline() {
                self.inline.cap as _
            } else {
                self.alloced.len as _
            }
        }
    }

    fn len_mut(&mut self) -> &mut Nid {
        unsafe {
            if self.is_inline() {
                &mut self.inline.cap
            } else {
                &mut self.alloced.len
            }
        }
    }

    fn as_ptr(&self) -> *const Nid {
        unsafe {
            match self.is_inline() {
                true => self.inline.elems.as_ptr().cast(),
                false => self.alloced.base.as_ptr(),
            }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut Nid {
        unsafe {
            match self.is_inline() {
                true => self.inline.elems.as_mut_ptr().cast(),
                false => self.alloced.base.as_ptr(),
            }
        }
    }

    pub fn as_slice(&self) -> &[Nid] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    fn as_slice_mut(&mut self) -> &mut [Nid] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    pub fn push(&mut self, value: Nid) {
        if let Some(layout) = self.layout()
            && unsafe { self.alloced.len == self.alloced.cap }
        {
            unsafe {
                self.alloced.cap *= 2;
                self.alloced.base = Unique::new_unchecked(
                    alloc::alloc::realloc(
                        self.alloced.base.as_ptr().cast(),
                        layout,
                        self.alloced.cap as usize * core::mem::size_of::<Nid>(),
                    )
                    .cast(),
                );
            }
        } else if self.len() == INLINE_ELEMS {
            unsafe {
                let mut allcd =
                    Self::alloc((self.inline.cap + 1).next_power_of_two() as _, self.len());
                core::ptr::copy_nonoverlapping(self.as_ptr(), allcd.as_mut_ptr(), self.len());
                *self = allcd;
            }
        }

        unsafe {
            *self.len_mut() += 1;
            self.as_mut_ptr().add(self.len() - 1).write(value);
        }
    }

    unsafe fn alloc(cap: usize, len: usize) -> Self {
        debug_assert!(cap > INLINE_ELEMS);
        let layout = unsafe { core::alloc::Layout::array::<Nid>(cap).unwrap_unchecked() };
        let alloc = unsafe { alloc::alloc::alloc(layout) };
        unsafe {
            Vc {
                alloced: AllocedVc {
                    base: Unique::new_unchecked(alloc.cast()),
                    len: len as _,
                    cap: cap as _,
                },
            }
        }
    }

    pub fn swap_remove(&mut self, index: usize) {
        let len = self.len() - 1;
        self.as_slice_mut().swap(index, len);
        *self.len_mut() -= 1;
    }

    pub fn remove(&mut self, index: usize) {
        self.as_slice_mut().copy_within(index + 1.., index);
        *self.len_mut() -= 1;
    }
}

impl Drop for Vc {
    fn drop(&mut self) {
        if let Some(layout) = self.layout() {
            unsafe {
                alloc::alloc::dealloc(self.alloced.base.as_ptr().cast(), layout);
            }
        }
    }
}

impl Clone for Vc {
    fn clone(&self) -> Self {
        self.as_slice().into()
    }
}

impl IntoIterator for Vc {
    type IntoIter = VcIntoIter;
    type Item = Nid;

    fn into_iter(self) -> Self::IntoIter {
        VcIntoIter { start: 0, end: self.len(), vc: self }
    }
}

pub struct VcIntoIter {
    start: usize,
    end: usize,
    vc: Vc,
}

impl Iterator for VcIntoIter {
    type Item = Nid;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        let ret = unsafe { core::ptr::read(self.vc.as_slice().get_unchecked(self.start)) };
        self.start += 1;
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for VcIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        self.end -= 1;
        Some(unsafe { core::ptr::read(self.vc.as_slice().get_unchecked(self.end)) })
    }
}

impl ExactSizeIterator for VcIntoIter {}

impl<const SIZE: usize> From<[Nid; SIZE]> for Vc {
    fn from(value: [Nid; SIZE]) -> Self {
        value.as_slice().into()
    }
}

impl<'a> From<&'a [Nid]> for Vc {
    fn from(value: &'a [Nid]) -> Self {
        if value.len() <= INLINE_ELEMS {
            let mut dflt = Self::default();
            unsafe {
                core::ptr::copy_nonoverlapping(value.as_ptr(), dflt.as_mut_ptr(), value.len())
            };
            dflt.inline.cap = value.len() as _;
            dflt
        } else {
            let mut allcd = unsafe { Self::alloc(value.len(), value.len()) };
            unsafe {
                core::ptr::copy_nonoverlapping(value.as_ptr(), allcd.as_mut_ptr(), value.len())
            };
            allcd
        }
    }
}

impl Deref for Vc {
    type Target = [Nid];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Vc {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct InlineVc {
    cap: Nid,
    elems: MaybeUninit<[Nid; INLINE_ELEMS]>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct AllocedVc {
    cap: Nid,
    len: Nid,
    base: Unique<Nid>,
}

#[derive(Default, Clone)]
pub struct BitSet {
    data: Vec<usize>,
}

impl BitSet {
    const ELEM_SIZE: usize = core::mem::size_of::<usize>() * 8;

    pub fn clear(&mut self, bit_size: usize) {
        let new_len = bit_size.div_ceil(Self::ELEM_SIZE);
        self.data.clear();
        self.data.resize(new_len, 0);
    }

    #[track_caller]
    pub fn set(&mut self, idx: Nid) -> bool {
        let idx = idx as usize;
        let data_idx = idx / Self::ELEM_SIZE;
        let sub_idx = idx % Self::ELEM_SIZE;
        let prev = self.data[data_idx] & (1 << sub_idx);
        self.data[data_idx] |= 1 << sub_idx;
        prev == 0
    }
}
