#![expect(dead_code)]
use {
    alloc::alloc,
    core::{
        alloc::Layout,
        fmt::Debug,
        hint::unreachable_unchecked,
        mem::MaybeUninit,
        ops::{Deref, DerefMut, Not},
        ptr::Unique,
    },
};

type Nid = u16;

pub union BitSet {
    inline: usize,
    alloced: Unique<AllocedBitSet>,
}

impl Debug for BitSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Clone for BitSet {
    fn clone(&self) -> Self {
        if self.is_inline() {
            Self { inline: unsafe { self.inline } }
        } else {
            let (data, _) = self.data_and_len();
            let (layout, _) = Self::layout(data.len());
            unsafe {
                let ptr = alloc::alloc(layout);
                ptr.copy_from_nonoverlapping(self.alloced.as_ptr() as _, layout.size());
                Self { alloced: Unique::new_unchecked(ptr as _) }
            }
        }
    }
}

impl Drop for BitSet {
    fn drop(&mut self) {
        if !self.is_inline() {
            unsafe {
                let cap = self.alloced.as_ref().cap;
                alloc::dealloc(self.alloced.as_ptr() as _, Self::layout(cap).0);
            }
        }
    }
}

impl Default for BitSet {
    fn default() -> Self {
        Self { inline: Self::FLAG }
    }
}

impl BitSet {
    const FLAG: usize = 1 << (Self::UNIT - 1);
    const INLINE_ELEMS: usize = Self::UNIT - 1;
    const UNIT: usize = core::mem::size_of::<usize>() * 8;

    fn is_inline(&self) -> bool {
        unsafe { self.inline & Self::FLAG != 0 }
    }

    fn data_and_len(&self) -> (&[usize], usize) {
        unsafe {
            if self.is_inline() {
                (core::slice::from_ref(&self.inline), Self::INLINE_ELEMS)
            } else {
                let small_vec = self.alloced.as_ref();
                (
                    core::slice::from_raw_parts(
                        &small_vec.data as *const _ as *const usize,
                        small_vec.cap,
                    ),
                    small_vec.cap * core::mem::size_of::<usize>() * 8,
                )
            }
        }
    }

    fn data_mut_and_len(&mut self) -> (&mut [usize], usize) {
        unsafe {
            if self.is_inline() {
                (core::slice::from_mut(&mut self.inline), INLINE_ELEMS)
            } else {
                let small_vec = self.alloced.as_mut();
                (
                    core::slice::from_raw_parts_mut(
                        &mut small_vec.data as *mut _ as *mut usize,
                        small_vec.cap,
                    ),
                    small_vec.cap * Self::UNIT,
                )
            }
        }
    }

    fn indexes(index: usize) -> (usize, usize) {
        (index / Self::UNIT, index % Self::UNIT)
    }

    pub fn get(&self, index: Nid) -> bool {
        let index = index as usize;
        let (data, len) = self.data_and_len();
        if index >= len {
            return false;
        }
        let (elem, bit) = Self::indexes(index);
        (unsafe { *data.get_unchecked(elem) }) & (1 << bit) != 0
    }

    pub fn set(&mut self, index: Nid) -> bool {
        let index = index as usize;
        let (mut data, len) = self.data_mut_and_len();
        if core::intrinsics::unlikely(index >= len) {
            self.grow(index.next_power_of_two().max(4 * Self::UNIT));
            (data, _) = self.data_mut_and_len();
        }

        let (elem, bit) = Self::indexes(index);
        let elem = unsafe { data.get_unchecked_mut(elem) };
        let prev = *elem;
        *elem |= 1 << bit;
        *elem != prev
    }

    fn grow(&mut self, size: usize) {
        debug_assert!(size.is_power_of_two());
        let slot_count = size / Self::UNIT;
        let (layout, off) = Self::layout(slot_count);
        let (ptr, prev_len) = unsafe {
            if self.is_inline() {
                let ptr = alloc::alloc(layout);
                *ptr.add(off).cast::<usize>() = self.inline & !Self::FLAG;
                (ptr, 1)
            } else {
                let prev_len = self.alloced.as_ref().cap;
                let (prev_layout, _) = Self::layout(prev_len);
                (alloc::realloc(self.alloced.as_ptr() as _, prev_layout, layout.size()), prev_len)
            }
        };
        unsafe {
            MaybeUninit::fill(
                core::slice::from_raw_parts_mut(
                    ptr.add(off).cast::<MaybeUninit<usize>>().add(prev_len),
                    slot_count - prev_len,
                ),
                0,
            );
            *ptr.cast::<usize>() = slot_count;
            core::ptr::write(self, Self { alloced: Unique::new_unchecked(ptr as _) });
        }
    }

    fn layout(slot_count: usize) -> (core::alloc::Layout, usize) {
        unsafe {
            core::alloc::Layout::new::<AllocedBitSet>()
                .extend(Layout::array::<usize>(slot_count).unwrap_unchecked())
                .unwrap_unchecked()
        }
    }

    pub fn iter(&self) -> BitSetIter {
        if self.is_inline() {
            BitSetIter { index: 0, current: unsafe { self.inline & !Self::FLAG }, remining: &[] }
        } else {
            let &[current, ref remining @ ..] = self.data_and_len().0 else {
                unsafe { unreachable_unchecked() }
            };
            BitSetIter { index: 0, current, remining }
        }
    }

    pub fn clear(&mut self, len: usize) {
        self.reserve(len);
        if self.is_inline() {
            unsafe { self.inline &= Self::FLAG };
        } else {
            self.data_mut_and_len().0.fill(0);
        }
    }

    pub fn units<'a>(&'a self, slot: &'a mut usize) -> &'a [usize] {
        if self.is_inline() {
            *slot = unsafe { self.inline } & !Self::FLAG;
            core::slice::from_ref(slot)
        } else {
            self.data_and_len().0
        }
    }

    pub fn reserve(&mut self, len: usize) {
        if len > self.data_and_len().1 {
            self.grow(len.next_power_of_two().max(4 * Self::UNIT));
        }
    }

    pub fn units_mut(&mut self) -> Result<&mut [usize], &mut InlineBitSetView> {
        if self.is_inline() {
            Err(unsafe {
                core::mem::transmute::<&mut usize, &mut InlineBitSetView>(&mut self.inline)
            })
        } else {
            Ok(self.data_mut_and_len().0)
        }
    }
}

pub struct InlineBitSetView(usize);

impl InlineBitSetView {
    pub(crate) fn add_mask(&mut self, tmp: usize) {
        debug_assert!(tmp & BitSet::FLAG == 0);
        self.0 |= tmp;
    }
}

pub struct BitSetIter<'a> {
    index: usize,
    current: usize,
    remining: &'a [usize],
}

impl Iterator for BitSetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current == 0 {
            self.current = *self.remining.take_first()?;
            self.index += 1;
        }

        let sub_idx = self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;
        Some(self.index * BitSet::UNIT + sub_idx)
    }
}

struct AllocedBitSet {
    cap: usize,
    data: [usize; 0],
}

#[cfg(test)]
#[test]
fn test_small_bit_set() {
    use std::vec::Vec;

    let mut sv = BitSet::default();

    sv.set(10);
    debug_assert!(sv.get(10));
    sv.set(100);
    debug_assert!(sv.get(100));
    sv.set(10000);
    debug_assert!(sv.get(10000));
    debug_assert_eq!(sv.iter().collect::<Vec<_>>(), &[10, 100, 10000]);
    sv.clear(10000);
    debug_assert_eq!(sv.iter().collect::<Vec<_>>(), &[]);
}

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

const INLINE_ELEMS: usize = VC_SIZE / 2 - 1;
const VC_SIZE: usize = 16;

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
                    alloc::realloc(
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
        let alloc = unsafe { alloc::alloc(layout) };
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
                alloc::dealloc(self.alloced.base.as_ptr().cast(), layout);
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
