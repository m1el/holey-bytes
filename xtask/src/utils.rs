use std::mem::MaybeUninit;

pub trait IterExt: Iterator {
    fn collect_array<const N: usize>(&mut self) -> Option<[Self::Item; N]>
    where
        Self: Sized,
    {
        let mut array: [MaybeUninit<Self::Item>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for item in &mut array {
            item.write(self.next()?);
        }

        Some(array.map(|item| unsafe { item.assume_init() }))
    }
}

impl<T: Iterator> IterExt for T {}
