pub type Ident = u32;

const LEN_BITS: u32 = 6;

pub fn len(ident: u32) -> u32 {
    ident & ((1 << LEN_BITS) - 1)
}

pub fn is_null(ident: u32) -> bool {
    (ident >> LEN_BITS) == 0
}

pub fn pos(ident: u32) -> u32 {
    (ident >> LEN_BITS).saturating_sub(1)
}

pub fn new(pos: u32, len: u32) -> u32 {
    debug_assert!(len < (1 << LEN_BITS));
    ((pos + 1) << LEN_BITS) | len
}

pub fn range(ident: u32) -> std::ops::Range<usize> {
    let (len, pos) = (len(ident) as usize, pos(ident) as usize);
    pos..pos + len
}
