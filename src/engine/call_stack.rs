use alloc::vec::Vec;

pub type CallStack = Vec<FnCall>;
pub struct FnCall {
    pub ret: usize,
}
