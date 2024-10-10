use std::alloc::Layout;

const STACK_SIZE: usize = 2; // MiB
type Stack = [u8; 1024 * 1024 * STACK_SIZE];

/// Allocate stack of size [`STACK_SIZE`] MiB
pub unsafe fn alloc_stack() -> Box<Stack> {
    let layout = Layout::new::<Stack>();
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }

    unsafe { Box::from_raw(ptr.cast()) }
}
