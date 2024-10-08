#![no_std]
#![feature(slice_take)]
#![feature(str_from_raw_parts)]

use hblang::parser::ParserCtx;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn handle_panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
};

const ARENA_SIZE: usize = 128 * 1024;

#[repr(C, align(32))]
struct SimpleAllocator {
    arena: UnsafeCell<[u8; ARENA_SIZE]>,
    head: UnsafeCell<*mut u8>,
}

impl SimpleAllocator {
    const fn new() -> Self {
        SimpleAllocator {
            arena: UnsafeCell::new([0; ARENA_SIZE]),
            head: UnsafeCell::new(core::ptr::null_mut()),
        }
    }

    unsafe fn reset(&self) {
        (*self.head.get()) = self.arena.get().cast::<u8>().add(ARENA_SIZE);
    }
}

unsafe impl Sync for SimpleAllocator {}

unsafe impl GlobalAlloc for SimpleAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        let until = self.arena.get() as *mut u8;

        let new_head = (*self.head.get()).sub(size);
        let aligned_head = (new_head as usize & (1 << (align - 1))) as *mut u8;

        if until > aligned_head {
            return core::ptr::null_mut();
        }

        *self.head.get() = aligned_head;
        aligned_head
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        /* lol */
    }
}

#[cfg_attr(target_arch = "wasm32", global_allocator)]
static ALLOCATOR: SimpleAllocator = SimpleAllocator::new();

const MAX_OUTPUT_SIZE: usize = 1024 * 10;

#[no_mangle]
static mut OUTPUT: [u8; MAX_OUTPUT_SIZE] = unsafe { core::mem::zeroed() };

#[no_mangle]
static mut OUTPUT_LEN: usize = 0;

#[no_mangle]
unsafe extern "C" fn fmt(code: *const u8, len: usize) {
    ALLOCATOR.reset();

    let code = core::str::from_raw_parts(code, len);

    let arena = hblang::parser::Arena::default();
    let mut ctx = ParserCtx::default();
    let exprs = hblang::parser::Parser::parse(&mut ctx, code, "source.hb", &|_, _| Ok(0), &arena);

    struct Write<'a>(&'a mut [u8]);

    impl core::fmt::Write for Write<'_> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            if let Some(m) = self.0.take_mut(..s.len()) {
                m.copy_from_slice(s.as_bytes());
                Ok(())
            } else {
                Err(core::fmt::Error)
            }
        }
    }

    let mut f = Write(unsafe { &mut OUTPUT[..] });
    hblang::fmt::fmt_file(exprs, code, &mut f).unwrap();
    unsafe { OUTPUT_LEN = MAX_OUTPUT_SIZE - f.0.len() };
}
