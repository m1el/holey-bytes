#![no_std]
#![feature(slice_take)]
#![feature(str_from_raw_parts)]
#![feature(alloc_error_handler)]

use {
    core::{
        alloc::{GlobalAlloc, Layout},
        cell::UnsafeCell,
    },
    hblang::parser::ParserCtx,
};

const ARENA_SIZE: usize = 128 * 1024;
const MAX_OUTPUT_SIZE: usize = 1024 * 10;
const MAX_INPUT_SIZE: usize = 1024 * 4;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
pub fn handle_panic(_info: &core::panic::PanicInfo) -> ! {
    //unsafe {
    //    use core::fmt::Write;
    //    let mut f = Write(&mut PANIC_MESSAGE[..]);
    //    _ = writeln!(f, "{}", info);
    //    PANIC_MESSAGE_LEN = 1024 - f.0.len();
    //}

    core::arch::wasm32::unreachable();
}

#[global_allocator]
static ALLOCATOR: ArenaAllocator = ArenaAllocator::new();

#[cfg(target_arch = "wasm32")]
#[alloc_error_handler]
fn alloc_error(_: core::alloc::Layout) -> ! {
    core::arch::wasm32::unreachable()
}

#[repr(C, align(32))]
struct ArenaAllocator {
    arena: UnsafeCell<[u8; ARENA_SIZE]>,
    head: UnsafeCell<*mut u8>,
}

impl ArenaAllocator {
    const fn new() -> Self {
        ArenaAllocator {
            arena: UnsafeCell::new([0; ARENA_SIZE]),
            head: UnsafeCell::new(core::ptr::null_mut()),
        }
    }

    unsafe fn reset(&self) {
        (*self.head.get()) = self.arena.get().cast::<u8>().add(ARENA_SIZE);
    }
}

unsafe impl Sync for ArenaAllocator {}

unsafe impl GlobalAlloc for ArenaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        let until = self.arena.get() as *mut u8;

        let new_head = (*self.head.get()).sub(size);
        let aligned_head = (new_head as usize & !(1 << (align - 1))) as *mut u8;

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

//#[no_mangle]
//static mut PANIC_MESSAGE: [u8; 1024] = unsafe { core::mem::zeroed() };
//#[no_mangle]
//static mut PANIC_MESSAGE_LEN: usize = 0;

#[no_mangle]
static mut OUTPUT: [u8; MAX_OUTPUT_SIZE] = unsafe { core::mem::zeroed() };
#[no_mangle]
static mut OUTPUT_LEN: usize = 0;

#[no_mangle]
static MAX_INPUT: usize = MAX_INPUT_SIZE;
#[no_mangle]
static mut INPUT: [u8; MAX_INPUT_SIZE] = unsafe { core::mem::zeroed() };
#[no_mangle]
static mut INPUT_LEN: usize = 0;

#[no_mangle]
unsafe extern "C" fn fmt() {
    ALLOCATOR.reset();

    let code = core::str::from_raw_parts(core::ptr::addr_of!(INPUT).cast(), INPUT_LEN);

    let arena =
        hblang::parser::Arena::with_capacity(code.len() * hblang::parser::SOURCE_TO_AST_FACTOR);
    let mut ctx = ParserCtx::default();
    let exprs = hblang::parser::Parser::parse(&mut ctx, code, "source.hb", &|_, _| Ok(0), &arena);

    let mut f = Write(&mut OUTPUT[..]);
    hblang::fmt::fmt_file(exprs, code, &mut f).unwrap();
    OUTPUT_LEN = MAX_OUTPUT_SIZE - f.0.len();
}

#[no_mangle]
unsafe extern "C" fn minify() {
    let code = core::str::from_raw_parts_mut(core::ptr::addr_of_mut!(OUTPUT).cast(), OUTPUT_LEN);
    OUTPUT_LEN = hblang::fmt::minify(code);
}
