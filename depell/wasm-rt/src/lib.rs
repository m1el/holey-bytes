#![feature(alloc_error_handler)]
#![feature(slice_take)]
#![no_std]

use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
};

extern crate alloc;

#[macro_export]
macro_rules! decl_buffer {
    ($cap:expr, $export_cap:ident, $export_base:ident, $export_len:ident) => {
        #[no_mangle]
        static $export_cap: usize = $cap;
        #[no_mangle]
        static mut $export_base: [u8; $cap] = [0; $cap];
        #[no_mangle]
        static mut $export_len: usize = 0;
    };
}

#[macro_export]
macro_rules! decl_runtime {
    ($memory_size:expr, $max_panic_size:expr) => {
        #[cfg(target_arch = "wasm32")]
        #[panic_handler]
        pub fn handle_panic(_info: &core::panic::PanicInfo) -> ! {
            #[cfg(debug_assertions)]
            {
                #[no_mangle]
                static mut PANIC_MESSAGE: [u8; $max_panic_size] = [0; $max_panic_size];
                #[no_mangle]
                static mut PANIC_MESSAGE_LEN: usize = 0;

                unsafe {
                    use core::fmt::Write;
                    let mut f = Write(&mut PANIC_MESSAGE[..]);
                    _ = writeln!(f, "{}", _info);
                    PANIC_MESSAGE_LEN = 1024 - f.0.len();
                }
            }

            core::arch::wasm32::unreachable();
        }

        #[global_allocator]
        static ALLOCATOR: $crate::ArenaAllocator<{ $memory_size }> = $crate::ArenaAllocator::new();

        #[cfg(target_arch = "wasm32")]
        #[alloc_error_handler]
        fn alloc_error(_: core::alloc::Layout) -> ! {
            core::arch::wasm32::unreachable()
        }
    };
}

#[cfg(feature = "log")]
pub struct Logger;

#[cfg(feature = "log")]
impl log::Log for Logger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            const MAX_LOG_MESSAGE: usize = 1024 * 4;
            #[no_mangle]
            static mut LOG_MESSAGES: [u8; MAX_LOG_MESSAGE] = [0; MAX_LOG_MESSAGE];
            #[no_mangle]
            static mut LOG_MESSAGES_LEN: usize = 0;

            unsafe {
                use core::fmt::Write;
                let mut f = Write(&mut LOG_MESSAGES[LOG_MESSAGES_LEN..]);
                _ = writeln!(f, "{}", record.args());
                LOG_MESSAGES_LEN = MAX_LOG_MESSAGE - f.0.len();
            }
        }
    }

    fn flush(&self) {}
}

pub struct ArenaAllocator<const SIZE: usize> {
    arena: UnsafeCell<[u8; SIZE]>,
    head: UnsafeCell<*mut u8>,
}

impl<const SIZE: usize> ArenaAllocator<SIZE> {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        ArenaAllocator {
            arena: UnsafeCell::new([0; SIZE]),
            head: UnsafeCell::new(core::ptr::null_mut()),
        }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn reset(&self) {
        (*self.head.get()) = self.arena.get().cast::<u8>().add(SIZE);
    }
}

unsafe impl<const SIZE: usize> Sync for ArenaAllocator<SIZE> {}

unsafe impl<const SIZE: usize> GlobalAlloc for ArenaAllocator<SIZE> {
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

pub struct Write<'a>(pub &'a mut [u8]);

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
