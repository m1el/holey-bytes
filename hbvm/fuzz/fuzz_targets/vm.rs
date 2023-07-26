#![no_main]

use {
    hbvm::{
        mem::{HandlePageFault, Memory, MemoryAccessReason, PageSize},
        Vm,
    },
    libfuzzer_sys::fuzz_target,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(mut vm) = Vm::<_, 16384>::new_validated(data, TestTrapHandler, Default::default()) {
        // Alloc and map some memory
        let pages = [
            alloc_and_map(&mut vm.memory, 0),
            alloc_and_map(&mut vm.memory, 4096),
        ];

        // Run VM
        let _ = vm.run();

        // Unmap and dealloc the memory
        for (i, page) in pages.into_iter().enumerate() {
            unmap_and_dealloc(&mut vm.memory, page, i as u64 * 4096);
        }
    }
});

fn alloc_and_map(memory: &mut Memory, at: u64) -> *mut u8 {
    let ptr = Box::into_raw(Box::<Page>::default()).cast();
    unsafe {
        memory
            .map(
                ptr,
                at,
                hbvm::mem::paging::Permission::Write,
                PageSize::Size4K,
            )
            .unwrap()
    };
    ptr
}

fn unmap_and_dealloc(memory: &mut Memory, ptr: *mut u8, from: u64) {
    memory.unmap(from).unwrap();
    let _ = unsafe { Box::from_raw(ptr.cast::<Page>()) };
}

#[repr(align(4096))]
struct Page([u8; 4096]);
impl Default for Page {
    fn default() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

struct TestTrapHandler;
impl HandlePageFault for TestTrapHandler {
    fn page_fault(
        &mut self,
        _: MemoryAccessReason,
        _: &mut Memory,
        _: u64,
        _: PageSize,
        _: *mut u8,
    ) -> bool {
        false
    }
}
