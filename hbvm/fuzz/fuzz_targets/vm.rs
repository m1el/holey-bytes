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
        let page = Box::into_raw(Box::<Page>::default());

        unsafe {
            vm.memory
                .map(
                    page.cast(),
                    0,
                    hbvm::mem::paging::Permission::Write,
                    PageSize::Size4K,
                )
                .unwrap()
        };

        let _ = vm.run();

        vm.memory.unmap(0).unwrap();
        let _ = unsafe { Box::from_raw(page) };
    }
});

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
