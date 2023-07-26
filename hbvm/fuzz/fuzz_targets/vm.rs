#![no_main]

use {
    hbvm::{
        mem::{HandlePageFault, Memory, MemoryAccessReason, PageSize},
        Vm,
    },
    libfuzzer_sys::fuzz_target,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(mut vm) = Vm::<_, 4096>::new_validated(data, TestTrapHandler, Default::default()) {
        let _ = vm.run();
    }
});

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
