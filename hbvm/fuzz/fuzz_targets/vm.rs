#![no_main]

use {
    hbbytecode::valider::validate,
    hbvm::{
        mem::{
            softpaging::{
                paging::{PageTable, Permission},
                HandlePageFault, PageSize, SoftPagedMem,
            },
            Address, MemoryAccessReason,
        },
        Vm,
    },
    libfuzzer_sys::fuzz_target,
};

fuzz_target!(|data: &[u8]| {
    if validate(data).is_ok() {
        let mut vm = unsafe {
            Vm::<_, 16384>::new(
                SoftPagedMem::<_, true> {
                    pf_handler: TestTrapHandler,
                    program:    data,
                    root_pt:    Box::into_raw(Default::default()),
                    icache:     Default::default(),
                },
                Address::new(4),
            )
        };

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

        let _ = unsafe { Box::from_raw(vm.memory.root_pt) };
    }
});

fn alloc_and_map(memory: &mut SoftPagedMem<TestTrapHandler>, at: u64) -> *mut u8 {
    let ptr = Box::into_raw(Box::<Page>::default()).cast();
    unsafe {
        memory
            .map(ptr, Address::new(at), Permission::Write, PageSize::Size4K)
            .unwrap()
    };
    ptr
}

fn unmap_and_dealloc(memory: &mut SoftPagedMem<TestTrapHandler>, ptr: *mut u8, from: u64) {
    memory.unmap(Address::new(from)).unwrap();
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
        _: &mut PageTable,
        _: Address,
        _: PageSize,
        _: *mut u8,
    ) -> bool {
        false
    }
}
