use {
    hbbytecode::valider::validate,
    hbvm::{
        mem::softpaging::{paging::PageTable, HandlePageFault, PageSize, SoftPagedMem},
        MemoryAccessReason, Vm,
    },
    std::io::{stdin, Read},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut prog = vec![];
    stdin().read_to_end(&mut prog)?;

    if let Err(e) = validate(&prog) {
        eprintln!("Program validation error: {e:?}");
        return Ok(());
    } else {
        unsafe {
            let mut vm = Vm::<_, 0>::new(
                SoftPagedMem::<_, true> {
                    pf_handler: TestTrapHandler,
                    program:    &prog,
                    root_pt:    Box::into_raw(Default::default()),
                    icache:     Default::default(),
                },
                4,
            );
            let data = {
                let ptr = std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align_unchecked(
                    4096, 4096,
                ));
                if ptr.is_null() {
                    panic!("Alloc error tbhl");
                }
                ptr
            };

            vm.memory
                .map(
                    data,
                    8192,
                    hbvm::mem::softpaging::paging::Permission::Write,
                    PageSize::Size4K,
                )
                .unwrap();

            println!("Program interrupt: {:?}", vm.run());
            println!("{:?}", vm.registers);

            std::alloc::dealloc(
                data,
                std::alloc::Layout::from_size_align_unchecked(4096, 4096),
            );
            vm.memory.unmap(8192).unwrap();
            let _ = Box::from_raw(vm.memory.root_pt);
        }
    }
    Ok(())
}

pub fn time() -> u32 {
    9
}

#[derive(Default)]
struct TestTrapHandler;
impl HandlePageFault for TestTrapHandler {
    fn page_fault(
        &mut self,
        _: MemoryAccessReason,
        _: &mut PageTable,
        _: u64,
        _: PageSize,
        _: *mut u8,
    ) -> bool {
        false
    }
}
