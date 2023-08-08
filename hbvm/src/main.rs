use hbvm::softpaging::paging::PageTable;

use {
    hbbytecode::valider::validate,
    hbvm::{
        softpaging::{HandlePageFault, PageSize, SoftPagedMem},
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
            let mut vm =
                Vm::<_, 0>::new_unchecked(&prog, SoftPagedMem::<TestTrapHandler>::default());
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
                    0,
                    hbvm::softpaging::paging::Permission::Write,
                    PageSize::Size4K,
                )
                .unwrap();

            println!("Program interrupt: {:?}", vm.run());
            println!("{:?}", vm.registers);

            println!("{:?}", core::slice::from_raw_parts(data, 4096));
            std::alloc::dealloc(
                data,
                std::alloc::Layout::from_size_align_unchecked(4096, 4096),
            );
            vm.memory.unmap(0).unwrap();
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
