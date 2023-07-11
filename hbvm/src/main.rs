use hbvm::vm::mem::{HandlePageFault, Memory, MemoryAccessReason, PageSize};

use {
    hbvm::{validate::validate, vm::Vm},
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
            let mut vm = Vm::<_, 0>::new_unchecked(&prog, TestTrapHandler);
            vm.memory.insert_test_page();
            println!("Program interrupt: {:?}", vm.run());
            println!("{:?}", vm.registers);
        }
    }
    Ok(())
}

pub fn time() -> u32 {
    9
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
