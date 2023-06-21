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
            let mut vm = Vm::new_unchecked(&prog);
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
