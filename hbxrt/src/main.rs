//! Holey Bytes Experimental Runtime

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(unix)]
#[path = "unix.rs"]
mod platform;

#[cfg(windows)]
#[path = "win32.rs"]
mod platform;

mod mem;

use {
    hbvm::{mem::Address, Vm, VmRunOk},
    std::{env::args, process::exit},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("== HB×RT (Holey Bytes Experimental Runtime) v0.1 ==");
    eprintln!("[W] Currently supporting only flat images");

    let Some(image_path) = args().nth(1) else {
        eprintln!("[E] Missing image path");
        exit(1);
    };

    // Allocate stack
    let stack_ptr = unsafe { platform::alloc_stack(1024 * 1024 * 2) }?;
    eprintln!("[I] Stack allocated at {stack_ptr:p}");

    // Load program
    eprintln!("[I] Loading image from \"{image_path}\"");
    let ptr = unsafe { platform::mmap_bytecode(image_path) }?;
    eprintln!("[I] Image loaded at {ptr:p}");

    let mut vm = unsafe { Vm::<_, 0>::new(mem::HostMemory, Address::new(ptr as u64)) };
    vm.write_reg(254, stack_ptr as u64);

    unsafe { platform::catch_mafs() }?;

    // Execute program
    let stat = loop {
        match vm.run() {
            Ok(VmRunOk::Breakpoint) => eprintln!(
                "[I] Hit breakpoint\nIP: {}\n== Registers ==\n{:?}",
                vm.pc, vm.registers
            ),
            Ok(VmRunOk::Timer) => (),
            Ok(VmRunOk::Ecall) => unsafe {
                std::arch::asm!(
                    "syscall",
                    inlateout("rax") vm.registers[1].0,
                    in("rdi") vm.registers[2].0,
                    in("rsi") vm.registers[3].0,
                    in("rdx") vm.registers[4].0,
                    in("r10") vm.registers[5].0,
                    in("r8")  vm.registers[6].0,
                    in("r9")  vm.registers[7].0,
                )
            },
            Ok(VmRunOk::End) => break Ok(()),
            Err(e) => break Err(e),
        }
    };

    eprintln!("\n== Registers ==\n{:?}", vm.registers);
    if let Err(e) = stat {
        eprintln!("\n[E] Runtime error: {e:?}");
        exit(2);
    }

    Ok(())
}
