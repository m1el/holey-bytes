//! Holey Bytes Experimental Runtime

#![deny(unsafe_op_in_unsafe_fn)]

mod mem;

use {
    hbvm::{mem::Address, Vm, VmRunOk},
    memmap2::Mmap,
    std::{env::args, fs::File, process::exit},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("== HB×RT (Holey Bytes Experimental Runtime) v0.1 ==");
    eprintln!("[W] Currently supporting only flat images");

    if !hbvm::FL_ARCH_SPECIFIC_SUPPORTED {
        eprintln!(
            "\
                [W] Architecture not fully supported!\n    \
                    FTI32, FTI64 will yield {:#x}\n    \
                    FC64T32 will yield NaN\
            ",
            i64::MAX,
        )
    }

    let mut args = args().skip(1);
    let Some(image_path) = args.next() else {
        eprintln!("[E] Missing image path");
        exit(1);
    };

    let dsls = args.next().as_deref() == Some("-L");
    if cfg!(not(target_os = "linux")) && dsls {
        eprintln!("[E] Unsupported platform for Direct Linux syscall mode");
        exit(1);
    }

    if dsls {
        eprintln!("[I] Direct Linux syscall mode activated")
    }

    // Allocate stack
    let mut stack = unsafe { mem::alloc_stack() };
    eprintln!("[I] Stack allocated at {:p}", stack.as_ptr());

    // Load program
    eprintln!("[I] Loading image from \"{image_path}\"");
    let file_handle = File::open(image_path)?;
    let mmap = unsafe { Mmap::map(&file_handle) }?;

    eprintln!("[I] Image loaded at {:p}", mmap.as_ptr());

    let mut vm = unsafe {
        Vm::<_, 0>::new(
            mem::HostMemory,
            Address::new(mmap.as_ptr().add(stack.len()) as u64),
        )
    };
    vm.write_reg(254, stack.as_mut_ptr() as u64);

    // Execute program
    let stat = loop {
        match vm.run() {
            Ok(VmRunOk::Breakpoint) => eprintln!(
                "[I] Hit breakpoint\nIP: {}\n== Registers ==\n{:?}",
                vm.pc, vm.registers
            ),
            Ok(VmRunOk::Timer) => (),
            Ok(VmRunOk::Ecall) if dsls => unsafe {
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
            Ok(VmRunOk::Ecall) => {
                eprintln!("[E] General environment calls not supported");
                exit(1);
            }
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
