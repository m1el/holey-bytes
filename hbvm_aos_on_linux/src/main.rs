//! Holey Bytes Experimental Runtime
mod mem;

use {
    hbvm::{mem::Address, Vm, VmRunOk},
    nix::sys::mman::{mmap, MapFlags, ProtFlags},
    std::{env::args, fs::File, num::NonZeroUsize, process::exit},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("== HB×RT (Holey Bytes Linux Runtime) v0.1 ==");
    eprintln!("[W] Currently supporting only flat images");

    let Some(image_path) = args().nth(1) else {
        eprintln!("[E] Missing image path");
        exit(1);
    };

    // Load program
    eprintln!("[I] Loading image from \"{image_path}\"");
    let file = File::open(image_path)?;
    let ptr = unsafe {
        mmap(
            None,
            NonZeroUsize::new(file.metadata()?.len() as usize).ok_or("File is empty")?,
            ProtFlags::PROT_READ,
            MapFlags::MAP_PRIVATE,
            Some(&file),
            0,
        )?
    };

    eprintln!("[I] Image loaded at {ptr:p}");

    // Execute program
    let mut vm = unsafe { Vm::<_, 0>::new(mem::HostMemory, Address::new(ptr as u64)) };

    // Memory access fault handling
    unsafe {
        use nix::sys::signal;

        extern "C" fn action(
            _: std::ffi::c_int,
            info: *mut nix::libc::siginfo_t,
            _: *mut std::ffi::c_void,
        ) {
            unsafe {
                eprintln!("[E] Memory access fault at {:p}", (*info).si_addr());
            }
        }

        signal::sigaction(
            signal::Signal::SIGSEGV,
            &nix::sys::signal::SigAction::new(
                signal::SigHandler::SigAction(action),
                signal::SaFlags::SA_NODEFER,
                nix::sys::signalfd::SigSet::empty(),
            ),
        )?;
    }

    let stat = loop {
        match vm.run() {
            Ok(VmRunOk::Breakpoint) => eprintln!(
                "[I] Hit breakpoint\nIP: {}\n== Registers ==\n{:?}",
                vm.pc, vm.registers
            ),
            Ok(VmRunOk::Timer) => (),
            Ok(VmRunOk::Ecall) => {

                //     unsafe {
                //         std::arch::asm!(
                //             "syscall",
                //             inlateout("rax") vm.registers[1].0,
                //             in("rdi") vm.registers[2].0,
                //             in("rsi") vm.registers[3].0,
                //             in("rdx") vm.registers[4].0,
                //             in("r10") vm.registers[5].0,
                //             in("r8")  vm.registers[6].0,
                //             in("r9")  vm.registers[7].0,
                //         )
                //     }
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
