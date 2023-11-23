use {
    nix::sys::mman::{mmap, MapFlags, ProtFlags},
    std::{fs::File, num::NonZeroUsize, path::Path, process::exit},
};

/// Allocate stack for program
pub unsafe fn alloc_stack(size: usize) -> nix::Result<*mut u8> {
    unsafe {
        Ok(mmap::<std::fs::File>(
            None,
            NonZeroUsize::new(size).expect("Stack size should be > 0"),
            ProtFlags::PROT_GROWSDOWN | ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            MapFlags::MAP_GROWSDOWN
                | MapFlags::MAP_STACK
                | MapFlags::MAP_ANON
                | MapFlags::MAP_PRIVATE,
            None,
            0,
        )?
        .cast())
    }
}

/// Memory map bytecode
pub unsafe fn mmap_bytecode(path: impl AsRef<Path>) -> Result<*mut u8, Box<dyn std::error::Error>> {
    let file = File::open(&path)?;
    Ok(unsafe {
        mmap(
            None,
            NonZeroUsize::new(file.metadata()?.len() as usize).ok_or("File is empty")?,
            ProtFlags::PROT_READ,
            MapFlags::MAP_PRIVATE,
            Some(&file),
            0,
        )?
        .cast()
    })
}

/// Set handler for page fault
pub unsafe fn hook_pagefault() -> nix::Result<()> {
    unsafe {
        use nix::sys::signal;

        extern "C" fn action(
            _: std::ffi::c_int,
            info: *mut nix::libc::siginfo_t,
            _: *mut std::ffi::c_void,
        ) {
            unsafe {
                eprintln!("[E] Memory access fault at {:p}", (*info).si_addr());
                exit(2);
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

    Ok(())
}
