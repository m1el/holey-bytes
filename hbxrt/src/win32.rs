use {
    std::path::Path,
    windows::{
        core::PCSTR,
        Win32::{
            Foundation::{GetLastError, GENERIC_READ},
            Storage::FileSystem,
            System::{
                Memory::{self},
                SystemServices::{self},
            },
        },
    },
};

/// Allocates tack for the program
pub unsafe fn alloc_stack(size: usize) -> windows::core::Result<*mut u8> {
    let ptr = unsafe {
        Memory::VirtualAlloc(
            None,
            size,
            Memory::VIRTUAL_ALLOCATION_TYPE(SystemServices::MEM_TOP_DOWN) | Memory::MEM_COMMIT,
            Memory::PAGE_READWRITE,
        )
    };

    if ptr.is_null() {
        unsafe { GetLastError() }?;
    }

    Ok(ptr.cast())
}

/// Memory map bytecode
pub unsafe fn mmap_bytecode(path: impl AsRef<Path>) -> windows::core::Result<*mut u8> {
    unsafe {
        let file = FileSystem::CreateFileA(
            PCSTR(path.as_ref().as_os_str().as_encoded_bytes().as_ptr()),
            GENERIC_READ.0,
            FileSystem::FILE_SHARE_READ,
            None,
            FileSystem::OPEN_EXISTING,
            FileSystem::FILE_ATTRIBUTE_NORMAL,
            None,
        )?;

        let h = Memory::CreateFileMappingA(
            file,
            None,
            Memory::PAGE_READONLY,
            0,
            0,
            PCSTR("Bytecode\0".as_ptr()),
        )?;

        let addr = Memory::MapViewOfFile(h, Memory::FILE_MAP_READ, 0, 0, 0);

        if addr.Value.is_null() {
            GetLastError()?;
        }

        Ok(addr.Value.cast())
    }
}

/// Catch memory access faults
pub unsafe fn catch_mafs() -> std::io::Result<()> {
    Ok(())
}
