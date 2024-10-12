#![feature(alloc_error_handler)]
#![feature(slice_take)]
#![no_std]

use {
    alloc::{string::String, vec::Vec},
    hblang::{codegen::Codegen, parser::FileId},
};

extern crate alloc;

wasm_rt::decl_runtime!(128 * 8 * 1024, 1024 * 4);

const MAX_INPUT_SIZE: usize = 32 * 4 * 1024;
wasm_rt::decl_buffer!(MAX_INPUT_SIZE, MAX_INPUT, INPUT, INPUT_LEN);

#[no_mangle]
unsafe fn compile_and_run(mut fuel: usize) {
    ALLOCATOR.reset();

    log::set_logger(&wasm_rt::Logger).unwrap();
    log::set_max_level(log::LevelFilter::Error);

    struct File<'a> {
        path: &'a str,
        code: &'a mut str,
    }

    let files = {
        let mut input_bytes =
            core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!(INPUT).cast::<u8>(), INPUT_LEN);

        let mut files = Vec::with_capacity(32);
        while let Some((&mut path_len, rest)) = input_bytes.split_first_chunk_mut() {
            let (path, rest) = rest.split_at_mut(u16::from_le_bytes(path_len) as usize);
            let (&mut code_len, rest) = rest.split_first_chunk_mut().unwrap();
            let (code, rest) = rest.split_at_mut(u16::from_le_bytes(code_len) as usize);
            files.push(File {
                path: core::str::from_utf8_unchecked(path),
                code: core::str::from_utf8_unchecked_mut(code),
            });
            input_bytes = rest;
        }

        hblang::quad_sort(&mut files, |a, b| a.path.cmp(b.path));
        files
    };

    let files = {
        let mut ctx = hblang::parser::ParserCtx::default();
        let paths = files.iter().map(|f| f.path).collect::<Vec<_>>();
        let mut loader = |path: &str, _: &str| Ok(paths.binary_search(&path).unwrap() as FileId);
        files
            .into_iter()
            .map(|f| {
                hblang::parser::Ast::new(
                    f.path,
                    // since 'free' does nothing this is fine
                    String::from_raw_parts(f.code.as_mut_ptr(), f.code.len(), f.code.len()),
                    &mut ctx,
                    &mut loader,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut ct = {
        let mut c = Codegen::default();
        c.files = files;
        c.generate();
        c.assemble_comptime()
    };

    while fuel != 0 {
        match ct.vm.run() {
            Ok(hbvm::VmRunOk::End) => {
                log::error!("exit code: {}", ct.vm.read_reg(1).0);
                break;
            }
            Ok(hbvm::VmRunOk::Ecall) => {
                let unknown = ct.vm.read_reg(2).0;
                log::error!("unknown ecall: {unknown}")
            }
            Ok(hbvm::VmRunOk::Timer) => fuel -= 1,
            Ok(hbvm::VmRunOk::Breakpoint) => todo!(),
            Err(e) => {
                log::error!("vm error: {e}");
                break;
            }
        }
    }
}
