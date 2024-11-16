#![no_std]
#![feature(str_from_raw_parts)]
#![feature(alloc_error_handler)]

use hblang::{fmt, parser};

wasm_rt::decl_runtime!(128 * 1024, 1024 * 4);

const MAX_OUTPUT_SIZE: usize = 1024 * 10;
wasm_rt::decl_buffer!(MAX_OUTPUT_SIZE, MAX_OUTPUT, OUTPUT, OUTPUT_LEN);

const MAX_INPUT_SIZE: usize = 1024 * 4;
wasm_rt::decl_buffer!(MAX_INPUT_SIZE, MAX_INPUT, INPUT, INPUT_LEN);

#[no_mangle]
unsafe extern "C" fn fmt() {
    ALLOCATOR.reset();

    let code = core::str::from_raw_parts(core::ptr::addr_of!(INPUT).cast(), INPUT_LEN);

    let arena = parser::Arena::with_capacity(code.len() * parser::SOURCE_TO_AST_FACTOR);
    let mut ctx = parser::Ctx::default();
    let exprs = parser::Parser::parse(&mut ctx, code, "source.hb", &mut parser::no_loader, &arena);

    let mut f = wasm_rt::Write(&mut OUTPUT[..]);
    fmt::fmt_file(exprs, code, &mut f).unwrap();
    OUTPUT_LEN = MAX_OUTPUT_SIZE - f.0.len();
}

#[no_mangle]
unsafe extern "C" fn tok() {
    let code = core::slice::from_raw_parts_mut(
        core::ptr::addr_of_mut!(OUTPUT).cast(), OUTPUT_LEN);
    OUTPUT_LEN = fmt::get_token_kinds(code);
}

#[no_mangle]
unsafe extern "C" fn minify() {
    let code = core::str::from_raw_parts_mut(
        core::ptr::addr_of_mut!(OUTPUT).cast(), OUTPUT_LEN);
    OUTPUT_LEN = fmt::minify(code);
}
