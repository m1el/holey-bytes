#![feature(noop_waker)]
#![feature(let_chains)]
#![feature(non_null_convenience)]
#![allow(dead_code)]
#![feature(const_mut_refs)]

#[macro_export]
macro_rules! run_tests {
    ($runner:path: $($name:ident => $input:expr;)*) => {$(
        #[test]
        fn $name() {
            $crate::tests::run_test(std::any::type_name_of_val(&$name), $input, $runner);
        }
    )*};
}

mod codegen;
mod ident;
mod instrs;
mod lexer;
mod parser;
mod tests;
mod typechk;

#[inline]
unsafe fn encode<T>(instr: T) -> (usize, [u8; instrs::MAX_SIZE]) {
    let mut buf = [0; instrs::MAX_SIZE];
    std::ptr::write(buf.as_mut_ptr() as *mut T, instr);
    (std::mem::size_of::<T>(), buf)
}
