#![feature(noop_waker)]
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

#[repr(packed)]
struct Args<A, B, C, D>(u8, A, B, C, D);
fn as_bytes<T>(args: &T) -> &[u8] {
    unsafe { core::slice::from_raw_parts(args as *const _ as *const u8, core::mem::size_of::<T>()) }
}

pub fn try_block<R>(f: impl FnOnce() -> R) -> R {
    f()
}
