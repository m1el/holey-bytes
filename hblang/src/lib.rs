#![feature(noop_waker)]
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
mod lexer;
mod parser;
mod tests;
mod typechk;

pub fn try_block<R>(f: impl FnOnce() -> R) -> R {
    f()
}
