macro_rules! arch_specific {
    {
        $({$($cfg:tt)*} : $mod:ident;)*
    } => {$(
        #[cfg($($cfg)*)]
        mod $mod;

        #[cfg($($cfg)*)]
        pub use $mod::*;
    )*};
}

arch_specific! {
    {target_arch = "x86_64" }: x86_64;
    {target_arch = "riscv64"}: riscv64;
    {target_arch = "aarch64"}: aarch64;
}
