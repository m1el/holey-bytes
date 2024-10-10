macro_rules! arch_specific {
    {
        $({$($cfg:tt)*} : $mod:ident;)*
    } => {
        $(
            #[cfg($($cfg)*)]
            mod $mod;

            #[cfg($($cfg)*)]
            pub use $mod::*;

            #[cfg($($cfg)*)]
            pub const FL_ARCH_SPECIFIC_SUPPORTED: bool = true;
        )*

        #[cfg(not(any($($($cfg)*),*)))]
        mod unsupported;

        #[cfg(not(any($($($cfg)*),*)))]
        pub use unsupported::*;

        #[cfg(not(any($($($cfg)*),*)))]
        pub const FL_ARCH_SPECIFIC_SUPPORTED: bool = false;
    };
}

arch_specific! {
    {target_arch = "x86_64" }: x86_64;
    {target_arch = "riscv64"}: riscv64;
    {target_arch = "aarch64"}: aarch64;
}
