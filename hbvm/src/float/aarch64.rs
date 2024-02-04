use {core::arch::asm, hbbytecode::RoundingMode};

macro_rules! fnsdef {
    {$(
        $(#[$attr:meta])*
        $vis:vis fn $name:ident[$inreg:ident -> $outreg:ident]($from:ident -> $to:ident): $ins:literal;
    )*} => {$(
        $(#[$attr])*
        $vis fn $name(val: $from, mode: RoundingMode) -> $to {
            let result: $to;
            unsafe {
                if mode == RoundingMode::NearestEven {
                    return;
                }

                let fpcr: u64;
                unsafe { asm!("mrs {}, fpcr", out(reg) fpcr) };

                let fpcr_new = fpcr & !(0b11 << 22)
                    | (match mode {
                        RoundingMode::NearestEven => 0b00,
                        RoundingMode::Truncate => 0b11,
                        RoundingMode::Up => 0b01,
                        RoundingMode::Down => 0b10,
                    }) << 22;

                unsafe { asm!("msr fpcr, {}", in(reg) fpcr_new) };
                asm!(
                    $ins,
                    out($outreg) result,
                    in($inreg)   val,
                );
                unsafe { asm!("msr fpcr, {}", in(reg) fpcr) };
            }
            result
        }
    )*};
}

fnsdef! {
    /// Convert [`f64`] to [`f32`] with chosen rounding mode
    pub fn conv64to32[vreg -> vreg](f64 -> f32): "fcvt {:s}, {:d}";

    /// Convert [`f32`] to [`i64`] with chosen rounding mode
    pub fn f32toint[vreg -> reg](f32 -> i64): "fcvtzs {}, {:s}";

    /// Convert [`f64`] to [`i64`] with chosen rounding mode
    pub fn f64toint[vreg -> reg](f64 -> i64): "fcvtzs {}, {:d}";
}
