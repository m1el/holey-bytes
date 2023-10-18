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
                set_rounding_mode(mode);
                asm!(
                    $ins,
                    out($outreg) result,
                    in($inreg)   val,
                );
                default_rounding_mode();
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

/// Set rounding mode
///
/// # Safety
/// - Do not call if rounding mode isn't [`RoundingMode::NearestEven`]
/// - Do not perform any Rust FP operations until reset using
///   [`default_rounding_mode`], you have to rely on inline assembly
#[inline(always)]
unsafe fn set_rounding_mode(mode: RoundingMode) {
    if mode == RoundingMode::NearestEven {
        return;
    }

    let fpcr: u64;
    asm!("mrs {}, fpcr", out(reg) fpcr);

    let fpcr = fpcr & !(0b11 << 22)
        | (match mode {
            RoundingMode::NearestEven => 0b00,
            RoundingMode::Truncate => 0b11,
            RoundingMode::Up => 0b01,
            RoundingMode::Down => 0b10,
        }) << 22;

    asm!("msr fpcr, {}", in(reg) fpcr);
}

#[inline(always)]
unsafe fn default_rounding_mode() {
    // I hope so much it gets optimised
    set_rounding_mode(RoundingMode::NearestEven);
}
