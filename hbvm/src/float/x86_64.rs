use {
    core::arch::{asm, x86_64 as arin},
    hbbytecode::RoundingMode,
};

macro_rules! gen_op {
    [$($ty:ident => $reg:ident),* $(,)?] => {
        macro_rules! op {
            $(
                ($ins:literal, $in:expr, $out:expr => $ty) => {
                    asm!(concat!($ins, " {}, {}"), out($reg) $out, in(xmm_reg) $in)
                };
            )*
        }
    };
}

gen_op![
    f32 => xmm_reg,
    f64 => xmm_reg,
    i64 => reg,
];

macro_rules! fnsdef {
    {$(
        $(#[$attr:meta])*
        $vis:vis fn $name:ident($from:ident -> $to:ident): $ins:literal;
    )*} => {$(
        $(#[$attr])*
        $vis fn $name(val: $from, mode: RoundingMode) -> $to {
            let result: $to;
            unsafe {
                set_rounding_mode(mode);
                op!($ins, val, result => $to);
                default_rounding_mode();
            }
            result
        }
    )*};
}

fnsdef! {
    /// Convert [`f64`] to [`f32`] with chosen rounding mode
    pub fn conv64to32(f64 -> f32): "cvtsd2ss";
    /// Convert [`f32`] to [`i64`] with chosen rounding mode
    pub fn f32toint(f32 -> i64): "cvttss2si";
    /// Convert [`f64`] to [`i64`] with chosen rounding mode
    pub fn f64toint(f64 -> i64): "cvttsd2si";
}

/// Set rounding mode
///
/// # Safety
/// - Do not call if rounding mode isn't [`RoundingMode::NearestEven`]
/// - Do not perform any Rust FP operations until reset using
///   [`default_rounding_mode`], you have to rely on inline assembly
#[inline(always)]
unsafe fn set_rounding_mode(mode: RoundingMode) {
    arin::_MM_SET_ROUNDING_MODE(match mode {
        RoundingMode::NearestEven => return,
        RoundingMode::Truncate => arin::_MM_ROUND_TOWARD_ZERO,
        RoundingMode::Up => arin::_MM_ROUND_UP,
        RoundingMode::Down => arin::_MM_ROUND_DOWN,
    })
}

#[inline(always)]
fn default_rounding_mode() {
    // SAFETY: This is said to be the default mode, do not trust me.
    unsafe { arin::_MM_SET_ROUNDING_MODE(arin::_MM_ROUND_NEAREST) };
}
