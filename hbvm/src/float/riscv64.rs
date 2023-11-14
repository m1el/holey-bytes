use {core::arch::asm, hbbytecode::RoundingMode};

macro_rules! roundm_op_litmode_internal {
    ($ins:literal, $in:expr, $out:expr => $outy:ident, $mode:literal $(,)?) => {
        asm!(
            concat!($ins, " {}, {}, ", $mode),
            out($outy) $out,
            in(freg)   $in,
        )
    };
}

macro_rules! gen_roundm_op_litmode {
    [$($ty:ident => $reg:ident),* $(,)?] => {
        macro_rules! roundm_op_litmode {
            $(
                ($ins:literal, $in:expr, $out:expr => $ty, $mode:literal) => {
                    roundm_op_litmode_internal!($ins, $in, $out => $reg, $mode)
                };
            )*
        }
    };
}

gen_roundm_op_litmode![
    f32 => freg,
    f64 => freg,
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
                match mode {
                    RoundingMode::NearestEven => roundm_op_litmode!($ins, val, result => $to, "rne"),
                    RoundingMode::Truncate    => roundm_op_litmode!($ins, val, result => $to, "rtz"),
                    RoundingMode::Up          => roundm_op_litmode!($ins, val, result => $to, "rup"),
                    RoundingMode::Down        => roundm_op_litmode!($ins, val, result => $to, "rdn"),
                }
            }
            result
        }
    )*};
}

fnsdef! {
    /// Convert [`f64`] to [`f32`] with chosen rounding mode
    pub fn conv64to32(f64 -> f32): "fcvt.s.d";
    /// Convert [`f32`] to [`i64`] with chosen rounding mode
    pub fn f32toint(f32 -> i64): "fcvt.l.s";
    /// Convert [`f64`] to [`i64`] with chosen rounding mode
    pub fn f64toint(f64 -> i64): "fcvt.l.d";
}
