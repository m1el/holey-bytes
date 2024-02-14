use hbbytecode::RoundingMode;

#[inline(always)]
pub fn conv64to32(_: f64, _: RoundingMode) -> f32 {
    f32::NAN
}

#[inline(always)]
pub fn f32toint(_: f32, _: RoundingMode) -> i64 {
    i64::MAX
}

#[inline(always)]
pub fn f64toint(_: f64, _: RoundingMode) -> i64 {
    i64::MAX
}
