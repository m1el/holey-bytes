#[rustfmt::skip]
#[derive(Debug, Clone, Copy)]
pub struct Registers {
    pub a0: u64, pub b0: u64, pub c0: u64, pub d0: u64, pub e0: u64, pub f0: u64,
    pub a1: u64, pub b1: u64, pub c1: u64, pub d1: u64, pub e1: u64, pub f1: u64,
    pub a2: u64, pub b2: u64, pub c2: u64, pub d2: u64, pub e2: u64, pub f2: u64,
    pub a3: u64, pub b3: u64, pub c3: u64, pub d3: u64, pub e3: u64, pub f3: u64,
    pub a4: u64, pub b4: u64, pub c4: u64, pub d4: u64, pub e4: u64, pub f4: u64,
    pub a5: u64, pub b5: u64, pub c5: u64, pub d5: u64, pub e5: u64, pub f5: u64,
    pub a6: u64, pub b6: u64, pub c6: u64, pub d6: u64, pub e6: u64, pub f6: u64,
    pub a7: u64, pub b7: u64, pub c7: u64, pub d7: u64, pub e7: u64, pub f7: u64,
    pub a8: u64, pub b8: u64, pub c8: u64, pub d8: u64, pub e8: u64, pub f8: u64,
    pub a9: u64, pub b9: u64, pub c9: u64, pub d9: u64, pub e9: u64, pub f9: u64,
}

impl Registers {
    #[rustfmt::skip]
    pub fn new() -> Self{
        Self {
            a0: 0, b0: 0, c0: 0, d0: 0, e0: 0, f0: 0,
            a1: 0, b1: 0, c1: 0, d1: 0, e1: 0, f1: 0,
            a2: 0, b2: 0, c2: 0, d2: 0, e2: 0, f2: 0,
            a3: 0, b3: 0, c3: 0, d3: 0, e3: 0, f3: 0,
            a4: 0, b4: 0, c4: 0, d4: 0, e4: 0, f4: 0,
            a5: 0, b5: 0, c5: 0, d5: 0, e5: 0, f5: 0,
            a6: 0, b6: 0, c6: 0, d6: 0, e6: 0, f6: 0,
            a7: 0, b7: 0, c7: 0, d7: 0, e7: 0, f7: 0,
            a8: 0, b8: 0, c8: 0, d8: 0, e8: 0, f8: 0,
            a9: 0, b9: 0, c9: 0, d9: 0, e9: 0, f9: 0,
        }
    }
}
