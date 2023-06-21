#![no_std]

macro_rules! constmod {
    ($vis:vis $mname:ident($repr:ty) {
        $(#![doc = $mdoc:literal])?
        $($cname:ident = $val:expr $(,$doc:literal)?;)*
    }) => {
        $(#[doc = $mdoc])?
        $vis mod $mname {
            $(
                $(#[doc = $doc])?
                pub const $cname: $repr = $val;
            )*
        }
    };
}

constmod!(pub opcode(u8) {
    //! Opcode constant module

    NOP = 0, "N; Do nothing";

    ADD  = 1,  "BBB; #0 ← #1 + #2";
    SUB  = 2,  "BBB; #0 ← #1 - #2";
    MUL  = 3,  "BBB; #0 ← #1 × #2";
    AND  = 4,  "BBB; #0 ← #1 & #2";
    OR   = 5,  "BBB; #0 ← #1 | #2";
    XOR  = 6,  "BBB; #0 ← #1 ^ #2";
    SL   = 7,  "BBB; #0 ← #1 « #2";
    SR   = 8,  "BBB; #0 ← #1 » #2";
    SRS  = 9,  "BBB; #0 ← #1 » #2 (signed)";
    CMP  = 10, "BBB; #0 ← #1 <=> #2";
    CMPU = 11, "BBB; #0 ← #1 <=> #2 (unsigned)";
    DIR  = 12, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";
    NEG  = 13, "BB; #0 ← ~#1";
    NOT  = 14, "BB; #0 ← !#1";

    ADDI  = 15, "BBD; #0 ← #1 + imm #2";
    MULI  = 16, "BBD; #0 ← #1 × imm #2";
    ANDI  = 17, "BBD; #0 ← #1 & imm #2";
    ORI   = 18, "BBD; #0 ← #1 | imm #2";
    XORI  = 19, "BBD; #0 ← #1 ^ imm #2";
    SLI   = 20, "BBD; #0 ← #1 « imm #2";
    SRI   = 21, "BBD; #0 ← #1 » imm #2";
    SRSI  = 22, "BBD; #0 ← #1 » imm #2 (signed)";
    CMPI  = 23, "BBD; #0 ← #1 <=> imm #2";
    CMPUI = 24, "BBD; #0 ← #1 <=> imm #2 (unsigned)";

    CP  = 25, "BB; Copy #0 ← #1";
    SWA = 26, "BB; Swap #0 and #1";
    LI  = 27, "BD; #0 ← imm #1";
    LD  = 28, "BBDB; #0 ← [#1 + imm #3], imm #4 bytes, overflowing";
    ST  = 29, "BBDB; [#1 + imm #3] ← #0, imm #4 bytes, overflowing";
    BMC = 30, "BBD; [#0] ← [#1], imm #2 bytes";
    BRC = 31, "BBB; #0 ← #1, imm #2 registers";

    JMP   = 32, "BD;  Unconditional jump [#0 + imm #1]";
    JEQ   = 33, "BBD; if #0 = #1 → jump imm #2";
    JNE   = 34, "BBD; if #0 ≠ #1 → jump imm #2";
    JLT   = 35, "BBD; if #0 < #1 → jump imm #2";
    JGT   = 36, "BBD; if #0 > #1 → jump imm #2";
    JLTU  = 37, "BBD; if #0 < #1 → jump imm #2 (unsigned)";
    JGTU  = 38, "BBD; if #0 > #1 → jump imm #2 (unsigned)";
    ECALL = 39, "N; Issue system call";

    ADDF  = 40, "BBB; #0 ← #1 +. #2";
    MULF  = 41, "BBB; #0 ← #1 +. #2";
    DIRF  = 42, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";

    ADDFI = 43, "BBD; #0 ← #1 +. imm #2";
    MULFI = 44, "BBD; #0 ← #1 *. imm #2";
});

#[repr(packed)]
pub struct ParamBBBB(pub u8, pub u8, pub u8, pub u8);

#[repr(packed)]
pub struct ParamBBB(pub u8, pub u8, pub u8);

#[repr(packed)]
pub struct ParamBBDH(pub u8, pub u8, pub u64, pub u16);

#[repr(packed)]
pub struct ParamBBDB(pub u8, pub u8, pub u64, pub u8);

#[repr(packed)]
pub struct ParamBBD(pub u8, pub u8, pub u64);

#[repr(packed)]
pub struct ParamBB(pub u8, pub u8);

#[repr(packed)]
pub struct ParamBD(pub u8, pub u64);

/// # Safety
/// Has to be valid to be decoded from bytecode.
pub unsafe trait OpParam {}
unsafe impl OpParam for ParamBBBB {}
unsafe impl OpParam for ParamBBB {}
unsafe impl OpParam for ParamBBDB {}
unsafe impl OpParam for ParamBBDH {}
unsafe impl OpParam for ParamBBD {}
unsafe impl OpParam for ParamBB {}
unsafe impl OpParam for ParamBD {}
unsafe impl OpParam for u64 {}
unsafe impl OpParam for () {}
