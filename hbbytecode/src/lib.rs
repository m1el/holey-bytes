#![no_std]

mod gen_valider;

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

/// Invoke macro with bytecode definition format
/// # Input syntax
/// ```no_run
/// macro!(
///     INSTRUCTION_TYPE(p0: TYPE, p1: TYPE, …)
///         => [INSTRUCTION_A, INSTRUCTION_B, …],
///     …
/// );
/// ```
/// - Instruction type determines opcode-generic, instruction-type-specific
///   function. Name: `i_param_INSTRUCTION_TYPE`
/// - Per-instructions there will be generated opcode-specific functions calling the generic ones
/// - Operand types
///     - R: Register (u8)
///     - I: Immediate (implements [`crate::Imm`] trait)
///     - L: Memory load / store size (u16)
///     - Other types are identity-mapped
#[macro_export]
macro_rules! invoke_with_def {
    ($macro:path) => {
        $macro!(
            bbbb(p0: R, p1: R, p2: R, p3: R)
                => [DIR, DIRF, FMAF],
            bbb(p0: R, p1: R, p2: R)
                => [ADD, SUB, MUL, AND, OR, XOR, SL, SR, SRS, CMP, CMPU, BRC, ADDF, SUBF, MULF],
            bbdh(p0: R, p1: R, p2: I, p3: L)
                => [LD, ST],
            bbd(p0: R, p1: R, p2: I)
                => [ADDI, MULI, ANDI, ORI, XORI, CMPI, CMPUI, BMC, JAL, JEQ, JNE, JLT, JGT, JLTU,
                    JGTU, ADDFI, MULFI],
            bbw(p0: R, p1: R, p2: u32)
                => [SLI, SRI, SRSI],
            bb(p0: R, p1: R)
                => [NEG, NOT, CP, SWA, NEGF, ITF, FTI],
            bd(p0: R, p1: I)
                => [LI],
            n()
                => [UN, NOP, ECALL],
        );
    };
}

invoke_with_def!(gen_valider::gen_valider);

constmod!(pub opcode(u8) {
    //! Opcode constant module

    UN  = 0, "N; Raises a trap";
    NOP = 1, "N; Do nothing";

    ADD  = 2,  "BBB; #0 ← #1 + #2";
    SUB  = 3,  "BBB; #0 ← #1 - #2";
    MUL  = 4,  "BBB; #0 ← #1 × #2";
    AND  = 5,  "BBB; #0 ← #1 & #2";
    OR   = 6,  "BBB; #0 ← #1 | #2";
    XOR  = 7,  "BBB; #0 ← #1 ^ #2";
    SL   = 8,  "BBB; #0 ← #1 « #2";
    SR   = 9,  "BBB; #0 ← #1 » #2";
    SRS  = 10,  "BBB; #0 ← #1 » #2 (signed)";
    CMP  = 11, "BBB; #0 ← #1 <=> #2";
    CMPU = 12, "BBB; #0 ← #1 <=> #2 (unsigned)";
    DIR  = 13, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";
    NEG  = 14, "BB; #0 ← -#1";
    NOT  = 15, "BB; #0 ← !#1";

    ADDI  = 16, "BBD; #0 ← #1 + imm #2";
    MULI  = 17, "BBD; #0 ← #1 × imm #2";
    ANDI  = 18, "BBD; #0 ← #1 & imm #2";
    ORI   = 19, "BBD; #0 ← #1 | imm #2";
    XORI  = 20, "BBD; #0 ← #1 ^ imm #2";
    SLI   = 21, "BBW; #0 ← #1 « imm #2";
    SRI   = 22, "BBW; #0 ← #1 » imm #2";
    SRSI  = 23, "BBW; #0 ← #1 » imm #2 (signed)";
    CMPI  = 24, "BBD; #0 ← #1 <=> imm #2";
    CMPUI = 25, "BBD; #0 ← #1 <=> imm #2 (unsigned)";

    CP  = 26, "BB; Copy #0 ← #1";
    SWA = 27, "BB; Swap #0 and #1";
    LI  = 28, "BD; #0 ← imm #1";
    LD  = 29, "BBDB; #0 ← [#1 + imm #3], imm #4 bytes, overflowing";
    ST  = 30, "BBDB; [#1 + imm #3] ← #0, imm #4 bytes, overflowing";
    BMC = 31, "BBD; [#0] ← [#1], imm #2 bytes";
    BRC = 32, "BBB; #0 ← #1, imm #2 registers";

    JAL   = 33, "BD;  Copy PC to #0 and unconditional jump [#1 + imm #2]";
    JEQ   = 34, "BBD; if #0 = #1 → jump imm #2";
    JNE   = 35, "BBD; if #0 ≠ #1 → jump imm #2";
    JLT   = 36, "BBD; if #0 < #1 → jump imm #2";
    JGT   = 37, "BBD; if #0 > #1 → jump imm #2";
    JLTU  = 38, "BBD; if #0 < #1 → jump imm #2 (unsigned)";
    JGTU  = 39, "BBD; if #0 > #1 → jump imm #2 (unsigned)";
    ECALL = 40, "N; Issue system call";

    ADDF = 41, "BBB; #0 ← #1 +. #2";
    SUBF = 42, "BBB; #0 ← #1 -. #2";
    MULF = 43, "BBB; #0 ← #1 +. #2";
    DIRF = 44, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";
    FMAF = 45, "BBBB; #0 ← (#1 * #2) + #3";
    NEGF = 46, "BB; #0 ← -#1";
    ITF  = 47, "BB; #0 ← #1 as float";
    FTI  = 48, "BB; #0 ← #1 as int";

    ADDFI = 49, "BBD; #0 ← #1 +. imm #2";
    MULFI = 50, "BBD; #0 ← #1 *. imm #2";
});

#[repr(packed)]
pub struct ParamBBBB(pub u8, pub u8, pub u8, pub u8);

#[repr(packed)]
pub struct ParamBBB(pub u8, pub u8, pub u8);

#[repr(packed)]
pub struct ParamBBDH(pub u8, pub u8, pub u64, pub u16);

#[repr(packed)]
pub struct ParamBBD(pub u8, pub u8, pub u64);

#[repr(packed)]
pub struct ParamBBW(pub u8, pub u8, pub u32);

#[repr(packed)]
pub struct ParamBB(pub u8, pub u8);

#[repr(packed)]
pub struct ParamBD(pub u8, pub u64);

/// # Safety
/// Has to be valid to be decoded from bytecode.
pub unsafe trait OpParam {}
unsafe impl OpParam for ParamBBBB {}
unsafe impl OpParam for ParamBBB {}
unsafe impl OpParam for ParamBBDH {}
unsafe impl OpParam for ParamBBD {}
unsafe impl OpParam for ParamBBW {}
unsafe impl OpParam for ParamBB {}
unsafe impl OpParam for ParamBD {}
unsafe impl OpParam for u64 {}
unsafe impl OpParam for () {}
