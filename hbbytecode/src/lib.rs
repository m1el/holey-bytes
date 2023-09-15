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

#[allow(rustdoc::invalid_rust_codeblocks)]
/// Invoke macro with bytecode definition
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
///     - I: Immediate
///     - L: Memory load / store size (u16)
///     - Other types are identity-mapped
///
/// # BRC special-case
/// BRC's 3rd operand is plain byte, not a register. Encoding is the same, but for some cases it may matter.
///
/// Please, if you distinguish in your API between byte and register, special case this one.
///
/// Sorry for that :(
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
            d(p0: I)
                => [JMP],
            n()
                => [UN, TX, NOP, ECALL],
        );
    };
}

invoke_with_def!(gen_valider::gen_valider);

constmod!(pub opcode(u8) {
    //! Opcode constant module

    UN  = 0, "N; Raises a trap";
    TX  = 1, "N; Terminate execution";
    NOP = 2, "N; Do nothing";

    ADD  = 3,  "BBB; #0 ← #1 + #2";
    SUB  = 4,  "BBB; #0 ← #1 - #2";
    MUL  = 5,  "BBB; #0 ← #1 × #2";
    AND  = 6,  "BBB; #0 ← #1 & #2";
    OR   = 7,  "BBB; #0 ← #1 | #2";
    XOR  = 8,  "BBB; #0 ← #1 ^ #2";
    SL   = 9,  "BBB; #0 ← #1 « #2";
    SR   = 10,  "BBB; #0 ← #1 » #2";
    SRS  = 11,  "BBB; #0 ← #1 » #2 (signed)";
    CMP  = 12, "BBB; #0 ← #1 <=> #2";
    CMPU = 13, "BBB; #0 ← #1 <=> #2 (unsigned)";
    DIR  = 14, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";
    NEG  = 15, "BB; #0 ← -#1";
    NOT  = 16, "BB; #0 ← !#1";

    ADDI  = 17, "BBD; #0 ← #1 + imm #2";
    MULI  = 18, "BBD; #0 ← #1 × imm #2";
    ANDI  = 19, "BBD; #0 ← #1 & imm #2";
    ORI   = 20, "BBD; #0 ← #1 | imm #2";
    XORI  = 21, "BBD; #0 ← #1 ^ imm #2";
    SLI   = 22, "BBW; #0 ← #1 « imm #2";
    SRI   = 23, "BBW; #0 ← #1 » imm #2";
    SRSI  = 24, "BBW; #0 ← #1 » imm #2 (signed)";
    CMPI  = 25, "BBD; #0 ← #1 <=> imm #2";
    CMPUI = 26, "BBD; #0 ← #1 <=> imm #2 (unsigned)";

    CP  = 27, "BB; Copy #0 ← #1";
    SWA = 28, "BB; Swap #0 and #1";
    LI  = 29, "BD; #0 ← imm #1";
    LD  = 30, "BBDB; #0 ← [#1 + imm #3], imm #4 bytes, overflowing";
    ST  = 31, "BBDB; [#1 + imm #3] ← #0, imm #4 bytes, overflowing";
    BMC = 32, "BBD; [#0] ← [#1], imm #2 bytes";
    BRC = 33, "BBB; #0 ← #1, imm #2 registers";

    JMP   = 34, "D; Unconditional, non-linking absolute jump";
    JAL   = 35, "BD;  Copy PC to #0 and unconditional jump [#1 + imm #2]";
    JEQ   = 36, "BBD; if #0 = #1 → jump imm #2";
    JNE   = 37, "BBD; if #0 ≠ #1 → jump imm #2";
    JLT   = 38, "BBD; if #0 < #1 → jump imm #2";
    JGT   = 39, "BBD; if #0 > #1 → jump imm #2";
    JLTU  = 40, "BBD; if #0 < #1 → jump imm #2 (unsigned)";
    JGTU  = 41, "BBD; if #0 > #1 → jump imm #2 (unsigned)";
    ECALL = 42, "N; Issue system call";

    ADDF = 43, "BBB; #0 ← #1 +. #2";
    SUBF = 44, "BBB; #0 ← #1 -. #2";
    MULF = 45, "BBB; #0 ← #1 +. #2";
    DIRF = 46, "BBBB; #0 ← #2 / #3, #1 ← #2 % #3";
    FMAF = 47, "BBBB; #0 ← (#1 * #2) + #3";
    NEGF = 48, "BB; #0 ← -#1";
    ITF  = 49, "BB; #0 ← #1 as float";
    FTI  = 50, "BB; #0 ← #1 as int";

    ADDFI = 51, "BBD; #0 ← #1 +. imm #2";
    MULFI = 52, "BBD; #0 ← #1 *. imm #2";
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
pub unsafe trait ProgramVal {}
unsafe impl ProgramVal for ParamBBBB {}
unsafe impl ProgramVal for ParamBBB {}
unsafe impl ProgramVal for ParamBBDH {}
unsafe impl ProgramVal for ParamBBD {}
unsafe impl ProgramVal for ParamBBW {}
unsafe impl ProgramVal for ParamBB {}
unsafe impl ProgramVal for ParamBD {}
unsafe impl ProgramVal for u64 {}
unsafe impl ProgramVal for u8 {} // Opcode
unsafe impl ProgramVal for () {}
