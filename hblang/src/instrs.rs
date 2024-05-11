pub const MAX_SIZE: usize = 13;
/// Cause an unreachable code trap
pub fn un() -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(N(0x00, )) }
}
/// Termiante execution
pub fn tx() -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(N(0x01, )) }
}
/// Do nothing
pub fn nop() -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(N(0x02, )) }
}
/// Addition (8b)
pub fn add8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x03, reg0, reg1, reg2)) }
}
/// Addition (16b)
pub fn add16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x04, reg0, reg1, reg2)) }
}
/// Addition (32b)
pub fn add32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x05, reg0, reg1, reg2)) }
}
/// Addition (64b)
pub fn add64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x06, reg0, reg1, reg2)) }
}
/// Subtraction (8b)
pub fn sub8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x07, reg0, reg1, reg2)) }
}
/// Subtraction (16b)
pub fn sub16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x08, reg0, reg1, reg2)) }
}
/// Subtraction (32b)
pub fn sub32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x09, reg0, reg1, reg2)) }
}
/// Subtraction (64b)
pub fn sub64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0A, reg0, reg1, reg2)) }
}
/// Multiplication (8b)
pub fn mul8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0B, reg0, reg1, reg2)) }
}
/// Multiplication (16b)
pub fn mul16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0C, reg0, reg1, reg2)) }
}
/// Multiplication (32b)
pub fn mul32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0D, reg0, reg1, reg2)) }
}
/// Multiplication (64b)
pub fn mul64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0E, reg0, reg1, reg2)) }
}
/// Bitand
pub fn and(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x0F, reg0, reg1, reg2)) }
}
/// Bitor
pub fn or(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x10, reg0, reg1, reg2)) }
}
/// Bitxor
pub fn xor(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x11, reg0, reg1, reg2)) }
}
/// Unsigned left bitshift (8b)
pub fn slu8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x12, reg0, reg1, reg2)) }
}
/// Unsigned left bitshift (16b)
pub fn slu16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x13, reg0, reg1, reg2)) }
}
/// Unsigned left bitshift (32b)
pub fn slu32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x14, reg0, reg1, reg2)) }
}
/// Unsigned left bitshift (64b)
pub fn slu64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x15, reg0, reg1, reg2)) }
}
/// Unsigned right bitshift (8b)
pub fn sru8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x16, reg0, reg1, reg2)) }
}
/// Unsigned right bitshift (16b)
pub fn sru16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x17, reg0, reg1, reg2)) }
}
/// Unsigned right bitshift (32b)
pub fn sru32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x18, reg0, reg1, reg2)) }
}
/// Unsigned right bitshift (64b)
pub fn sru64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x19, reg0, reg1, reg2)) }
}
/// Signed right bitshift (8b)
pub fn srs8(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1A, reg0, reg1, reg2)) }
}
/// Signed right bitshift (16b)
pub fn srs16(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1B, reg0, reg1, reg2)) }
}
/// Signed right bitshift (32b)
pub fn srs32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1C, reg0, reg1, reg2)) }
}
/// Signed right bitshift (64b)
pub fn srs64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1D, reg0, reg1, reg2)) }
}
/// Unsigned comparsion
pub fn cmpu(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1E, reg0, reg1, reg2)) }
}
/// Signed comparsion
pub fn cmps(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x1F, reg0, reg1, reg2)) }
}
/// Merged divide-remainder (unsigned 8b)
pub fn diru8(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x20, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (unsigned 16b)
pub fn diru16(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x21, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (unsigned 32b)
pub fn diru32(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x22, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (unsigned 64b)
pub fn diru64(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x23, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (signed 8b)
pub fn dirs8(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x24, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (signed 16b)
pub fn dirs16(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x25, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (signed 32b)
pub fn dirs32(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x26, reg0, reg1, reg2, reg3)) }
}
/// Merged divide-remainder (signed 64b)
pub fn dirs64(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x27, reg0, reg1, reg2, reg3)) }
}
/// Bit negation
pub fn neg(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x28, reg0, reg1)) }
}
/// Logical negation
pub fn not(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x29, reg0, reg1)) }
}
/// Sign extend 8b to 64b
pub fn sxt8(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x2A, reg0, reg1)) }
}
/// Sign extend 16b to 64b
pub fn sxt16(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x2B, reg0, reg1)) }
}
/// Sign extend 32b to 64b
pub fn sxt32(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x2C, reg0, reg1)) }
}
/// Addition with immediate (8b)
pub fn addi8(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x2D, reg0, reg1, imm2)) }
}
/// Addition with immediate (16b)
pub fn addi16(reg0: u8, reg1: u8, imm2: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRH(0x2E, reg0, reg1, imm2)) }
}
/// Addition with immediate (32b)
pub fn addi32(reg0: u8, reg1: u8, imm2: u32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRW(0x2F, reg0, reg1, imm2)) }
}
/// Addition with immediate (64b)
pub fn addi64(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x30, reg0, reg1, imm2)) }
}
/// Multiplication with immediate (8b)
pub fn muli8(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x31, reg0, reg1, imm2)) }
}
/// Multiplication with immediate (16b)
pub fn muli16(reg0: u8, reg1: u8, imm2: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRH(0x32, reg0, reg1, imm2)) }
}
/// Multiplication with immediate (32b)
pub fn muli32(reg0: u8, reg1: u8, imm2: u32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRW(0x33, reg0, reg1, imm2)) }
}
/// Multiplication with immediate (64b)
pub fn muli64(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x34, reg0, reg1, imm2)) }
}
/// Bitand with immediate
pub fn andi(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x35, reg0, reg1, imm2)) }
}
/// Bitor with immediate
pub fn ori(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x36, reg0, reg1, imm2)) }
}
/// Bitxor with immediate
pub fn xori(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x37, reg0, reg1, imm2)) }
}
/// Unsigned left bitshift with immedidate (8b)
pub fn slui8(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x38, reg0, reg1, imm2)) }
}
/// Unsigned left bitshift with immedidate (16b)
pub fn slui16(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x39, reg0, reg1, imm2)) }
}
/// Unsigned left bitshift with immedidate (32b)
pub fn slui32(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3A, reg0, reg1, imm2)) }
}
/// Unsigned left bitshift with immedidate (64b)
pub fn slui64(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3B, reg0, reg1, imm2)) }
}
/// Unsigned right bitshift with immediate (8b)
pub fn srui8(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3C, reg0, reg1, imm2)) }
}
/// Unsigned right bitshift with immediate (16b)
pub fn srui16(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3D, reg0, reg1, imm2)) }
}
/// Unsigned right bitshift with immediate (32b)
pub fn srui32(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3E, reg0, reg1, imm2)) }
}
/// Unsigned right bitshift with immediate (64b)
pub fn srui64(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x3F, reg0, reg1, imm2)) }
}
/// Signed right bitshift with immediate
pub fn srsi8(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x40, reg0, reg1, imm2)) }
}
/// Signed right bitshift with immediate
pub fn srsi16(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x41, reg0, reg1, imm2)) }
}
/// Signed right bitshift with immediate
pub fn srsi32(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x42, reg0, reg1, imm2)) }
}
/// Signed right bitshift with immediate
pub fn srsi64(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x43, reg0, reg1, imm2)) }
}
/// Unsigned compare with immediate
pub fn cmpui(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x44, reg0, reg1, imm2)) }
}
/// Signed compare with immediate
pub fn cmpsi(reg0: u8, reg1: u8, imm2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRD(0x45, reg0, reg1, imm2)) }
}
/// Copy register
pub fn cp(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x46, reg0, reg1)) }
}
/// Swap registers
pub fn swa(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x47, reg0, reg1)) }
}
/// Load immediate (8b)
pub fn li8(reg0: u8, imm1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RB(0x48, reg0, imm1)) }
}
/// Load immediate (16b)
pub fn li16(reg0: u8, imm1: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RH(0x49, reg0, imm1)) }
}
/// Load immediate (32b)
pub fn li32(reg0: u8, imm1: u32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RW(0x4A, reg0, imm1)) }
}
/// Load immediate (64b)
pub fn li64(reg0: u8, imm1: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RD(0x4B, reg0, imm1)) }
}
/// Load relative address
pub fn lra(reg0: u8, reg1: u8, offset2: i32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRO(0x4C, reg0, reg1, offset2)) }
}
/// Load from absolute address
pub fn ld(reg0: u8, reg1: u8, addr2: u64, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRAH(0x4D, reg0, reg1, addr2, imm3)) }
}
/// Store to absolute address
pub fn st(reg0: u8, reg1: u8, addr2: u64, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRAH(0x4E, reg0, reg1, addr2, imm3)) }
}
/// Load from relative address
pub fn ldr(reg0: u8, reg1: u8, offset2: i32, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RROH(0x4F, reg0, reg1, offset2, imm3)) }
}
/// Store to relative address
pub fn str(reg0: u8, reg1: u8, offset2: i32, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RROH(0x50, reg0, reg1, offset2, imm3)) }
}
/// Copy block of memory
pub fn bmc(reg0: u8, reg1: u8, imm2: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRH(0x51, reg0, reg1, imm2)) }
}
/// Copy register block
pub fn brc(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x52, reg0, reg1, imm2)) }
}
/// Relative jump
pub fn jmp(offset0: i32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(O(0x53, offset0)) }
}
/// Linking relative jump
pub fn jal(reg0: u8, reg1: u8, offset2: i32) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRO(0x54, reg0, reg1, offset2)) }
}
/// Linking absolute jump
pub fn jala(reg0: u8, reg1: u8, addr2: u64) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRA(0x55, reg0, reg1, addr2)) }
}
/// Branch on equal
pub fn jeq(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x56, reg0, reg1, offset2)) }
}
/// Branch on nonequal
pub fn jne(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x57, reg0, reg1, offset2)) }
}
/// Branch on lesser-than (unsigned)
pub fn jltu(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x58, reg0, reg1, offset2)) }
}
/// Branch on greater-than (unsigned)
pub fn jgtu(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x59, reg0, reg1, offset2)) }
}
/// Branch on lesser-than (signed)
pub fn jlts(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x5A, reg0, reg1, offset2)) }
}
/// Branch on greater-than (signed)
pub fn jgts(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x5B, reg0, reg1, offset2)) }
}
/// Environment call trap
pub fn eca() -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(N(0x5C, )) }
}
/// Environment breakpoint
pub fn ebp() -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(N(0x5D, )) }
}
/// Floating point addition (32b)
pub fn fadd32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x5E, reg0, reg1, reg2)) }
}
/// Floating point addition (64b)
pub fn fadd64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x5F, reg0, reg1, reg2)) }
}
/// Floating point subtraction (32b)
pub fn fsub32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x60, reg0, reg1, reg2)) }
}
/// Floating point subtraction (64b)
pub fn fsub64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x61, reg0, reg1, reg2)) }
}
/// Floating point multiply (32b)
pub fn fmul32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x62, reg0, reg1, reg2)) }
}
/// Floating point multiply (64b)
pub fn fmul64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x63, reg0, reg1, reg2)) }
}
/// Floating point division (32b)
pub fn fdiv32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x64, reg0, reg1, reg2)) }
}
/// Floating point division (64b)
pub fn fdiv64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x65, reg0, reg1, reg2)) }
}
/// Float fused multiply-add (32b)
pub fn fma32(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x66, reg0, reg1, reg2, reg3)) }
}
/// Float fused multiply-add (64b)
pub fn fma64(reg0: u8, reg1: u8, reg2: u8, reg3: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRRR(0x67, reg0, reg1, reg2, reg3)) }
}
/// Float reciprocal (32b)
pub fn finv32(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x68, reg0, reg1)) }
}
/// Float reciprocal (64b)
pub fn finv64(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x69, reg0, reg1)) }
}
/// Flaot compare less than (32b)
pub fn fcmplt32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x6A, reg0, reg1, reg2)) }
}
/// Flaot compare less than (64b)
pub fn fcmplt64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x6B, reg0, reg1, reg2)) }
}
/// Flaot compare greater than (32b)
pub fn fcmpgt32(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x6C, reg0, reg1, reg2)) }
}
/// Flaot compare greater than (64b)
pub fn fcmpgt64(reg0: u8, reg1: u8, reg2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRR(0x6D, reg0, reg1, reg2)) }
}
/// Int to 32 bit float
pub fn itf32(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x6E, reg0, reg1)) }
}
/// Int to 64 bit float
pub fn itf64(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x6F, reg0, reg1)) }
}
/// Float 32 to int
pub fn fti32(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x70, reg0, reg1, imm2)) }
}
/// Float 64 to int
pub fn fti64(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x71, reg0, reg1, imm2)) }
}
/// Float 64 to Float 32
pub fn fc32t64(reg0: u8, reg1: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RR(0x72, reg0, reg1)) }
}
/// Float 32 to Float 64
pub fn fc64t32(reg0: u8, reg1: u8, imm2: u8) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRB(0x73, reg0, reg1, imm2)) }
}
/// Load relative immediate (16 bit)
pub fn lra16(reg0: u8, reg1: u8, offset2: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRP(0x74, reg0, reg1, offset2)) }
}
/// Load from relative address (16 bit)
pub fn ldr16(reg0: u8, reg1: u8, offset2: i16, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRPH(0x75, reg0, reg1, offset2, imm3)) }
}
/// Store to relative address (16 bit)
pub fn str16(reg0: u8, reg1: u8, offset2: i16, imm3: u16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(RRPH(0x76, reg0, reg1, offset2, imm3)) }
}
/// Relative jump (16 bit)
pub fn jmp16(offset0: i16) -> (usize, [u8; MAX_SIZE]) {
    unsafe { crate::encode(P(0x77, offset0)) }
}
#[repr(packed)] pub struct N(u8, );
#[repr(packed)] pub struct RRR(u8, u8, u8, u8);
#[repr(packed)] pub struct RRRR(u8, u8, u8, u8, u8);
#[repr(packed)] pub struct RR(u8, u8, u8);
#[repr(packed)] pub struct RRB(u8, u8, u8, u8);
#[repr(packed)] pub struct RRH(u8, u8, u8, u16);
#[repr(packed)] pub struct RRW(u8, u8, u8, u32);
#[repr(packed)] pub struct RRD(u8, u8, u8, u64);
#[repr(packed)] pub struct RB(u8, u8, u8);
#[repr(packed)] pub struct RH(u8, u8, u16);
#[repr(packed)] pub struct RW(u8, u8, u32);
#[repr(packed)] pub struct RD(u8, u8, u64);
#[repr(packed)] pub struct RRO(u8, u8, u8, i32);
#[repr(packed)] pub struct RRAH(u8, u8, u8, u64, u16);
#[repr(packed)] pub struct RROH(u8, u8, u8, i32, u16);
#[repr(packed)] pub struct O(u8, i32);
#[repr(packed)] pub struct RRA(u8, u8, u8, u64);
#[repr(packed)] pub struct RRP(u8, u8, u8, i16);
#[repr(packed)] pub struct RRPH(u8, u8, u8, i16, u16);
#[repr(packed)] pub struct P(u8, i16);
pub const NAMES: [&str; 120] = [
    "un",
    "tx",
    "nop",
    "add8",
    "add16",
    "add32",
    "add64",
    "sub8",
    "sub16",
    "sub32",
    "sub64",
    "mul8",
    "mul16",
    "mul32",
    "mul64",
    "and",
    "or",
    "xor",
    "slu8",
    "slu16",
    "slu32",
    "slu64",
    "sru8",
    "sru16",
    "sru32",
    "sru64",
    "srs8",
    "srs16",
    "srs32",
    "srs64",
    "cmpu",
    "cmps",
    "diru8",
    "diru16",
    "diru32",
    "diru64",
    "dirs8",
    "dirs16",
    "dirs32",
    "dirs64",
    "neg",
    "not",
    "sxt8",
    "sxt16",
    "sxt32",
    "addi8",
    "addi16",
    "addi32",
    "addi64",
    "muli8",
    "muli16",
    "muli32",
    "muli64",
    "andi",
    "ori",
    "xori",
    "slui8",
    "slui16",
    "slui32",
    "slui64",
    "srui8",
    "srui16",
    "srui32",
    "srui64",
    "srsi8",
    "srsi16",
    "srsi32",
    "srsi64",
    "cmpui",
    "cmpsi",
    "cp",
    "swa",
    "li8",
    "li16",
    "li32",
    "li64",
    "lra",
    "ld",
    "st",
    "ldr",
    "str",
    "bmc",
    "brc",
    "jmp",
    "jal",
    "jala",
    "jeq",
    "jne",
    "jltu",
    "jgtu",
    "jlts",
    "jgts",
    "eca",
    "ebp",
    "fadd32",
    "fadd64",
    "fsub32",
    "fsub64",
    "fmul32",
    "fmul64",
    "fdiv32",
    "fdiv64",
    "fma32",
    "fma64",
    "finv32",
    "finv64",
    "fcmplt32",
    "fcmplt64",
    "fcmpgt32",
    "fcmpgt64",
    "itf32",
    "itf64",
    "fti32",
    "fti64",
    "fc32t64",
    "fc64t32",
    "lra16",
    "ldr16",
    "str16",
    "jmp16",
];
