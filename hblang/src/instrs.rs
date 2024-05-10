impl crate::codegen::Func {
/// Cause an unreachable code trap
pub fn un(&mut self) {
    self.extend(crate::as_bytes(&crate::Args(0x00, (), (), (), ())));
}
/// Termiante execution
pub fn tx(&mut self) {
    self.extend(crate::as_bytes(&crate::Args(0x01, (), (), (), ())));
}
/// Do nothing
pub fn nop(&mut self) {
    self.extend(crate::as_bytes(&crate::Args(0x02, (), (), (), ())));
}
/// Addition (8b)
pub fn add8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x03, reg0, reg1, reg2, ())));
}
/// Addition (16b)
pub fn add16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x04, reg0, reg1, reg2, ())));
}
/// Addition (32b)
pub fn add32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x05, reg0, reg1, reg2, ())));
}
/// Addition (64b)
pub fn add64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x06, reg0, reg1, reg2, ())));
}
/// Subtraction (8b)
pub fn sub8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x07, reg0, reg1, reg2, ())));
}
/// Subtraction (16b)
pub fn sub16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x08, reg0, reg1, reg2, ())));
}
/// Subtraction (32b)
pub fn sub32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x09, reg0, reg1, reg2, ())));
}
/// Subtraction (64b)
pub fn sub64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0A, reg0, reg1, reg2, ())));
}
/// Multiplication (8b)
pub fn mul8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0B, reg0, reg1, reg2, ())));
}
/// Multiplication (16b)
pub fn mul16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0C, reg0, reg1, reg2, ())));
}
/// Multiplication (32b)
pub fn mul32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0D, reg0, reg1, reg2, ())));
}
/// Multiplication (64b)
pub fn mul64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0E, reg0, reg1, reg2, ())));
}
/// Bitand
pub fn and(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x0F, reg0, reg1, reg2, ())));
}
/// Bitor
pub fn or(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x10, reg0, reg1, reg2, ())));
}
/// Bitxor
pub fn xor(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x11, reg0, reg1, reg2, ())));
}
/// Unsigned left bitshift (8b)
pub fn slu8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x12, reg0, reg1, reg2, ())));
}
/// Unsigned left bitshift (16b)
pub fn slu16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x13, reg0, reg1, reg2, ())));
}
/// Unsigned left bitshift (32b)
pub fn slu32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x14, reg0, reg1, reg2, ())));
}
/// Unsigned left bitshift (64b)
pub fn slu64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x15, reg0, reg1, reg2, ())));
}
/// Unsigned right bitshift (8b)
pub fn sru8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x16, reg0, reg1, reg2, ())));
}
/// Unsigned right bitshift (16b)
pub fn sru16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x17, reg0, reg1, reg2, ())));
}
/// Unsigned right bitshift (32b)
pub fn sru32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x18, reg0, reg1, reg2, ())));
}
/// Unsigned right bitshift (64b)
pub fn sru64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x19, reg0, reg1, reg2, ())));
}
/// Signed right bitshift (8b)
pub fn srs8(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1A, reg0, reg1, reg2, ())));
}
/// Signed right bitshift (16b)
pub fn srs16(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1B, reg0, reg1, reg2, ())));
}
/// Signed right bitshift (32b)
pub fn srs32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1C, reg0, reg1, reg2, ())));
}
/// Signed right bitshift (64b)
pub fn srs64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1D, reg0, reg1, reg2, ())));
}
/// Unsigned comparsion
pub fn cmpu(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1E, reg0, reg1, reg2, ())));
}
/// Signed comparsion
pub fn cmps(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x1F, reg0, reg1, reg2, ())));
}
/// Merged divide-remainder (unsigned 8b)
pub fn diru8(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x20, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (unsigned 16b)
pub fn diru16(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x21, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (unsigned 32b)
pub fn diru32(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x22, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (unsigned 64b)
pub fn diru64(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x23, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (signed 8b)
pub fn dirs8(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x24, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (signed 16b)
pub fn dirs16(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x25, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (signed 32b)
pub fn dirs32(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x26, reg0, reg1, reg2, reg3)));
}
/// Merged divide-remainder (signed 64b)
pub fn dirs64(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x27, reg0, reg1, reg2, reg3)));
}
/// Bit negation
pub fn neg(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x28, reg0, reg1, (), ())));
}
/// Logical negation
pub fn not(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x29, reg0, reg1, (), ())));
}
/// Sign extend 8b to 64b
pub fn sxt8(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x2A, reg0, reg1, (), ())));
}
/// Sign extend 16b to 64b
pub fn sxt16(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x2B, reg0, reg1, (), ())));
}
/// Sign extend 32b to 64b
pub fn sxt32(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x2C, reg0, reg1, (), ())));
}
/// Addition with immediate (8b)
pub fn addi8(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x2D, reg0, reg1, imm2, ())));
}
/// Addition with immediate (16b)
pub fn addi16(&mut self, reg0: u8, reg1: u8, imm2: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x2E, reg0, reg1, imm2, ())));
}
/// Addition with immediate (32b)
pub fn addi32(&mut self, reg0: u8, reg1: u8, imm2: u32) {
    self.extend(crate::as_bytes(&crate::Args(0x2F, reg0, reg1, imm2, ())));
}
/// Addition with immediate (64b)
pub fn addi64(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x30, reg0, reg1, imm2, ())));
}
/// Multiplication with immediate (8b)
pub fn muli8(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x31, reg0, reg1, imm2, ())));
}
/// Multiplication with immediate (16b)
pub fn muli16(&mut self, reg0: u8, reg1: u8, imm2: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x32, reg0, reg1, imm2, ())));
}
/// Multiplication with immediate (32b)
pub fn muli32(&mut self, reg0: u8, reg1: u8, imm2: u32) {
    self.extend(crate::as_bytes(&crate::Args(0x33, reg0, reg1, imm2, ())));
}
/// Multiplication with immediate (64b)
pub fn muli64(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x34, reg0, reg1, imm2, ())));
}
/// Bitand with immediate
pub fn andi(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x35, reg0, reg1, imm2, ())));
}
/// Bitor with immediate
pub fn ori(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x36, reg0, reg1, imm2, ())));
}
/// Bitxor with immediate
pub fn xori(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x37, reg0, reg1, imm2, ())));
}
/// Unsigned left bitshift with immedidate (8b)
pub fn slui8(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x38, reg0, reg1, imm2, ())));
}
/// Unsigned left bitshift with immedidate (16b)
pub fn slui16(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x39, reg0, reg1, imm2, ())));
}
/// Unsigned left bitshift with immedidate (32b)
pub fn slui32(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3A, reg0, reg1, imm2, ())));
}
/// Unsigned left bitshift with immedidate (64b)
pub fn slui64(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3B, reg0, reg1, imm2, ())));
}
/// Unsigned right bitshift with immediate (8b)
pub fn srui8(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3C, reg0, reg1, imm2, ())));
}
/// Unsigned right bitshift with immediate (16b)
pub fn srui16(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3D, reg0, reg1, imm2, ())));
}
/// Unsigned right bitshift with immediate (32b)
pub fn srui32(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3E, reg0, reg1, imm2, ())));
}
/// Unsigned right bitshift with immediate (64b)
pub fn srui64(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x3F, reg0, reg1, imm2, ())));
}
/// Signed right bitshift with immediate
pub fn srsi8(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x40, reg0, reg1, imm2, ())));
}
/// Signed right bitshift with immediate
pub fn srsi16(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x41, reg0, reg1, imm2, ())));
}
/// Signed right bitshift with immediate
pub fn srsi32(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x42, reg0, reg1, imm2, ())));
}
/// Signed right bitshift with immediate
pub fn srsi64(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x43, reg0, reg1, imm2, ())));
}
/// Unsigned compare with immediate
pub fn cmpui(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x44, reg0, reg1, imm2, ())));
}
/// Signed compare with immediate
pub fn cmpsi(&mut self, reg0: u8, reg1: u8, imm2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x45, reg0, reg1, imm2, ())));
}
/// Copy register
pub fn cp(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x46, reg0, reg1, (), ())));
}
/// Swap registers
pub fn swa(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x47, reg0, reg1, (), ())));
}
/// Load immediate (8b)
pub fn li8(&mut self, reg0: u8, imm1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x48, reg0, imm1, (), ())));
}
/// Load immediate (16b)
pub fn li16(&mut self, reg0: u8, imm1: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x49, reg0, imm1, (), ())));
}
/// Load immediate (32b)
pub fn li32(&mut self, reg0: u8, imm1: u32) {
    self.extend(crate::as_bytes(&crate::Args(0x4A, reg0, imm1, (), ())));
}
/// Load immediate (64b)
pub fn li64(&mut self, reg0: u8, imm1: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x4B, reg0, imm1, (), ())));
}
/// Load relative address
pub fn lra(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 4);
    self.extend(crate::as_bytes(&crate::Args(0x4C, reg0, reg1, 0u32, ())));
}
/// Load from absolute address
pub fn ld(&mut self, reg0: u8, reg1: u8, addr2: u64, imm3: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x4D, reg0, reg1, addr2, imm3)));
}
/// Store to absolute address
pub fn st(&mut self, reg0: u8, reg1: u8, addr2: u64, imm3: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x4E, reg0, reg1, addr2, imm3)));
}
/// Load from relative address
pub fn ldr(&mut self, reg0: u8, reg1: u8, offset2: u32, imm3: u16) {
    self.offset(offset2, 3, 4);
    self.extend(crate::as_bytes(&crate::Args(0x4F, reg0, reg1, 0u32, imm3)));
}
/// Store to relative address
pub fn str(&mut self, reg0: u8, reg1: u8, offset2: u32, imm3: u16) {
    self.offset(offset2, 3, 4);
    self.extend(crate::as_bytes(&crate::Args(0x50, reg0, reg1, 0u32, imm3)));
}
/// Copy block of memory
pub fn bmc(&mut self, reg0: u8, reg1: u8, imm2: u16) {
    self.extend(crate::as_bytes(&crate::Args(0x51, reg0, reg1, imm2, ())));
}
/// Copy register block
pub fn brc(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x52, reg0, reg1, imm2, ())));
}
/// Relative jump
pub fn jmp(&mut self, offset0: u32) {
    self.offset(offset0, 1, 4);
    self.extend(crate::as_bytes(&crate::Args(0x53, 0u32, (), (), ())));
}
/// Linking relative jump
pub fn jal(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 4);
    self.extend(crate::as_bytes(&crate::Args(0x54, reg0, reg1, 0u32, ())));
}
/// Linking absolute jump
pub fn jala(&mut self, reg0: u8, reg1: u8, addr2: u64) {
    self.extend(crate::as_bytes(&crate::Args(0x55, reg0, reg1, addr2, ())));
}
/// Branch on equal
pub fn jeq(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x56, reg0, reg1, 0u16, ())));
}
/// Branch on nonequal
pub fn jne(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x57, reg0, reg1, 0u16, ())));
}
/// Branch on lesser-than (unsigned)
pub fn jltu(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x58, reg0, reg1, 0u16, ())));
}
/// Branch on greater-than (unsigned)
pub fn jgtu(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x59, reg0, reg1, 0u16, ())));
}
/// Branch on lesser-than (signed)
pub fn jlts(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x5A, reg0, reg1, 0u16, ())));
}
/// Branch on greater-than (signed)
pub fn jgts(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x5B, reg0, reg1, 0u16, ())));
}
/// Environment call trap
pub fn eca(&mut self) {
    self.extend(crate::as_bytes(&crate::Args(0x5C, (), (), (), ())));
}
/// Environment breakpoint
pub fn ebp(&mut self) {
    self.extend(crate::as_bytes(&crate::Args(0x5D, (), (), (), ())));
}
/// Floating point addition (32b)
pub fn fadd32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x5E, reg0, reg1, reg2, ())));
}
/// Floating point addition (64b)
pub fn fadd64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x5F, reg0, reg1, reg2, ())));
}
/// Floating point subtraction (32b)
pub fn fsub32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x60, reg0, reg1, reg2, ())));
}
/// Floating point subtraction (64b)
pub fn fsub64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x61, reg0, reg1, reg2, ())));
}
/// Floating point multiply (32b)
pub fn fmul32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x62, reg0, reg1, reg2, ())));
}
/// Floating point multiply (64b)
pub fn fmul64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x63, reg0, reg1, reg2, ())));
}
/// Floating point division (32b)
pub fn fdiv32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x64, reg0, reg1, reg2, ())));
}
/// Floating point division (64b)
pub fn fdiv64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x65, reg0, reg1, reg2, ())));
}
/// Float fused multiply-add (32b)
pub fn fma32(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x66, reg0, reg1, reg2, reg3)));
}
/// Float fused multiply-add (64b)
pub fn fma64(&mut self, reg0: u8, reg1: u8, reg2: u8, reg3: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x67, reg0, reg1, reg2, reg3)));
}
/// Float reciprocal (32b)
pub fn finv32(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x68, reg0, reg1, (), ())));
}
/// Float reciprocal (64b)
pub fn finv64(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x69, reg0, reg1, (), ())));
}
/// Flaot compare less than (32b)
pub fn fcmplt32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6A, reg0, reg1, reg2, ())));
}
/// Flaot compare less than (64b)
pub fn fcmplt64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6B, reg0, reg1, reg2, ())));
}
/// Flaot compare greater than (32b)
pub fn fcmpgt32(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6C, reg0, reg1, reg2, ())));
}
/// Flaot compare greater than (64b)
pub fn fcmpgt64(&mut self, reg0: u8, reg1: u8, reg2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6D, reg0, reg1, reg2, ())));
}
/// Int to 32 bit float
pub fn itf32(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6E, reg0, reg1, (), ())));
}
/// Int to 64 bit float
pub fn itf64(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x6F, reg0, reg1, (), ())));
}
/// Float 32 to int
pub fn fti32(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x70, reg0, reg1, imm2, ())));
}
/// Float 64 to int
pub fn fti64(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x71, reg0, reg1, imm2, ())));
}
/// Float 64 to Float 32
pub fn fc32t64(&mut self, reg0: u8, reg1: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x72, reg0, reg1, (), ())));
}
/// Float 32 to Float 64
pub fn fc64t32(&mut self, reg0: u8, reg1: u8, imm2: u8) {
    self.extend(crate::as_bytes(&crate::Args(0x73, reg0, reg1, imm2, ())));
}
/// Load relative immediate (16 bit)
pub fn lra16(&mut self, reg0: u8, reg1: u8, offset2: u32) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x74, reg0, reg1, 0u16, ())));
}
/// Load from relative address (16 bit)
pub fn ldr16(&mut self, reg0: u8, reg1: u8, offset2: u32, imm3: u16) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x75, reg0, reg1, 0u16, imm3)));
}
/// Store to relative address (16 bit)
pub fn str16(&mut self, reg0: u8, reg1: u8, offset2: u32, imm3: u16) {
    self.offset(offset2, 3, 2);
    self.extend(crate::as_bytes(&crate::Args(0x76, reg0, reg1, 0u16, imm3)));
}
/// Relative jump (16 bit)
pub fn jmp16(&mut self, offset0: u32) {
    self.offset(offset0, 1, 2);
    self.extend(crate::as_bytes(&crate::Args(0x77, 0u16, (), (), ())));
}
}
