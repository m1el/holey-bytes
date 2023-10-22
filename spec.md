# HoleyBytes ISA Specification

# Bytecode format
- Image format is not specified, though ELF is recommended
- All numbers are encoded little-endian
- There is 256 registers, they are represented by a byte
- Immediate values are 8, 16, 32 or 64 bit

## Instruction encoding
- Instruction operands are packed (no alignment)
- [opcode, operand 0, operand 1, …]

## Instruction parameter types
- `R`: Register (8 bits)
- Relative program-counter offset immediates:
    - `O`: 32 bit (Si32)
    - `P`: 16 bit (Si16)
- Immediates:
    - `B`: Byte, 8 bit (Xi8)
    - `H`: Half-word, 16 bit (Xi16)
    - `W`: Word, 32 bit (Xi32)
    - `D`: Double-word, 64 bit (Xi64)
- `A`: Absolute address immediate, 64 bit (Ui64)

## Types
- Si*n*: Signed integer of size *n* bits (Si8, Si16, Si32, Si64)
- Ui*n*: Unsigned integer of size *n* bits (Ui8, Ui16, Ui32, Ui64)
- Xi*n*: Sign-agnostic integer of size *n* bits (Xi8, Xi16, Xi32, Xi64)
- Fl*n*: Floating point number of size *n* bits (Fl32, Fl64)

# Behaviour
- Integer operations are wrapping, including signed numbers
    - Bitshifts are truncating
- Two's complement
- Floats as specified by IEEE 754
- Execution model is implementation defined as long all observable
  effects are performed in correct order

## Relative addressing
Relative addresses are computed from address of the first byte
of offset in the code. Not from the beginning of current or following instruction.

## Zero register
- Register 0
- Cannot be clobbered
    - Write is no-op
- Load always yields 0

## Rounding modes
| Rounding mode            | Value |
|:-------------------------|:------|
| To nearest, ties to even | 0b00  |
| Towards 0 (truncate)     | 0b01  |
| Towards +∞ (up)          | 0b10  |
| Towards -∞ (down)        | 0b11  |

- Remaining values in the byte traps with invalid operand exception

# Memory
- Memory implementation is implementation-defined
- Zero address (`0x0`) is considered invalid

# Traps
- Environment call
- Environment breakpoint

Program counter goes to the following instruction

## Exceptions
- Memory access fault
- Invalid operand
- Unknown opcode

Program counter stays on the currently executed instruction

# Instructions
- `#n`: register in parameter *n*
- `$n`: for immediate in parameter *n*
- `#P ← V`: Set register P to value V
- `[x]`: Address x
- `XY`: X bytes from location Y
- `pc`: Program counter
- `<XYZ>`: Placeholder
- `Type(X)`: Cast

## Program execution control
- Type `N`

| Opcode | Mnemonic | Action                                      |
|:-------|:---------|:--------------------------------------------|
| 0x00   | UN       | Throw unreachable code exception            |
| 0x01   | TX       | Terminate execution (eg. on end of program) |
| 0x02   | NOP      | Do nothing                                  |

## Binary register-immediate ops
- Type `RR<IMM>`
- Action: `#0 ← #1 <OP> #2`

## Addition (`+`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x03   | ADD8     | Xi8  |
| 0x04   | ADD16    | Xi16 |
| 0x05   | ADD32    | Xi32 |
| 0x06   | ADD64    | Xi64 |

## Subtraction (`-`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x07   | SUB8     | Xi8  |
| 0x08   | SUB16    | Xi16 |
| 0x09   | SUB32    | Xi32 |
| 0x0A   | SUB64    | Xi64 |

## Multiplication (`*`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x0B   | MUL8     | Xi8  |
| 0x0C   | MUL16    | Xi16 |
| 0x0D   | MUL32    | Xi32 |
| 0x0E   | MUL64    | Xi64 |

## Bitwise ops (type: Xi64)
| Opcode | Mnemonic | Operation           |
|:-------|:---------|:--------------------|
| 0x0F   | AND      | Conjunction (&)     |
| 0x10   | OR       | Disjunction (\|)    |
| 0x11   | XOR      | Non-equivalence (^) |

## Unsigned left bitshift (`<<`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x12   | SLU8     | Ui8  |
| 0x13   | SLU16    | Ui16 |
| 0x14   | SLU32    | Ui32 |
| 0x15   | SLU64    | Ui64 |

## Unsigned right bitshift (`>>`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x16   | SRU8     | Ui8  |
| 0x17   | SRU16    | Ui16 |
| 0x18   | SRU32    | Ui32 |
| 0x19   | SRU64    | Ui64 |

## Signed right bitshift (`>>`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x1A   | SRS8     | Si8  |
| 0x1B   | SRS16    | Si16 |
| 0x1C   | SRS32    | Si32 |
| 0x1D   | SRS64    | Si64 |

## Comparsion
- Compares two numbers, saves result to register
- Operation: `#0 ← #1 <=> #2`

| Ordering | Number |
|:---------|:-------|
| <        | -1     |
| =        | 0      |
| >        | 1      |

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x1E   | CMPU     | Ui64 |
| 0x1F   | CMPS     | Si64 |

# Merged divide-remainder
- Type `RRRR`
- Operation:
    - `#0 ← #2 / #3`
    - `#1 ← #2 % #3`

- If dividing by zero:
    - `#0 ← Ui64(-1)`
    - `#1 ← #2`

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x20   | DIRU8    | Ui8  |
| 0x21   | DIRU16   | Ui16 |
| 0x22   | DIRU32   | Ui32 |
| 0x23   | DIRU64   | Ui64 |
| 0x24   | DIRS8    | Si8  |
| 0x25   | DIRS16   | Si16 |
| 0x26   | DIRS32   | Si32 |
| 0x27   | DIRS64   | Si64 |

# Unary register operations (type: Xi64)
- Type: `RR`
- Operation: `#0 ← <OP> #1`

| Opcode | Mnemonic | Operation                |
|:-------|:---------|:-------------------------|
| 0x28   | NEG      | Bitwise complement (`~`) |
| 0x29   | NOT      | Logical negation (`!`)   |

## Sign extensions
- Operation: `#0 ← Si64(#1)`

| Opcode | Mnemonic | Source type |
|:-------|:---------|:------------|
| 0x2A   | SXT8     | Si8         |
| 0x2B   | SXT16    | Si16        |
| 0x2C   | SXT32    | Si32        |

# Binary register-immediate operations
- Type: `RR<IMM>`
- Operation: `#0 ← #1 <OP> $2`

## Addition (`+`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x2D   | ADDI8    | Xi8  |
| 0x2E   | ADDI16   | Xi16 |
| 0x2F   | ADDI32   | Xi32 |
| 0x30   | ADDI64   | Xi64 |

## Multiplication (`*`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x31   | MULI8    | Xi8  |
| 0x32   | MULI16   | Xi16 |
| 0x33   | MULI32   | Xi32 |
| 0x34   | MULI64   | Xi64 |

## Bitwise ops (type: Xi64)
| Opcode | Mnemonic | Operation           |
|:-------|:---------|:--------------------|
| 0x35   | ANDI     | Conjunction (&)     |
| 0x36   | ORI      | Disjunction (\|)    |
| 0x37   | XORI     | Non-equivalence (^) |

# Register-immediate bitshifts
- Type: `RRB`
- Operation: `#0 ← #1 <OP> $2`

## Unsigned left bitshift (`<<`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x38   | SLUI8    | Ui8  |
| 0x39   | SLUI16   | Ui16 |
| 0x3A   | SLUI32   | Ui32 |
| 0x3B   | SLUI64   | Ui64 |

## Unsigned right bitshift (`>>`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x3C   | SRUI8    | Ui8  |
| 0x3D   | SRUI16   | Ui16 |
| 0x3E   | SRUI32   | Ui32 |
| 0x3F   | SRUI64   | Ui64 |

## Signed right bitshift (`>>`)
| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x40   | SRSI8    | Si8  |
| 0x41   | SRSI16   | Si16 |
| 0x42   | SRSI32   | Si32 |
| 0x43   | SRSI64   | Si64 |

## Comparsion
- Compares two numbers, saves result to register
- Operation: `#0 ← #1 <=> $2`
- Comparsion table same for register-register one

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x44   | CMPUI    | Ui64 |
| 0x45   | CMPSI    | Si64 |

# Register copies
- Type: `RR`

| Opcode | Mnemonic | Operation                        |
|:-------|:---------|:---------------------------------|
| 0x46   | CP       | Copy register value (`#0 ← #1`)  |
| 0x47   | SWA      | Swap register values (`#0 ⇆ #1`) |

# Load immediate
- Load immediate value from code to register
- Type: `R<IMM>`
- Operation: `#0 ← $1`

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x48   | LI8      | Xi8  |
| 0x49   | LI16     | Xi16 |
| 0x4A   | Li32     | Xi32 |
| 0x4B   | Li64     | Xi64 |

# Load relative address
- Compute value from program counter, register value and offset
- Type: `RRO`
- Operation: `#0 ← pc + #1 + $2`

| Opcode | Mnemonic |
|:-------|:---------|
| 0x4C   | LRA      |

# Memory access operations
- Immediate `$3` specifies size
- If size is greater than register size,
    it overflows to adjecent register
    (eg. copying 16 bytes to register `r1` copies first 8 bytes to it
         and the remaining to `r2`)

## Absolute addressing
- Type: `RRAH`
- Computes address from base register and absolute offset

| Opcode | Mnemonic | Operation          |
|:-------|:---------|:-------------------|
| 0x4D   | LD       | `#0 ← $3[#1 + $2]` |
| 0x4E   | ST       | `$3[#1 + $2] ← #0` |

## Relative addressing
- Type: `RROH`
- Computes address from register and offset from program counter

| Opcode | Mnemonic | Operation               |
|:-------|:---------|:------------------------|
| 0x4F   | LDR      | `#0 ← $3[pc + #1 + $2]` |
| 0x50   | STR      | `$3[pc + #1 + $2] ← #0` |

# Block memory copy
- Type: `RRH`
- Copies block of `$3` bytes from memory location on address on `#0` to `#1`

| Opcode | Mnemonic | Operation         |
|:-------|:---------|:------------------|
| 0x51   | BMC      | `$3[#1] ← $3[x0]` |

# Block register copy
- Type: `RRB`
- Copy block of `$3` registers starting with `#0` to `#1`
- Copying over the 256 registers causes an exception

| Opcode | Mnemonic | Operation     |
|:-------|:---------|:--------------|
| 0x52   | BRC      | `$3#1 ← $3#0` |

# Relative jump
- Type: `O`

| Opcode | Mnemonic | Operation      |
|:-------|:---------|:---------------|
| 0x53   | JMP      | `pc ← pc + $0` |

# Linking jump
- Operation:
    - Save address of following instruction to `#0`
        - `#0 ← pc+<instruction size>`
    - Jump to specified address 

| Opcode | Mnemonic | Instruction type  | Address                  |
|:-------|:---------|:------------------|:-------------------------|
| 0x54   | JAL      | RRO (size = 6 B)  | Relative, `pc + #1 + $2` |
| 0x55   | JALA     | RRA (size = 10 B) | Absolute, `#1 + $2`      |

# Conditional jump
- Perform comparsion, if operation met, jump to relative address
- Type: `RRP`
- Operation: `if #0 <CMP> #1 { pc ← pc + $2 }`

| Opcode | Mnemonic | Condition          | Type |
|:-------|:---------|:-------------------|:-----|
| 0x56   | JEQ      | Equals (`=`)       | Xi64 |
| 0x57   | JNE      | Not-equals (`≠`)   | Xi64 |
| 0x58   | JLTU     | Less-than (`<`)    | Ui64 |
| 0x59   | JGTU     | Greater-than (`>`) | Ui64 |
| 0x5A   | JLTS     | Less-than (`<`)    | Si64 |
| 0x5B   | JGTS     | Greater-than (`>`) | Si64 |

# Environment traps
- Traps to the environment
- Type: `N`

| Opcode | Mnemonic | Trap type        |
|:-------|:---------|:-----------------|
| 0x5C   | ECA      | Environment call |
| 0x5D   | EBP      | Breakpoint       |

# Floating point binary operations
- Type: `RRR`
- Operation: `#0 ← #1 <OP> #2`

| Opcode | Mnemonic | Operation            | Type |
|:-------|:---------|:---------------------|:-----|
| 0x5E   | FADD32   | Addition (`+`)       | Fl32 |
| 0x5F   | FADD64   | Addition (`+`)       | Fl64 |
| 0x60   | FSUB32   | Subtraction (`-`)    | Fl32 |
| 0x61   | FSUB64   | Subtraction (`-`)    | Fl64 |
| 0x62   | FMUL32   | Multiplication (`*`) | Fl32 |
| 0x63   | FMUL64   | Multiplication (`*`) | Fl64 |
| 0x64   | FDIV32   | Division (`/`)       | Fl32 |
| 0x65   | FDIV64   | Division (`/`)       | Fl64 |

# Fused multiply-add
- Type: `RRRR`
- Operation: `#0 ← (#1 * #2) + #3`

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x66   | FMA32    | Fl32 |
| 0x67   | FMA64    | Fl64 |

# Comparsions
- Type: `RRR`
- Operation: `#0 ← #1 <=> #2`
- Comparsion table same as for `CMPx`/`CMPxI`
- NaN is less-than/greater-than depends on variant

| Opcode | Mnemonic | Type | NaN is |
|:-------|:---------|:-----|:-------|
| 0x6A   | FCMPLT32 | Fl32 | <      |
| 0x6B   | FCMPLT64 | Fl64 | <      |
| 0x6C   | FCMPGT32 | Fl32 | >      |
| 0x6D   | FCMPGT64 | Fl64 | >      |

# Int to float
- Type: `RR`
- Converts from `Si64`
- Operation: `#0 ← Fl<SIZE>(#1)`

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x6E   | ITF32    | Fl32 |
| 0x6F   | ITF64    | Fl64 |

# Float to int
- Type: `RRB`
- Operation: `#0 ← Si64(#1)`
- Immediate `$2` specifies rounding mode

| Opcode | Mnemonic | Type |
|:-------|:---------|:-----|
| 0x70   | FTI32    | Fl32 |
| 0x71   | FTI64    | Fl64 |

# Fl32 to Fl64
- Type: `RR`
- Operation: `#0 ← Fl64(#1)`

| Opcode | Mnemonic |
|:-------|:---------|
| 0x72   | FC32T64  |

# Fl64 to Fl32
- Type: `RRB`
- Operation: `#0 ← Fl32(#1)`
- Immediate `$2` specified rounding mode

| Opcode | Mnemonic |
|:-------|:---------|
| 0x73   | FC64T32  |

# 16-bit relative address instruction variants

| Opcode | Mnemonic | Type | Variant of |
|:-------|:---------|:-----|:-----------|
| 0x74   | LRA16    | RRP  | LRA        |
| 0x75   | LDR16    | RRPH | LDR        |
| 0x76   | STR16    | RRPH | STR        |
| 0x77   | JMP16    | P    | JMP        |
