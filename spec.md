# HoleyBytes ISA Specification

# Bytecode format
- Holey Bytes program should start with following magic: `[0xAB, 0x1E, 0x0B]`
- All numbers are encoded little-endian
- There is 256 registers, they are represented by a byte
- Immediate values are 64 bit
- Program is by spec required to be terminated with 12 zero bytes

### Instruction encoding
- Instruction parameters are packed (no alignment)
- [opcode, …parameters…]

### Instruction parameter types
- B = Byte
- D = Doubleword (64 bits)
- H = Halfword (16 bits)

| Name | Size    |
|:----:|:--------|
| BBBB | 32 bits |
| BBB  | 24 bits |
| BBDH | 96 bits |
| BBD  | 80 bits |
| BBW  | 48 bits |
|  BB  | 16 bits |
|  BD  | 72 bits |
|  D   | 64 bits |
|  N   | 0  bits |

# Instructions
- `#n`: register in parameter *n*
- `imm #n`: for immediate in parameter *n*
- `P ← V`: Set register P to value V
- `[x]`: Address x

## Program execution control
- N type

| Opcode | Name |            Action             |
|:------:|:----:|:-----------------------------:|
|   0    |  UN  | Trigger unreachable code trap |
|   1    |  TX  |      Terminate execution      |
|   2    | NOP  |          Do nothing           |

## Integer binary ops.
- BBB type
- `#0 ← #1 <op> #2`

| Opcode | Name |         Action          |
|:------:|:----:|:-----------------------:|
|   3    | ADD  |    Wrapping addition    |
|   4    | SUB  |  Wrapping subtraction   |
|   5    | MUL  | Wrapping multiplication |
|   6    | AND  |         Bitand          |
|   7    |  OR  |          Bitor          |
|   8    | XOR  |         Bitxor          |
|   9    |  SL  | Unsigned left bitshift  |
|   10   |  SR  | Unsigned right bitshift |
|   11   | SRS  |  Signed right bitshift  |

### Comparsion
| Opcode | Name |       Action        |
|:------:|:----:|:-------------------:|
|   12   | CMP  |  Signed comparsion  |
|   13   | CMPU | Unsigned comparsion |

#### Comparsion table
| #1 *op* #2 | Result |
|:----------:|:------:|
|     <      |   0    |
|     =      |   1    |
|     >      |   2    |

### Division-remainder
- Type BBBB
- In case of `#3` is zero, the resulting value is all-ones
- `#0 ← #2 ÷ #3`
- `#1 ← #2 % #3`

| Opcode | Name |             Action              |
|:------:|:----:|:-------------------------------:|
|   14   | DIR  | Divide and remainder combinated |

### Negations
- Type BB
- `#0 ← #1 <op> #2`

| Opcode | Name |      Action      |
|:------:|:----:|:----------------:|
|   15   | NEG  |   Bit negation   |
|   16   | NOT  | Logical negation |

## Integer immediate binary ops.
- Type BBD
- `#0 ← #1 <op> imm #2`

| Opcode | Name |        Action        |
|:------:|:----:|:--------------------:|
|   17   | ADDI |  Wrapping addition   |
|   18   | MULI | Wrapping subtraction |
|   19   | ANDI |        Bitand        |
|   20   | ORI  |        Bitor         |
|   21   | XORI |        Bitxor        |

### Bitshifts
- Type BBW
| Opcode | Name |         Action          |
|:------:|:----:|:-----------------------:|
|   22   | SLI  | Unsigned left bitshift  |
|   23   | SRI  | Unsigned right bitshift |
|   24   | SRSI |  Signed right bitshift  |

### Comparsion
- Comparsion is the same as when RRR type

| Opcode | Name  |       Action        |
|:------:|:-----:|:-------------------:|
|   25   | CMPI  |  Signed comparsion  |
|   26   | CMPUI | Unsigned comparsion |

## Register value set / copy

### Copy
- Type BB
- `#0 ← #1`

| Opcode | Name | Action |
|:------:|:----:|:------:|
|   27   |  CP  |  Copy  |

### Swap
- Type BB
- Swap #0 and #1
- Zero register rules:
    - Both: no-op
    - One: Copy zero to the non-zero register

| Opcode | Name | Action |
|:------:|:----:|:------:|
|   28   | SWA  |  Swap  |

### Load immediate
- Type BD
- `#0 ← #1`

| Opcode | Name |     Action     |
|:------:|:----:|:--------------:|
|   29   |  LI  | Load immediate |

## Memory operations
- Type BBDH
- If loaded/store value exceeds one register size, continue accessing following registers

### Load / Store
| Opcode | Name |                 Action                  |
|:------:|:----:|:---------------------------------------:|
|   30   |  LD  | `#0 ← [#1 + imm #3], copy imm #4 bytes` |
|   31   |  ST  | `[#1 + imm #3] ← #0, copy imm #4 bytes` |

## Block copy
- Block copy source and target can overlap

### Memory copy
- Type BBD

| Opcode | Name |              Action              |
|:------:|:----:|:--------------------------------:|
|   32   | BMC  | `[#1] ← [#0], copy imm #2 bytes` |

### Register copy
- Type BBB
- Copy a block a register to another location (again, overflowing to following registers)

| Opcode | Name |              Action              |
|:------:|:----:|:--------------------------------:|
|   33   | BRC  | `#1 ← #0, copy imm #2 registers` |

## Control flow

### Unconditional jump
- Type D
| Opcode | Name |             Action              |
|:------:|:----:|:-------------------------------:|
|   34   | JMP  | Unconditional, non-linking jump |

### Unconditional linking jump
- Type BBD

| Opcode | Name |                       Action                       |
|:------:|:----:|:--------------------------------------------------:|
|   35   | JAL  | Save PC past JAL to `#0` and jump at `#1 + imm #2` |

### Conditional jumps
- Type BBD
- Jump at `imm #2` if `#0 <op> #1`

| Opcode | Name |  Comparsion  |
|:------:|:----:|:------------:|
|   36   | JEQ  |      =       |
|   37   | JNE  |      ≠       |
|   38   | JLT  |  < (signed)  |
|   39   | JGT  |  > (signed)  |
|   40   | JLTU | < (unsigned) |
|   41   | JGTU | > (unsigned) |

### Environment call
- Type N

| Opcode | Name  |                Action                 |
|:------:|:-----:|:-------------------------------------:|
|   42   | ECALL | Cause an trap to the host environment |

## Floating point operations
- Type BBB
- `#0 ← #1 <op> #2`

| Opcode | Name |     Action     |
|:------:|:----:|:--------------:|
|   43   | ADDF |    Addition    |
|   44   | SUBF |  Subtraction   |
|   45   | MULF | Multiplication |

### Division-remainder
- Type BBBB

| Opcode | Name |          Action           |
|:------:|:----:|:-------------------------:|
|   46   | DIRF | Same as for integer `DIR` |

### Fused Multiply-Add
- Type BBBB

| Opcode | Name |        Action         |
|:------:|:----:|:---------------------:|
|   47   | FMAF | `#0 ← (#1 * #2) + #3` |

### Negation
- Type BB
| Opcode | Name |   Action   |
|:------:|:----:|:----------:|
|   48   | NEGF | `#0 ← -#1` |

### Conversion
- Type BB
- Signed
- `#0 ← #1 as _`

| Opcode | Name |    Action    |
|:------:|:----:|:------------:|
|   49   | ITF  | Int to Float |
|   50   | FTI  | Float to Int |

## Floating point immediate operations
- Type BBD
- `#0 ← #1 <op> imm #2`

| Opcode | Name  |     Action     |
|:------:|:-----:|:--------------:|
|   51   | ADDFI |    Addition    |
|   52   | MULFI | Multiplication |

# Registers
- There is 255 registers + one zero register (with index 0)
- Reading from zero register yields zero
- Writing to zero register is a no-op

# Memory
- Addresses are 64 bit
- Program should be in the same address space as all other data
- Memory implementation is arbitrary
    - Address `0x0` may or may not be valid. Count with compilers
      considering it invalid!
- In case of accessing invalid address:
    - Program shall trap (LoadAccessEx, StoreAccessEx) with parameter of accessed address
    - Value of register when trapped is undefined

## Recommendations
- If paging used:
    - Leave first page invalid
    - Pages should be at least 4 KiB

# Program execution
- The way of program execution is implementation defined
- The execution is arbitrary, as long all effects are obervable
    in the way as program was executed literally, in order.

# Program validation
- Invalid program should cause runtime error:
    - The form of error is arbitrary. Can be a trap or an interpreter-specified error
    - It shall not be handleable from within the program
- Executing invalid opcode should trap
- Program can be validaded either before execution or when executing

# Traps
Program should at least implement these traps:
- Environment call
- Invalid instruction exception
- Load address exception
- Store address exception
- Unreachable instruction

and executing environment should be able to get information about them,
like the opcode of invalid instruction or attempted address to load/store.
Details about these are left as an implementation detail.

# Assembly
HoleyBytes assembly format is not defined, this is just a weak description
of `hbasm` syntax.

- Opcode names correspond to specified opcode names, lowercase (`nop`)
- Parameters are separated by comma (`addi r0, r0, 1`)
- Instructions are separated by either line feed or semicolon
- Registers are represented by `r` followed by the number (`r10`)
- Labels are defined by label name followed with colon (`loop:`)
- Labels are references simply by their name (`print`)
- Immediates are entered plainly. Negative numbers supported.