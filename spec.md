# HoleyBytes ISA Specification

# Bytecode format
- All numbers are encoded little-endian
- There is 256 registers, they are represented by a byte
- Immediate values are 64 bit

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
|  BB  | 16 bits |
|  BD  | 72 bits |
|  D   | 64 bits |
|  N   | 0  bits |

# Instructions
- `#n`: register in parameter *n*
- `imm #n`: for immediate in parameter *n*
- `P ← V`: Set register P to value V
- `[x]`: Address x

## No-op
- N type

| Opcode | Name |   Action   |
|:------:|:----:|:----------:|
|   0    | NOP  | Do nothing |

## Integer binary ops.
- BBB type
- `#0 ← #1 <op> #2`

| Opcode | Name |         Action          |
|:------:|:----:|:-----------------------:|
|   1    | ADD  |    Wrapping addition    |
|   2    | SUB  |  Wrapping subtraction   |
|   3    | MUL  | Wrapping multiplication |
|   4    | AND  |         Bitand          |
|   5    |  OR  |          Bitor          |
|   6    | XOR  |         Bitxor          |
|   7    |  SL  | Unsigned left bitshift  |
|   8    |  SR  | Unsigned right bitshift |
|   9    | SRS  |  Signed right bitshift  |

### Comparsion
| Opcode | Name |       Action        |
|:------:|:----:|:-------------------:|
|   10   | CMP  |  Signed comparsion  |
|   11   | CMPU | Unsigned comparsion |

#### Comparsion table
| #1 *op* #2 | Result |
|:----------:|:------:|
|     <      |   -1   |
|     =      |   0    |
|     >      |   1    |

### Division-remainder
- Type BBBB
- In case of `#3` is zero, the resulting value is all-ones
- `#0 ← #2 ÷ #3`
- `#1 ← #2 % #3`

| Opcode | Name |             Action              |
|:------:|:----:|:-------------------------------:|
|   12   | DIR  | Divide and remainder combinated |

### Negations
- Type BB
- `#0 ← #1 <op> #2`

| Opcode | Name |      Action      |
|:------:|:----:|:----------------:|
|   13   | NEG  |   Bit negation   |
|   14   | NOT  | Logical negation |

## Integer immediate binary ops.
- Type BBD
- `#0 ← #1 <op> imm #2`

| Opcode | Name |         Action          |
|:------:|:----:|:-----------------------:|
|   15   | ADDI |    Wrapping addition    |
|   16   | MULI |  Wrapping subtraction   |
|   17   | ANDI |         Bitand          |
|   18   | ORI  |          Bitor          |
|   19   | XORI |         Bitxor          |
|   20   | SLI  | Unsigned left bitshift  |
|   21   | SRI  | Unsigned right bitshift |
|   22   | SRSI |  Signed right bitshift  |

### Comparsion
- Comparsion is the same as when RRR type

| Opcode | Name  |       Action        |
|:------:|:-----:|:-------------------:|
|   23   | CMPI  |  Signed comparsion  |
|   24   | CMPUI | Unsigned comparsion |

## Register value set / copy

### Copy
- Type BB
- `#0 ← #1`

| Opcode | Name | Action |
|:------:|:----:|:------:|
|   25   |  CP  |  Copy  |

### Swap
- Type BB
- Swap #0 and #1

| Opcode | Name | Action |
|:------:|:----:|:------:|
|   26   | SWA  |  Swap  |

### Load immediate
- Type BD
- `#0 ← #1`

| Opcode | Name |     Action     |
|:------:|:----:|:--------------:|
|   27   |  LI  | Load immediate |

## Memory operations
- Type BBDH
- If loaded/store value exceeds one register size, continue accessing following registers

### Load / Store
| Opcode | Name |                 Action                  |
|:------:|:----:|:---------------------------------------:|
|   28   |  LD  | `#0 ← [#1 + imm #3], copy imm #4 bytes` |
|   29   |  ST  | `[#1 + imm #3] ← #0, copy imm #4 bytes` |

## Block copy
- Block copy source and target can overlap

### Memory copy
- Type BBD

| Opcode | Name |              Action              |
|:------:|:----:|:--------------------------------:|
|   30   | BMC  | `[#0] ← [#1], copy imm #2 bytes` |

### Register copy
- Type BBB
- Copy a block a register to another location (again, overflowing to following registers)

| Opcode | Name |              Action              |
|:------:|:----:|:--------------------------------:|
|   31   | BRC  | `#0 ← #1, copy imm #2 registers` |

## Control flow

### Unconditional jump
- Type BD

| Opcode | Name |        Action         |
|:------:|:----:|:---------------------:|
|   32   | JMP  | Jump at `#0 + imm #1` |

### Conditional jumps
- Type BBD
- Jump at `imm #2` if `#0 <op> #1`

| Opcode | Name |  Comparsion  |
|:------:|:----:|:------------:|
|   33   | JEQ  |      =       |
|   34   | JNE  |      ≠       |
|   35   | JLT  |  < (signed)  |
|   36   | JGT  |  > (signed)  |
|   37   | JLTU | < (unsigned) |
|   38   | JGTU | > (unsigned) |

### Environment call
- Type N

| Opcode | Name  |                Action                 |
|:------:|:-----:|:-------------------------------------:|
|   39   | ECALL | Cause an trap to the host environment |

## Floating point operations
- Type BBB
- `#0 ← #1 <op> #2`

| Opcode | Name |     Action     |
|:------:|:----:|:--------------:|
|   40   | ADDF |    Addition    |
|   41   | SUBF |  Subtraction   |
|   42   | MULF | Multiplication |

### Division-remainder
- Type BBBB

| Opcode | Name |          Action           |
|:------:|:----:|:-------------------------:|
|   43   | DIRF | Same as for integer `DIR` |

### Fused Multiply-Add
- Type BBBB

| Opcode | Name |        Action         |
|:------:|:----:|:---------------------:|
|   44   | FMAF | `#0 ← (#1 * #2) + #3` |

### Negation
- Type BB
| Opcode | Name |   Action   |
|:------:|:----:|:----------:|
|   45   | NEGF | `#0 ← -#1` |

### Conversion
- Type BB
- Signed
- `#0 ← #1 as _`

| Opcode | Name |    Action    |
|:------:|:----:|:------------:|
|   46   | ITF  | Int to Float |
|   47   | FTI  | Float to Int |

## Floating point immediate operations
- Type BBD
- `#0 ← #1 <op> imm #2`

| Opcode | Name  |     Action     |
|:------:|:-----:|:--------------:|
|   48   | ADDFI |    Addition    |
|   49   | MULFI | Multiplication |

# Registers
- There is 255 registers + one zero register (with index 0)
- Reading from zero register yields zero
- Writing to zero register is a no-op

# Memory
- Addresses are 64 bit
- Memory implementation is arbitrary
- In case of accessing invalid address:
    - Program shall trap (LoadAccessEx, StoreAccessEx) with parameter of accessed address
    - Value of register when trapped is undefined

## Recommendations
- Leave address `0x0` as invalid
- If paging used:
    - Leave first page invalid
    - Pages should be at least 4 KiB

# Program execution
- The way of program execution is implementation defined
- The order of instruction is arbitrary, as long all observable
    effects are applied in the program's order

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