Holey Bytes has two bit widths
8 bit to help with byte level manipulation and 64 bit numbers because nothing else should ever be used in a modern setting

this leaves us with an amount of registers that should be defined
I'd like to use a letter and a number to represent registers 
like `a0` or `d0` the first of which would be reserved for 8 bit numbers and the later of which 64 bit.

instructions
### NOP
`0`

### ADD_8 TYPE LHS RHS LOCATION
`1`
### SUB TYPE LHS RHS LOCATION
`2`
### MUL TYPE LHS RHS LOCATION
`3`
### MUL TYPE LHS RHS LOCATION
`4`
### DIV TYPE LHS RHS LOCATION
`5`

### JUMP ADDR
`100`
an unconditional jump to an address

### JUMP_EQ LHS RHS ADDR
`101`
A conditional jump 
if LHS is equal to RHS then jump to address
### JUMP_NEQ LHS RHS ADDR
`102`
A conditional jump 
if LHS is not equal to RHS then jump to address

### RET
pop off the callstack