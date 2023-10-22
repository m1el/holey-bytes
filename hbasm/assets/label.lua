label "label" -- set named label
local a = label {} -- unassigned label

jmp16("label")
a:here() -- assign label
jmp16(a)

addi8(r3, r4, 5)