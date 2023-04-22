load 0 a0        ;; 05 00 A0
load 10 a1       ;; 05 10 A1
add a0 1 a0      ;; 01 A0 01 A0
jump_neq a0 a1 0 ;; a1 A0 A1 0