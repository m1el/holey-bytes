let hello = data::str("Hello, world!");

li8     (r1, 1);            // Write syscall
li8     (r2, 1);            // Stdout FD
lra16   (r3, r0, hello);    // String buffer
li8     (r4, hello.len);    // String length
eca     ();                 // System call

tx      ();                 // End program