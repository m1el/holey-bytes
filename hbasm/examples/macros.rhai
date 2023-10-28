li8(r1, 69);
li8(r2,  0);

if_eq(r1, r2,
    || puts("Equals!"),
    || puts("Not equals!"),
);


tx(); // END OF MAIN

/// Inline function – write text to stdout
fn puts(string) {
    let d = data::str(string);
    li8   (r1, 1);     // Write syscall
    li8   (r2, 1);     // Stdout handle
    lra16 (r3, r0, d);
    li64  (r4, d.len);
    eca   ();
}

fn if_eq(a, b, thenblk, elseblk) {
    let elselbl = declabel();
    let endlbl  = declabel();

    jne(a, b, elselbl);
    thenblk.call();
    jmp16(endlbl);

    elselbl.here();
    elseblk.call();
    endlbl.here();
}