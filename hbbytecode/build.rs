#![feature(iter_next_chunk)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=instructions.in");

    let mut generated = String::new();
    gen_op_structs(&mut generated)?;
    std::fs::write("src/ops.rs", generated)?;

    let mut generated = String::new();
    gen_op_codes(&mut generated)?;
    std::fs::write("src/opcode.rs", generated)?;

    Ok(())
}

fn gen_op_structs(generated: &mut String) -> std::fmt::Result {
    use std::fmt::Write;
    let mut seen = std::collections::HashSet::new();
    writeln!(generated, "use crate::*;")?;
    for [.., args, _] in instructions() {
        if !seen.insert(args) {
            continue;
        }

        writeln!(generated, "#[derive(Clone, Copy, Debug)]")?;
        writeln!(generated, "#[repr(packed)]")?;
        write!(generated, "pub struct Ops{args}(")?;
        let mut first = true;
        for ch in args.chars().filter(|&ch| ch != 'N') {
            if !std::mem::take(&mut first) {
                write!(generated, ",")?;
            }
            write!(generated, "pub Op{ch}")?;
        }
        writeln!(generated, ");")?;
        writeln!(generated, "unsafe impl BytecodeItem for Ops{args} {{}}")?;
    }

    Ok(())
}

fn gen_op_codes(generated: &mut String) -> std::fmt::Result {
    use std::fmt::Write;
    for [op, name, _, comment] in instructions() {
        writeln!(generated, "#[doc = {comment}]")?;
        writeln!(generated, "pub const {name}: u8 = {op};")?;
    }
    Ok(())
}

fn instructions() -> impl Iterator<Item = [&'static str; 4]> {
    include_str!("../hbbytecode/instructions.in")
        .lines()
        .map(|line| line.strip_suffix(';').unwrap())
        .map(|line| line.splitn(4, ',').map(str::trim).next_chunk().unwrap())
}
