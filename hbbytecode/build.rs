#![feature(iter_next_chunk)]

use std::{collections::HashSet, fmt::Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=instructions.in");

    let mut generated = String::new();
    gen_instrs(&mut generated)?;
    std::fs::write("src/instrs.rs", generated)?;

    Ok(())
}

fn gen_instrs(generated: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(generated, "#![allow(dead_code)] #![allow(clippy::upper_case_acronyms)]")?;
    writeln!(generated, "use crate::*;")?;

    '_opcode_structs: {
        let mut seen = HashSet::new();
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
    }

    '_max_size: {
        let max = instructions()
            .map(
                |[_, _, ty, _]| {
                    if ty == "N" {
                        1
                    } else {
                        iter_args(ty).map(arg_to_width).sum::<usize>() + 1
                    }
                },
            )
            .max()
            .unwrap();

        writeln!(generated, "pub const MAX_SIZE: usize = {max};")?;
    }

    '_encoders: {
        for [op, name, ty, doc] in instructions() {
            writeln!(generated, "/// {}", doc.trim_matches('"'))?;
            let name = name.to_lowercase();
            let args = comma_sep(
                iter_args(ty)
                    .enumerate()
                    .map(|(i, c)| format!("{}{i}: {}", arg_to_name(c), arg_to_type(c))),
            );
            writeln!(generated, "pub fn {name}({args}) -> (usize, [u8; MAX_SIZE]) {{")?;
            let arg_names =
                comma_sep(iter_args(ty).enumerate().map(|(i, c)| format!("{}{i}", arg_to_name(c))));
            writeln!(generated, "    unsafe {{ crate::encode({ty}({op}, {arg_names})) }}")?;
            writeln!(generated, "}}")?;
        }
    }

    '_structs: {
        let mut seen = std::collections::HashSet::new();
        for [_, _, ty, _] in instructions() {
            if !seen.insert(ty) {
                continue;
            }
            let types = comma_sep(iter_args(ty).map(arg_to_type).map(|s| s.to_string()));
            writeln!(generated, "#[repr(packed)] pub struct {ty}(u8, {types});")?;
        }
    }

    '_name_list: {
        writeln!(generated, "pub const NAMES: [&str; {}] = [", instructions().count())?;
        for [_, name, _, _] in instructions() {
            writeln!(generated, "    \"{}\",", name.to_lowercase())?;
        }
        writeln!(generated, "];")?;
    }

    let instr = "Instr";
    let oper = "Oper";

    '_instr_enum: {
        writeln!(generated, "#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(u8)]")?;
        writeln!(generated, "pub enum {instr} {{")?;
        for [id, name, ..] in instructions() {
            writeln!(generated, "    {name} = {id},")?;
        }
        writeln!(generated, "}}")?;
    }

    '_arg_kind: {
        writeln!(generated, "#[derive(Debug, Clone, Copy, PartialEq, Eq)]")?;
        writeln!(generated, "pub enum {oper} {{")?;
        let mut seen = HashSet::new();
        for ty in instructions().flat_map(|[.., ty, _]| iter_args(ty)) {
            if !seen.insert(ty) {
                continue;
            }
            writeln!(generated, "    {ty}({}),", arg_to_type(ty))?;
        }
        writeln!(generated, "}}")?;
    }

    '_parse_opers: {
        writeln!(
            generated,
            "/// This assumes the instruction byte is still at the beginning of the buffer"
        )?;
        writeln!(generated, "#[cfg(feature = \"disasm\")]")?;
        writeln!(generated, "pub fn parse_args(bytes: &mut &[u8], kind: {instr}, buf: &mut std::vec::Vec<{oper}>) -> Option<()> {{")?;
        writeln!(generated, "    match kind {{")?;
        let mut instrs = instructions().collect::<Vec<_>>();
        instrs.sort_unstable_by_key(|&[.., ty, _]| ty);
        for group in instrs.chunk_by(|[.., a, _], [.., b, _]| a == b) {
            let ty = group[0][2];
            for &[_, name, ..] in group {
                writeln!(generated, "        | {instr}::{name}")?;
            }
            generated.pop();
            writeln!(generated, " => {{")?;
            if iter_args(ty).count() != 0 {
                writeln!(generated, "            let data = crate::decode::<{ty}>(bytes)?;")?;
                writeln!(
                    generated,
                    "            buf.extend([{}]);",
                    comma_sep(
                        iter_args(ty).zip(1u32..).map(|(t, i)| format!("{oper}::{t}(data.{i})"))
                    )
                )?;
            } else {
                writeln!(generated, "            crate::decode::<{ty}>(bytes)?;")?;
            }

            writeln!(generated, "        }}")?;
        }
        writeln!(generated, "    }}")?;
        writeln!(generated, "    Some(())")?;
        writeln!(generated, "}}")?;
    }

    std::fs::write("src/instrs.rs", generated)?;
    Ok(())
}

fn comma_sep(items: impl Iterator<Item = String>) -> String {
    items.map(|item| item.to_string()).collect::<Vec<_>>().join(", ")
}

fn instructions() -> impl Iterator<Item = [&'static str; 4]> {
    include_str!("../hbbytecode/instructions.in")
        .lines()
        .filter_map(|line| line.strip_suffix(';'))
        .map(|line| line.splitn(4, ',').map(str::trim).next_chunk().unwrap())
}

fn arg_to_type(arg: char) -> &'static str {
    match arg {
        'R' | 'B' => "u8",
        'H' => "u16",
        'W' => "u32",
        'D' | 'A' => "u64",
        'P' => "i16",
        'O' => "i32",
        _ => panic!("unknown type: {}", arg),
    }
}

fn arg_to_width(arg: char) -> usize {
    match arg {
        'R' | 'B' => 1,
        'H' => 2,
        'W' => 4,
        'D' | 'A' => 8,
        'P' => 2,
        'O' => 4,
        _ => panic!("unknown type: {}", arg),
    }
}

fn arg_to_name(arg: char) -> &'static str {
    match arg {
        'R' => "reg",
        'B' | 'H' | 'W' | 'D' => "imm",
        'P' | 'O' => "offset",
        'A' => "addr",
        _ => panic!("unknown type: {}", arg),
    }
}

fn iter_args(ty: &'static str) -> impl Iterator<Item = char> {
    ty.chars().filter(|c| *c != 'N')
}
