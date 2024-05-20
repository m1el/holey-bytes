#![feature(iter_next_chunk)]
use std::fmt::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../hbbytecode/instructions.in");

    let mut generated = String::new();

    writeln!(generated, "#![allow(dead_code)]")?;
    gen_max_size(&mut generated)?;
    gen_encodes(&mut generated)?;
    gen_structs(&mut generated)?;
    gen_name_list(&mut generated)?;

    std::fs::write("src/instrs.rs", generated)?;

    Ok(())
}

fn gen_name_list(generated: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(
        generated,
        "pub const NAMES: [&str; {}] = [",
        instructions().count()
    )?;
    for [_, name, _, _] in instructions() {
        writeln!(generated, "    \"{}\",", name.to_lowercase())?;
    }
    writeln!(generated, "];")?;

    Ok(())
}

fn gen_max_size(generated: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    let max = instructions()
        .map(|[_, _, ty, _]| {
            if ty == "N" {
                1
            } else {
                iter_args(ty).map(|(_, c)| arg_to_width(c)).sum::<usize>() + 1
            }
        })
        .max()
        .unwrap();

    writeln!(generated, "pub const MAX_SIZE: usize = {};", max)?;

    Ok(())
}

fn gen_encodes(generated: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    for [op, name, ty, doc] in instructions() {
        writeln!(generated, "/// {}", doc.trim_matches('"'))?;
        let name = name.to_lowercase();
        let args = comma_sep(
            iter_args(ty).map(|(i, c)| format!("{}{i}: {}", arg_to_name(c), arg_to_type(c))),
        );
        writeln!(
            generated,
            "pub fn {name}({args}) -> (usize, [u8; MAX_SIZE]) {{"
        )?;
        let arg_names = comma_sep(iter_args(ty).map(|(i, c)| format!("{}{i}", arg_to_name(c))));
        writeln!(
            generated,
            "    unsafe {{ crate::encode({ty}({op}, {arg_names})) }}"
        )?;
        writeln!(generated, "}}")?;
    }

    Ok(())
}

fn gen_structs(generated: &mut String) -> Result<(), Box<dyn std::error::Error>> {
    let mut seen = std::collections::HashSet::new();
    for [_, _, ty, _] in instructions() {
        if !seen.insert(ty) {
            continue;
        }
        let types = comma_sep(iter_args(ty).map(|(_, c)| arg_to_type(c).to_string()));
        writeln!(generated, "#[repr(packed)] pub struct {ty}(u8, {types});")?;
    }

    Ok(())
}

fn comma_sep(items: impl Iterator<Item = String>) -> String {
    items
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join(", ")
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

fn iter_args(ty: &'static str) -> impl Iterator<Item = (usize, char)> {
    ty.chars().enumerate().filter(|(_, c)| *c != 'N')
}
