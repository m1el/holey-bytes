#![feature(iter_next_chunk)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../hbbytecode/instructions.in");

    let instructions = include_str!("../hbbytecode/instructions.in");

    let mut generated = String::new();
    use std::fmt::Write;

    writeln!(&mut generated, "impl crate::codegen::Func {{")?;

    for line in instructions.lines() {
        let line = line.strip_suffix(";").unwrap();
        let [opcode, name, ty, doc] = line.splitn(4, ',').map(str::trim).next_chunk().unwrap();

        writeln!(&mut generated, "/// {}", doc.trim_matches('"'))?;
        write!(&mut generated, "pub fn {}(&mut self", name.to_lowercase())?;
        for (i, c) in ty.chars().enumerate() {
            let (name, ty) = match c {
                'N' => continue,
                'R' => ("reg", "u8"),
                'B' => ("imm", "u8"),
                'H' => ("imm", "u16"),
                'W' => ("imm", "u32"),
                'D' => ("imm", "u64"),
                'P' => ("offset", "u32"),
                'O' => ("offset", "u32"),
                'A' => ("addr", "u64"),
                _ => panic!("unknown type: {}", c),
            };
            write!(&mut generated, ", {name}{i}: {ty}")?;
        }
        writeln!(&mut generated, ") {{")?;

        let mut offset = 1;
        for (i, c) in ty.chars().enumerate() {
            let width = match c {
                'N' => 0,
                'R' => 1,
                'B' => 1,
                'H' => 2,
                'W' => 4,
                'D' => 8,
                'A' => 8,
                'P' => 2,
                'O' => 4,
                _ => panic!("unknown type: {}", c),
            };

            if matches!(c, 'P' | 'O') {
                writeln!(
                    &mut generated,
                    "    self.offset(offset{i}, {offset}, {width});",
                )?;
            }

            offset += width;
        }

        write!(
            &mut generated,
            "    self.extend(crate::as_bytes(&crate::Args({opcode}"
        )?;
        for (i, c) in ty.chars().enumerate() {
            let name = match c {
                'N' => continue,
                'R' => "reg",
                'B' | 'H' | 'W' | 'D' => "imm",
                'P' => "0u16",
                'O' => "0u32",
                'A' => "addr",
                _ => panic!("unknown type: {}", c),
            };

            if matches!(c, 'P' | 'O') {
                write!(&mut generated, ", {name}")?;
            } else {
                write!(&mut generated, ", {name}{i}")?;
            }
        }
        for _ in ty.len() - (ty == "N") as usize..4 {
            write!(&mut generated, ", ()")?;
        }
        writeln!(&mut generated, ")));")?;

        writeln!(&mut generated, "}}")?;
    }

    writeln!(&mut generated, "}}")?;

    std::fs::write("src/instrs.rs", generated)?;

    Ok(())
}
