mod utils;

use {
    crate::utils::IterExt,
    std::{
        fs::File,
        io::{self, BufRead, BufReader, BufWriter, Seek, Write},
        path::Path,
    },
};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn build_cmd(cmd: impl AsRef<str>) -> std::process::Command {
    let mut args = cmd.as_ref().split_whitespace();
    let mut c = std::process::Command::new(args.next().unwrap());
    for arg in args {
        c.arg(arg);
    }
    c
}

fn exec(mut cmd: std::process::Command) -> io::Result<()> {
    if !cmd.status()?.success() {
        return Err(io::Error::other(format!("command failed: {:?}", cmd)));
    }
    Ok(())
}

fn build_wasm_blob(name: &str, debug: bool) -> io::Result<()> {
    let mut c = build_cmd(if debug { "cargo wasm-build-debug" } else { "cargo wasm-build" });
    c.arg(format!("wasm-{name}"));
    exec(c)?;

    let out_path = format!("target/wasm32-unknown-unknown/small/wasm_{name}.wasm");
    if !debug {
        exec(build_cmd(format!("wasm-opt -Oz {out_path} -o {out_path}")))?;
    }
    exec(build_cmd(format!("cp {out_path} depell/src/wasm-{name}.wasm")))
}

fn main() -> io::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args[0].as_str() {
        "fmt" => fmt(args[1] == "-r" || args[1] == "--renumber"),
        "build-depell" => {
            build_wasm_blob("hbfmt", false)?;
            build_wasm_blob("hbc", false)?;
            exec(build_cmd("cargo build -p depell --release"))?;
            Ok(())
        }
        "build-depell-debug" => {
            build_wasm_blob("hbfmt", true)?;
            build_wasm_blob("hbc", true)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

pub fn fmt(renumber: bool) -> io::Result<()> {
    let mut file = File::options()
        .read(true)
        .write(true)
        .open(crate::root().join("bytecode/instructions.in"))?;

    // Extract records
    let reader = BufReader::new(&file);
    let mut recs = vec![];
    let mut lens = [0_usize; 4];

    for rec in reader.split(b';').filter_map(|r| {
        r.map(|ln| {
            let s = String::from_utf8_lossy(&ln);
            let s = s.trim_matches('\n');
            if s.is_empty() {
                return None;
            }

            s.split(',').map(|s| Box::<str>::from(s.trim())).collect_array::<4>().map(Ok::<_, ()>)
        })
        .transpose()
    }) {
        let rec = rec?.expect("Valid record format");
        for (current, next) in lens.iter_mut().zip(rec.iter()) {
            *current = (*current).max(next.len());
        }

        recs.push(rec);
    }

    // Clear file!
    file.set_len(0)?;
    file.seek(std::io::SeekFrom::Start(0))?;

    let mut writer = BufWriter::new(file);

    let ord_opco_len = digit_count(recs.len()) as usize;
    for (n, rec) in recs.iter().enumerate() {
        // Write opcode number
        if renumber {
            let n = format!("{n:#04X}");
            write!(writer, "{n}, {}", padding(ord_opco_len, &n))?;
        } else {
            write!(writer, "{}, {}", rec[0], padding(lens[0], &rec[0]))?;
        }

        // Write other fields
        writeln!(
            writer,
            "{}, {}{},{} {}{};",
            rec[1],
            padding(lens[1], &rec[1]),
            rec[2],
            padding(lens[2], &rec[2]),
            rec[3],
            padding(lens[3], &rec[3]),
        )?;
    }

    Ok(())
}

fn padding(req: usize, s: &str) -> Box<str> {
    " ".repeat(req.saturating_sub(s.len())).into()
}

#[inline]
fn digit_count(n: usize) -> u32 {
    n.checked_ilog10().unwrap_or(0) + 1
}
