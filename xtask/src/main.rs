mod utils;

use {
    crate::utils::IterExt,
    anyhow::Context,
    std::{
        fs::File,
        io::{self, BufRead, BufReader, BufWriter, Seek, Write},
        path::Path,
    },
    walrus::{ir::Value, ConstExpr, GlobalKind, ValType},
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

fn insert_stack_pointer_shim(file: impl AsRef<str>) -> anyhow::Result<()> {
    let mut module = walrus::Module::from_file(file.as_ref())?;

    let global = module
        .globals
        .iter()
        .find(|g| g.ty == ValType::I32 && g.mutable)
        .filter(|g| match g.kind {
            GlobalKind::Local(ConstExpr::Value(Value::I32(n))) => n != 0,
            _ => false,
        })
        .context("binary is missing a stak pointer")?;

    module.exports.add("stack_pointer", global.id());

    module.emit_wasm_file(file.as_ref())
}

fn build_wasm_blob(name: &str, debug: bool) -> anyhow::Result<()> {
    let mut c = build_cmd(if debug { "cargo wasm-build-debug" } else { "cargo wasm-build" });
    c.arg(format!("wasm-{name}"));
    exec(c)?;
    let profile = if debug { "small-dev" } else { "small" };
    let out_path = format!("target/wasm32-unknown-unknown/{profile}/wasm_{name}.wasm");
    if !debug {
        exec(build_cmd(format!("wasm-opt -Oz {out_path} -o {out_path}")))?;
    }
    exec(build_cmd(format!("cp {out_path} depell/src/{name}.wasm")))?;
    insert_stack_pointer_shim(format!("depell/src/{name}.wasm"))?;
    exec(build_cmd(format!("gzip -f depell/src/{name}.wasm")))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args[0].as_str() {
        "fmt" => fmt(args[1] == "-r" || args[1] == "--renumber").context(""),
        "build-depell-debug" => {
            build_wasm_blob("hbfmt", true)?;
            build_wasm_blob("hbc", true)?;
            exec(build_cmd("gzip -k -f depell/src/index.js"))?;
            exec(build_cmd("gzip -k -f depell/src/index.css"))?;
            Ok(())
        }
        "build-depell" => {
            build_wasm_blob("hbfmt", false)?;
            build_wasm_blob("hbc", false)?;
            exec(build_cmd("gzip -k -f depell/src/index.js"))?;
            exec(build_cmd("gzip -k -f depell/src/index.css"))?;
            Ok(())
        }
        "watch-depell-debug" => {
            let mut c = build_cmd("cargo watch");
            c.args(["--exec=xtask build-depell-debug", "--exec=run -p depell"]);
            exec(c)?;
            Ok(())
        }
        "watch-depell" => {
            let mut c = build_cmd("cargo watch");
            c.args(["--exec=xtask build-depell", "--exec=run -p depell --release"]);
            exec(c)?;
            Ok(())
        }
        "release-depell" => {
            exec(build_cmd("cargo xtask build-depell"))?;
            exec(build_cmd(
                "cargo build -p depell --release --features tls
                --target x86_64-unknown-linux-musl",
            ))?;
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
