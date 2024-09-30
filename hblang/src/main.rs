#[cfg(feature = "std")]
fn main() -> std::io::Result<()> {
    use std::{io::Write, num::NonZeroUsize};

    log::set_logger(&hblang::Logger).unwrap();

    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();

    if args.contains(&"--help") || args.contains(&"-h") {
        println!("Usage: hbc [OPTIONS...] <FILE>");
        println!(include_str!("../command-help.txt"));
        return Err(std::io::ErrorKind::Other.into());
    }

    let mut out = Vec::new();
    hblang::run_compiler(
        args.iter().filter(|a| !a.starts_with('-')).nth(1).copied().unwrap_or("main.hb"),
        hblang::Options {
            fmt: args.contains(&"--fmt"),
            fmt_stdout: args.contains(&"--fmt-stdout"),
            dump_asm: args.contains(&"--dump-asm"),
            extra_threads: args
                .iter()
                .position(|&a| a == "--threads")
                .map(|i| args[i + 1].parse::<NonZeroUsize>().expect("--threads expects integer"))
                .map_or(1, NonZeroUsize::get)
                - 1,
        },
        &mut out,
    )?;
    std::io::stdout().write_all(&out)
}
