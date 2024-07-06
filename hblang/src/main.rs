fn main() -> std::io::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();

    if args.contains(&"--help") || args.contains(&"-h") {
        println!("Usage: hbc [OPTIONS...] <FILE>");
        println!(include_str!("../command-help.txt"));
        return Err(std::io::ErrorKind::Other.into());
    }

    hblang::run_compiler(
        args.get(1).copied().unwrap_or("main.hb"),
        hblang::Options {
            fmt:         args.contains(&"--fmt"),
            fmt_current: args.contains(&"--fmt-current"),
        },
        &mut std::io::stdout(),
    )
}
