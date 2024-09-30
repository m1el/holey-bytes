#[cfg(feature = "std")]
fn main() -> std::io::Result<()> {
    use std::io::Write;

    log::set_logger(&hblang::Logger).unwrap();

    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let opts = hblang::Options::from_args(&args)?;
    let file = args.iter().filter(|a| !a.starts_with('-')).nth(1).copied().unwrap_or("main.hb");

    let mut out = Vec::new();
    hblang::run_compiler(file, opts, &mut out)?;
    std::io::stdout().write_all(&out)
}
