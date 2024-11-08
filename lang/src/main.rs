#[cfg(feature = "std")]
fn main() {
    use std::io::Write;

    fn run(out: &mut Vec<u8>) -> std::io::Result<()> {
        let args = std::env::args().collect::<Vec<_>>();
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();

        let opts = hblang::Options::from_args(&args, out)?;
        let file = args.iter().filter(|a| !a.starts_with('-')).nth(1).copied().unwrap_or("main.hb");

        hblang::run_compiler(file, opts, out)
    }

    let mut out = Vec::new();
    match run(&mut out) {
        Ok(_) => std::io::stdout().write_all(&out).unwrap(),
        Err(_) => std::io::stderr().write_all(&out).unwrap(),
    }
}
