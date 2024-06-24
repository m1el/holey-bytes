fn main() -> std::io::Result<()> {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "main.hb".to_string());

    let parsed = hblang::parse_from_fs(1, &root)?;
    let mut codegen = hblang::codegen::Codegen::default();
    codegen.files = parsed;

    codegen.generate();
    codegen.dump(&mut std::io::stdout())
}
