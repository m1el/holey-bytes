fn main() -> std::io::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let root = args.get(1).copied().unwrap_or("main.hb");

    if args.contains(&"--help") || args.contains(&"-h") {
        println!("Usage: hbc [OPTIONS...] <FILE>");
        println!(include_str!("../command-help.txt"));
        return Err(std::io::ErrorKind::Other.into());
    }

    let parsed = hblang::parse_from_fs(1, root)?;

    fn format_to_stdout(ast: hblang::parser::Ast) -> std::io::Result<()> {
        let source = std::fs::read_to_string(&*ast.path)?;
        hblang::parser::with_fmt_source(&source, || {
            for expr in ast.exprs() {
                use std::io::Write;
                writeln!(std::io::stdout(), "{expr}")?;
            }
            std::io::Result::Ok(())
        })
    }

    fn format_ast(ast: hblang::parser::Ast) -> std::io::Result<()> {
        let source = std::fs::read_to_string(&*ast.path)?;
        let mut output = Vec::new();
        hblang::parser::with_fmt_source(&source, || {
            for expr in ast.exprs() {
                use std::io::Write;
                writeln!(output, "{expr}")?;
            }
            std::io::Result::Ok(())
        })?;

        std::fs::write(&*ast.path, output)?;

        Ok(())
    }

    if args.contains(&"--fmt") {
        for parsed in parsed {
            format_ast(parsed)?;
        }
    } else if args.contains(&"--fmt-current") {
        format_to_stdout(parsed.into_iter().next().unwrap())?;
    } else {
        let mut codegen = hblang::codegen::Codegen::default();
        codegen.files = parsed;

        codegen.generate();
        codegen.dump(&mut std::io::stdout())?;
    }

    Ok(())
}
