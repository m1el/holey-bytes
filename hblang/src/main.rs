use std::io;

use hblang::{codegen, parser};

fn main() -> io::Result<()> {
    if std::env::args().len() == 1 {
        eprintln!("Usage: hblang <file1> <file2> ...");
        eprintln!(" 1. compiled binary will be printed to stdout");
        eprintln!(" 2. order of files matters");
        std::process::exit(1);
    }

    let files = std::env::args()
        .skip(1)
        .map(|path| std::fs::read_to_string(&path).map(|src| (path, src)))
        .collect::<io::Result<Vec<_>>>()?;
    let mut arena = parser::Arena::default();
    let mut codegen = codegen::Codegen::default();
    for (path, content) in files.iter() {
        let file = parser::Parser::new(content, path, &arena).file();
        codegen.file(path, content.as_bytes(), file);
        arena.clear();
    }
    codegen.dump(&mut std::io::stdout())
}
