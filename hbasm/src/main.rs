use std::{io::stdout, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or("Missing path")?);
    hbasm::assembler(&mut stdout(), |engine| engine.run_file(path))?;
    
    Ok(())
}
