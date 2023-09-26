use std::io::{stdin, Read};

/// Holey Bytes Experimental Runtime

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut prog = vec![];
    stdin().read_to_end(&mut prog)?;

    eprintln!("WARNING! Bytecode valider has not been yet implemented and running program can lead to undefiend behaviour.");
    Ok(())
}
