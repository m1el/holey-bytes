use std::{
    error::Error,
    io::{stdin, stdout, Read, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut code = String::new();
    stdin().read_to_string(&mut code)?;

    let mut buf = vec![];
    if let Err(e) = hbasm::assembly(&code, &mut buf) {
        eprintln!(
            "Error {:?} at {:?} (`{}`)",
            e.kind,
            e.span.clone(),
            &code[e.span],
        );
    }
    stdout().write_all(&buf)?;
    Ok(())
}
