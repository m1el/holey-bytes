use std::io::Write;

use hbasm::Assembler;

use {
    ariadne::{ColorGenerator, Label, Report, ReportKind, Source},
    std::{
        error::Error,
        io::{stdin, Read},
    },
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut code = String::new();
    stdin().read_to_string(&mut code)?;

    let mut assembler = Assembler::default();
    if let Err(e) = hbasm::text::assemble(&mut assembler, &code) {
        let mut colors = ColorGenerator::new();

        let e_code = match e.kind {
            hbasm::text::ErrorKind::UnexpectedToken => 1,
            hbasm::text::ErrorKind::InvalidToken => 2,
            hbasm::text::ErrorKind::UnexpectedEnd => 3,
            hbasm::text::ErrorKind::InvalidSymbol => 4,
        };
        let message = match e.kind {
            hbasm::text::ErrorKind::UnexpectedToken => "This token is not expected!",
            hbasm::text::ErrorKind::InvalidToken => "The token is not valid!",
            hbasm::text::ErrorKind::UnexpectedEnd => {
                "The assembler reached the end of input unexpectedly!"
            }
            hbasm::text::ErrorKind::InvalidSymbol => {
                "This referenced symbol doesn't have a corresponding label!"
            }
        };
        let a = colors.next();

        Report::build(ReportKind::Error, "engine_internal", e.span.clone().start)
            .with_code(e_code)
            .with_message(format!("{:?}", e.kind))
            .with_label(
                Label::new(("engine_internal", e.span))
                    .with_message(message)
                    .with_color(a),
            )
            .finish()
            .eprint(("engine_internal", Source::from(&code)))
            .unwrap();
    } else {
        assembler.finalise();
        std::io::stdout().lock().write_all(&assembler.buf).unwrap();
    }

    Ok(())
}
