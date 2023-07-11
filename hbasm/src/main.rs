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

    let mut buf = vec![];

    if let Err(e) = hbasm::text::assembly(&code, &mut buf) {
        let mut colors = ColorGenerator::new();

        let e_code = match e.kind {
            hbasm::text::ErrorKind::UnexpectedToken => 1,
            hbasm::text::ErrorKind::InvalidToken => 2,
            hbasm::text::ErrorKind::UnexpectedEnd => 3,
            hbasm::text::ErrorKind::InvalidSymbol => 4,
        };
        let a = colors.next();

        Report::build(ReportKind::Error, "engine_internal", 12)
        .with_code(e_code)
        .with_message(format!("{:?}", e.kind))
        .with_label(
            Label::new(("engine_internal", e.span.clone()))
                .with_message(format!("{:?}", e.kind))
                .with_color(a),
        )
        .finish()
        .eprint(("engine_internal", Source::from(&code)))
        .unwrap();
    }

    Ok(())
}
