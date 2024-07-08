mod fmt;
mod utils;

use {
    argh::FromArgs,
    std::{io, path::Path},
};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

/// xTask for Holey Bytes project
#[derive(FromArgs)]
struct Command {
    #[argh(subcommand)]
    subcom: Subcommands,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Subcommands {
    Format(fmt::Command),
}

fn main() -> io::Result<()> {
    match argh::from_env::<Command>().subcom {
        Subcommands::Format(com) => fmt::command(com),
    }
}
