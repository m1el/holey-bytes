mod fmt;
mod utils;

use {argh::FromArgs, color_eyre::Result, once_cell::sync::Lazy, std::path::Path};

static ROOT: Lazy<&Path> = Lazy::new(|| Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());

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

fn main() -> Result<()> {
    match argh::from_env::<Command>().subcom {
        Subcommands::Format(com) => fmt::command(com),
    }
}
