//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

#![forbid(unsafe_code)]
//#![deny(warnings)]
#![warn(clippy::nursery)]

use std::path::PathBuf;

use clap::Parser;

mod compiler;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about,
    long_about = r#"A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR."#
)]
struct Args {
    /// The input sierra file.
    #[arg(short, long)]
    input: PathBuf,

    /// The output file.
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let _args = Args::parse();

    Ok(())
}
