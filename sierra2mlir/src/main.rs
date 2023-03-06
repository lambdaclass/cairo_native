//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::nursery)]

use std::path::PathBuf;

use clap::Parser;

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

fn main() {
    let _args = Args::parse();
}
