//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

#![forbid(unsafe_code)]
//#![deny(warnings)]
#![warn(clippy::nursery)]
#![allow(unused)]

use std::{fs, path::PathBuf};

use cairo_lang_sierra::{program::Program, ProgramParser};
use clap::Parser;
use melior::{
    dialect,
    ir::{operation, Attribute, Block, Identifier, Location, Module, Region, Type, Value},
    utility::register_all_dialects,
    Context,
};

use crate::compiler::Compiler;

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
    let args = Args::parse();

    let code = fs::read_to_string(args.input)?;

    let compiler = Compiler::new(&code)?;
    compiler.run()?;
    Ok(())
}
