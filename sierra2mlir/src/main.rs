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
    pass,
    utility::register_all_dialects,
    Context, ExecutionEngine,
};

use crate::compiler::Compiler;

mod compiler;
mod libfuncs;
mod statements;
mod types;

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

    let mut compiler = Compiler::new(&code)?;
    //let op = compiler.compile()?;
    compiler.run_fib()?;

    let pass_manager = pass::Manager::new(&compiler.context);
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());

    pass_manager
        .nested_under("func.func")
        .add_pass(pass::conversion::convert_arithmetic_to_llvm());

    assert_eq!(pass_manager.run(&mut compiler.module), Ok(()));

    //let engine = ExecutionEngine::new(&compiler.module, 2, &[]);

    let op = compiler.module.as_operation();
    if op.verify() {
        let output = op.to_string();
        fs::write(args.output, output);
        Ok(())
    } else {
        Err(color_eyre::eyre::eyre!("error verifiying"))
    }
}
