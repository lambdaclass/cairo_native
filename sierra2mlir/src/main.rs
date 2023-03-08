//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

//#![deny(warnings)]
#![warn(clippy::nursery)]
#![allow(unused)]

use std::{fs, path::PathBuf, time::Instant};

use cairo_lang_sierra::{program::Program, ProgramParser};
use clap::Parser;
use melior_next::{
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
    pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());

    pass_manager.enable_verifier(true);

    pass_manager.run(&mut compiler.module)?;

    let engine = ExecutionEngine::new(&compiler.module, 2, &[]);

    let mut result1: i32 = -1;
    let mut result2: i32 = -1;

    let now = Instant::now();
    unsafe {
        engine.invoke_packed(
            "main",
            &mut [
                &mut result1 as *mut i32 as *mut (),
                &mut result2 as *mut i32 as *mut (),
            ],
        );
    };
    let done = now.elapsed();
    println!("{done:?}");

    dbg!(result1);
    dbg!(result2);

    let op = compiler.module.as_operation();
    if op.verify() {
        let output = op.to_string();
        fs::write(args.output, output);
        Ok(())
    } else {
        Err(color_eyre::eyre::eyre!("error verifiying"))
    }
}
