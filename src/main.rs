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
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};

use sierra2mlir::compiler::Compiler;

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
    compiler.run_gpu()?;

    let pass_manager = pass::Manager::new(&compiler.context);
    register_all_passes();
    //pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    //pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    //pass_manager.add_pass(pass::conversion::convert_gpu_to_llvm());
    //pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());
    //pass_manager.enable_verifier(true);
    pass_manager.run(&mut compiler.module)?;

    /*
    let engine = ExecutionEngine::new(&compiler.module, 2, &[], false);

    let mut result: i32 = -1;

    println!("unjitted");
    let now = Instant::now();
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .unwrap();
    };
    let done = now.elapsed();
    println!("{done:?}");
    println!("jitted");
    let now = Instant::now();
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .unwrap();
    };
    let done = now.elapsed();
    println!("{done:?}");

    dbg!(result);
    */

    let op = compiler.module.as_operation();
    dbg!(&op);
    if op.verify() {
        let output = op.to_string();
        fs::write(args.output, output);
        Ok(())
    } else {
        Err(color_eyre::eyre::eyre!("error verifiying"))
    }
}
