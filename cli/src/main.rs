//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

//#![deny(warnings)]
#![warn(clippy::nursery)]
#![allow(unused)]

use clap::{Parser, Subcommand};
use sierra2mlir::compiler::Compiler;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
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

    /// Output optimized MLIR.
    #[arg(short, long)]
    optimize: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile to MLIR with LLVM dialect, ready to be converted by `mlir-translate --mlir-to-llvmir`
    Compile {
        /// The output file. If not specified its output will be stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Turn on debug info.
        #[arg(short, long)]
        debug: bool,

        /// Add a main method with a print on the return value.
        #[arg(short, long)]
        main_print: bool,
    },
    /// Compile and run a program. The entry point must be a function without arguments.
    Run {
        /// The function to run. Can only run functions without arguments and return types.
        #[arg(short, long)]
        function: String,
    },
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let code = fs::read_to_string(args.input)?;

    match args.command {
        Commands::Compile { output, debug, main_print } => {
            let mlir_output = sierra2mlir::compile(&code, args.optimize, debug, main_print)?;

            if let Some(output) = output {
                fs::write(output, mlir_output);
            } else {
                println!("{mlir_output}");
            }
        }
        Commands::Run { function } => {
            let engine = sierra2mlir::execute(&code)?;

            unsafe {
                engine.invoke_packed(&function, &mut [])?;
            };
        }
    }

    Ok(())
}
