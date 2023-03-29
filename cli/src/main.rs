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
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile to MLIR with LLVM dialect, ready to be converted by `mlir-translate --mlir-to-llvmir`
    Compile {
        /// The input sierra file.
        input: PathBuf,

        /// Output optimized MLIR.
        #[arg(long)]
        optimize: bool,

        /// The output file. If not specified its output will be stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Turn on debug info.
        #[arg(short, long)]
        debug: bool,

        /// Enables printing the returned main value.
        #[arg(short, long)]
        main_print: bool,

        /// Set the file descriptor where print instructions will write to. A negative value will
        /// disable all prints.
        ///
        /// Common values:
        ///   1: stdout
        ///   2: stderr
        #[arg(short, long, default_value_t = 1)]
        print_target: i32,
    },
    /// Compile and run a program. The entry point must be a function without arguments.
    Run {
        /// The input sierra file.
        input: PathBuf,

        /// The function to run. Can only run functions without arguments and return types.
        #[arg(short, long)]
        function: String,

        /// Enables printing the returned main value.
        #[arg(short, long)]
        main_print: bool,

        /// Set the file descriptor where print instructions will write to. A negative value will
        /// disable all prints.
        ///
        /// Common values:
        ///   1: stdout
        ///   2: stderr
        #[arg(short, long, default_value_t = 1)]
        print_target: i32,
    },
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.command {
        Commands::Compile { input, optimize, output, debug, main_print, print_target } => {
            let code = fs::read_to_string(input)?;
            let mlir_output =
                sierra2mlir::compile(&code, optimize, debug, main_print, print_target)?;

            if let Some(output) = output {
                fs::write(output, mlir_output);
            } else {
                println!("{mlir_output}");
            }
        }
        Commands::Run { function, input, main_print, print_target } => {
            let code = fs::read_to_string(input)?;
            let engine = sierra2mlir::execute(&code, main_print, print_target)?;

            unsafe {
                engine.invoke_packed(&function, &mut [])?;
            };
        }
    }

    Ok(())
}
