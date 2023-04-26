//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

//#![deny(warnings)]
#![warn(clippy::nursery)]
#![allow(unused)]

use cairo_lang_compiler::{project::ProjectConfig, CompilerConfig};
use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};
use cairo_lang_sierra::program::Program;
use clap::{Parser, Subcommand};
use sierra2mlir::compiler::Compiler;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

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
        /// Path to the Cairo or Sierra source code.
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

        /// The available gas
        #[arg(short, long)]
        available_gas: Option<usize>,
    },
    /// Compile and run a program. The entry point must be a function without arguments.
    Run {
        /// Path to the Cairo or Sierra source code.
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

        /// The available gas
        #[arg(short, long)]
        available_gas: Option<usize>,
    },
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.command {
        Commands::Compile {
            input,
            optimize,
            output,
            debug,
            main_print,
            print_target,
            available_gas,
        } => {
            let program = load_program(&input);

            let mlir_output = sierra2mlir::compile(
                &program,
                optimize,
                debug,
                main_print,
                print_target,
                available_gas.unwrap_or(0),
            )?;

            if let Some(output) = output {
                fs::write(output, mlir_output);
            } else {
                println!("{mlir_output}");
            }
        }
        Commands::Run { function, input, main_print, print_target, available_gas } => {
            let program = load_program(&input);
            if !program.funcs.iter().any(|x| {
                x.id.debug_name.as_deref() == Some(&function)
                    || (main_print
                        && x.id.debug_name.as_ref().is_some_and(|x| x.as_str().ends_with("::main")))
            }) {
                panic!("Entry point {function} doesn't exist.");
            }

            let engine = sierra2mlir::execute(
                &program,
                main_print,
                print_target,
                available_gas.unwrap_or(0),
            )?;

            unsafe {
                engine.invoke_packed(&function, &mut [])?;
            };
        }
    }

    Ok(())
}

fn load_program(input: &Path) -> Program {
    match input.extension().map(|x| x.to_str().unwrap()) {
        Some("cairo") => Arc::try_unwrap(
            cairo_lang_compiler::compile_cairo_project_at_path(
                input,
                CompilerConfig { replace_ids: true, ..Default::default() },
            )
            .unwrap(),
        )
        .unwrap(),
        Some("sierra") => cairo_lang_sierra::ProgramParser::new()
            .parse(fs::read_to_string(input).unwrap().as_str())
            .unwrap(),
        _ => todo!("unknown file extension"),
    }
}
