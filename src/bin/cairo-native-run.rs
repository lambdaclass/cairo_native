mod utils;

use anyhow::Context;
use cairo_lang_compiler::{
    db::RootDatabase,
    diagnostics::DiagnosticsReporter,
    project::{check_compiler_path, setup_project},
};
use cairo_lang_diagnostics::ToOption;
use cairo_lang_runner::short_string::as_cairo_short_string;
use cairo_lang_sierra_generator::{
    db::SierraGenGroup,
    replace_ids::{DebugReplacer, SierraIdReplacer},
};
use cairo_lang_starknet::contract::get_contracts_info;
use cairo_native::{
    context::NativeContext,
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    utils::cairo_get_debug_locations,
};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use utils::{find_function, result_to_runresult};

#[derive(Clone, Debug, ValueEnum)]
enum RunMode {
    Aot,
    Jit,
}

/// Command line args parser.
/// Exits with 1 if the compilation or run fails, otherwise 0.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The Cairo project path to compile and run its tests.
    path: PathBuf,
    /// Whether path is a single file.
    #[arg(short, long)]
    single_file: bool,
    /// Allows the compilation to succeed with warnings.
    #[arg(long)]
    allow_warnings: bool,
    /// In cases where gas is available, the amount of provided gas.
    #[arg(long)]
    available_gas: Option<usize>,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
}

fn main() -> anyhow::Result<()> {
    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    let args = Args::parse();

    // Check if args.path is a file or a directory.
    check_compiler_path(args.single_file, &args.path)?;

    let db = &mut RootDatabase::builder().detect_corelib().build()?;

    let main_crate_ids = setup_project(db, Path::new(&args.path))?;

    let mut reporter = DiagnosticsReporter::stderr();
    if args.allow_warnings {
        reporter = reporter.allow_warnings();
    }
    if reporter.check(db) {
        anyhow::bail!("failed to compile: {}", args.path.display());
    }

    let sierra_program = db
        .get_sierra_program(main_crate_ids.clone())
        .to_option()
        .with_context(|| "Compilation failed without any diagnostics.")?
        .program
        .clone();
    let replacer = DebugReplacer { db };
    if args.available_gas.is_none() && sierra_program.requires_gas_counter() {
        anyhow::bail!("Program requires gas counter, please provide `--available-gas` argument.");
    }

    let _contracts_info = get_contracts_info(db, main_crate_ids, &replacer)?;
    let sierra_program = replacer.apply(&sierra_program);

    let native_context = NativeContext::new();

    let debug_locations = cairo_get_debug_locations(native_context.context(), db, &sierra_program)?;

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile(&sierra_program, Some(debug_locations))
        .unwrap();

    let native_executor: NativeExecutor = match args.run_mode {
        RunMode::Aot => {
            AotNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
        }
        RunMode::Jit => {
            JitNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
        }
    };

    let gas_metadata =
        GasMetadata::new(&sierra_program, Some(MetadataComputationConfig::default())).unwrap();

    let func = find_function(&sierra_program, "::main")?;

    let initial_gas = gas_metadata
        .get_initial_available_gas(&func.id, args.available_gas.map(|x| x.try_into().unwrap()))
        .with_context(|| "not enough gas to run")?;

    let result = native_executor
        .invoke_dynamic(&func.id, &[], Some(initial_gas))
        .with_context(|| "Failed to run the function.")?;

    let run_result = result_to_runresult(&result)?;

    match run_result {
        cairo_lang_runner::RunResultValue::Success(values) => {
            println!("Run completed successfully, returning {values:?}")
        }
        cairo_lang_runner::RunResultValue::Panic(values) => {
            print!("Run panicked with [");
            for value in &values {
                match as_cairo_short_string(value) {
                    Some(as_string) => print!("{value} ('{as_string}'), "),
                    None => print!("{value}, "),
                }
            }
            println!("].")
        }
    }
    if let Some(gas) = result.remaining_gas {
        println!("Remaining gas: {gas}");
    }

    Ok(())
}
