use anyhow::Context;
use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_runner::short_string::as_cairo_short_string;
use cairo_native::{
    context::NativeContext,
    executor::{AotNativeExecutor, JitNativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    starknet_stub::StubSyscallHandler,
};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use utils::{find_function, result_to_runresult};

mod utils;

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

    let mut db = RootDatabase::builder().detect_corelib().build()?;
    let main_crate_ids = setup_project(&mut db, &args.path)?;

    let sierra_program = compile_prepared_db(
        &db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )?
    .program;

    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context.compile(&sierra_program, false).unwrap();

    let native_executor: Box<dyn Fn(_, _, _, &mut StubSyscallHandler) -> _> = match args.run_mode {
        RunMode::Aot => {
            let executor =
                AotNativeExecutor::from_native_module(native_module, args.opt_level.into());
            Box::new(move |function_id, args, gas, syscall_handler| {
                executor.invoke_dynamic_with_syscall_handler(
                    function_id,
                    args,
                    gas,
                    syscall_handler,
                )
            })
        }
        RunMode::Jit => {
            let executor =
                JitNativeExecutor::from_native_module(native_module, args.opt_level.into());
            Box::new(move |function_id, args, gas, syscall_handler| {
                executor.invoke_dynamic_with_syscall_handler(
                    function_id,
                    args,
                    gas,
                    syscall_handler,
                )
            })
        }
    };

    let gas_metadata =
        GasMetadata::new(&sierra_program, Some(MetadataComputationConfig::default())).unwrap();

    let func = find_function(&sierra_program, "::main")?;

    let initial_gas = gas_metadata
        .get_initial_available_gas(&func.id, args.available_gas.map(|x| x.try_into().unwrap()))
        .with_context(|| "not enough gas to run")?;

    let mut syscall_handler = StubSyscallHandler::default();

    let result = native_executor(&func.id, &[], Some(initial_gas), &mut syscall_handler)
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
