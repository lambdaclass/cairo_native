use anyhow::{bail, Context};
use cairo_lang_compiler::{
    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
};
use cairo_lang_diagnostics::ToOption;
use cairo_lang_runner::{short_string::as_cairo_short_string, RunResultValue};
use cairo_lang_sierra::program::{Function, Program};
use cairo_lang_sierra_generator::{
    db::SierraGenGroup,
    replace_ids::{DebugReplacer, SierraIdReplacer},
};
use cairo_lang_starknet::contract::get_contracts_info;
use cairo_native::{
    context::NativeContext,
    execution_result::ExecutionResult,
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    metadata::{
        gas::{GasMetadata, MetadataComputationConfig},
        MetadataStorage,
    },
    values::JitValue,
    OptLevel,
};
use clap::{Parser, ValueEnum};
use itertools::Itertools;
use starknet_types_core::felt::Felt;
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

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

    let (sierra_program, _statements_locations) = db
        .get_sierra_program(main_crate_ids.clone())
        .to_option()
        .with_context(|| "Compilation failed without any diagnostics.")?;
    let replacer = DebugReplacer { db };
    if args.available_gas.is_none() && sierra_program.requires_gas_counter() {
        anyhow::bail!("Program requires gas counter, please provide `--available-gas` argument.");
    }

    let _contracts_info = get_contracts_info(db, main_crate_ids, &replacer)?;
    let sierra_program = replacer.apply(&sierra_program);

    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile(&sierra_program, MetadataStorage::default())
        .unwrap();

    let opt_level = match args.opt_level {
        0 => OptLevel::None,
        1 => OptLevel::Less,
        2 => OptLevel::Default,
        _ => OptLevel::Aggressive,
    };

    let native_executor: NativeExecutor = match args.run_mode {
        RunMode::Aot => AotNativeExecutor::from_native_module(native_module, opt_level).into(),
        RunMode::Jit => JitNativeExecutor::from_native_module(native_module, opt_level).into(),
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

pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
    if path.is_file() {
        if !single_file {
            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
        }
    } else if path.is_dir() {
        if single_file {
            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
        }
    } else {
        anyhow::bail!("The given path does not exist.");
    }
    Ok(())
}

pub fn find_function<'a>(
    sierra_program: &'a Program,
    name_suffix: &str,
) -> anyhow::Result<&'a Function> {
    if let Some(x) = sierra_program.funcs.iter().find(|f| {
        if let Some(name) = &f.id.debug_name {
            name.ends_with(name_suffix)
        } else {
            false
        }
    }) {
        Ok(x)
    } else {
        bail!("function {name_suffix} not found")
    }
}

fn result_to_runresult(result: &ExecutionResult) -> anyhow::Result<RunResultValue> {
    let is_success;
    let mut felts: Vec<Felt> = Vec::new();

    match &result.return_value {
        outer_value @ JitValue::Enum {
            tag,
            value,
            debug_name,
        } => {
            let debug_name = debug_name.as_ref().expect("missing debug name");

            if debug_name.starts_with("core::panics::PanicResult::") {
                is_success = *tag == 0;

                if !is_success {
                    match &**value {
                        JitValue::Struct { fields, .. } => {
                            for field in fields {
                                let felt = jitvalue_to_felt(field);
                                felts.extend(felt);
                            }
                        }
                        _ => bail!("unsuported return value in cairo-native"),
                    }
                } else {
                    felts.extend(jitvalue_to_felt(value));
                }
            } else {
                is_success = true;
                felts.extend(jitvalue_to_felt(outer_value));
            }
        }
        x => {
            is_success = true;
            felts.extend(jitvalue_to_felt(x));
        }
    }

    let return_values = felts
        .into_iter()
        .map(|x| x.to_bigint().into())
        .collect_vec();

    Ok(match is_success {
        true => RunResultValue::Success(return_values),
        false => RunResultValue::Panic(return_values),
    })
}

fn jitvalue_to_felt(value: &JitValue) -> Vec<Felt> {
    let mut felts = Vec::new();
    match value {
        JitValue::Felt252(felt) => vec![felt.to_bigint().into()],
        JitValue::Bytes31(_) => todo!(),
        JitValue::Array(values) => {
            for value in values {
                let felt = jitvalue_to_felt(value);
                felts.extend(felt);
            }
            felts
        }
        JitValue::Struct { fields, .. } => {
            for field in fields {
                let felt = jitvalue_to_felt(field);
                felts.extend(felt);
            }
            felts
        }
        JitValue::Enum {
            value: _,
            tag,
            debug_name,
        } => {
            if let Some(debug_name) = debug_name {
                if debug_name == "core::bool" {
                    vec![(*tag == 1).into()]
                } else {
                    todo!()
                }
            } else {
                todo!()
            }
        }
        JitValue::Felt252Dict { .. } => todo!(),
        JitValue::Uint8(x) => vec![(*x).into()],
        JitValue::Uint16(x) => vec![(*x).into()],
        JitValue::Uint32(x) => vec![(*x).into()],
        JitValue::Uint64(x) => vec![(*x).into()],
        JitValue::Uint128(x) => vec![(*x).into()],
        JitValue::Sint8(x) => vec![(*x).into()],
        JitValue::Sint16(x) => vec![(*x).into()],
        JitValue::Sint32(x) => vec![(*x).into()],
        JitValue::Sint64(x) => vec![(*x).into()],
        JitValue::Sint128(x) => vec![(*x).into()],
        JitValue::EcPoint(_, _) => todo!(),
        JitValue::EcState(_, _, _, _) => todo!(),
        JitValue::Secp256K1Point { .. } => todo!(),
        JitValue::Secp256R1Point { .. } => todo!(),
        JitValue::Null => vec![0.into()],
    }
}
