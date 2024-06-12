//use anyhow::{bail, Context};
use anyhow::{bail, Context};
//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
//};
};
//use cairo_lang_diagnostics::ToOption;
use cairo_lang_diagnostics::ToOption;
//use cairo_lang_runner::{short_string::as_cairo_short_string, RunResultValue};
use cairo_lang_runner::{short_string::as_cairo_short_string, RunResultValue};
//use cairo_lang_sierra::program::{Function, Program};
use cairo_lang_sierra::program::{Function, Program};
//use cairo_lang_sierra_generator::{
use cairo_lang_sierra_generator::{
//    db::SierraGenGroup,
    db::SierraGenGroup,
//    replace_ids::{DebugReplacer, SierraIdReplacer},
    replace_ids::{DebugReplacer, SierraIdReplacer},
//};
};
//use cairo_lang_starknet::contract::get_contracts_info;
use cairo_lang_starknet::contract::get_contracts_info;
//use cairo_native::{
use cairo_native::{
//    context::NativeContext,
    context::NativeContext,
//    debug_info::{DebugInfo, DebugLocations},
    debug_info::{DebugInfo, DebugLocations},
//    execution_result::ExecutionResult,
    execution_result::ExecutionResult,
//    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
//    metadata::gas::{GasMetadata, MetadataComputationConfig},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
//    values::JitValue,
    values::JitValue,
//};
};
//use clap::{Parser, ValueEnum};
use clap::{Parser, ValueEnum};
//use itertools::Itertools;
use itertools::Itertools;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::path::{Path, PathBuf};
use std::path::{Path, PathBuf};
//use tracing_subscriber::{EnvFilter, FmtSubscriber};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
//

//#[derive(Clone, Debug, ValueEnum)]
#[derive(Clone, Debug, ValueEnum)]
//enum RunMode {
enum RunMode {
//    Aot,
    Aot,
//    Jit,
    Jit,
//}
}
//

///// Command line args parser.
/// Command line args parser.
///// Exits with 1 if the compilation or run fails, otherwise 0.
/// Exits with 1 if the compilation or run fails, otherwise 0.
//#[derive(Parser, Debug)]
#[derive(Parser, Debug)]
//#[clap(version, verbatim_doc_comment)]
#[clap(version, verbatim_doc_comment)]
//struct Args {
struct Args {
//    /// The Cairo project path to compile and run its tests.
    /// The Cairo project path to compile and run its tests.
//    path: PathBuf,
    path: PathBuf,
//    /// Whether path is a single file.
    /// Whether path is a single file.
//    #[arg(short, long)]
    #[arg(short, long)]
//    single_file: bool,
    single_file: bool,
//    /// Allows the compilation to succeed with warnings.
    /// Allows the compilation to succeed with warnings.
//    #[arg(long)]
    #[arg(long)]
//    allow_warnings: bool,
    allow_warnings: bool,
//    /// In cases where gas is available, the amount of provided gas.
    /// In cases where gas is available, the amount of provided gas.
//    #[arg(long)]
    #[arg(long)]
//    available_gas: Option<usize>,
    available_gas: Option<usize>,
//    /// Run with JIT or AOT (compiled).
    /// Run with JIT or AOT (compiled).
//    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
//    run_mode: RunMode,
    run_mode: RunMode,
//    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
//    #[arg(short = 'O', long, default_value_t = 0)]
    #[arg(short = 'O', long, default_value_t = 0)]
//    opt_level: u8,
    opt_level: u8,
//}
}
//

//fn main() -> anyhow::Result<()> {
fn main() -> anyhow::Result<()> {
//    // Configure logging and error handling.
    // Configure logging and error handling.
//    tracing::subscriber::set_global_default(
    tracing::subscriber::set_global_default(
//        FmtSubscriber::builder()
        FmtSubscriber::builder()
//            .with_env_filter(EnvFilter::from_default_env())
            .with_env_filter(EnvFilter::from_default_env())
//            .finish(),
            .finish(),
//    )?;
    )?;
//

//    let args = Args::parse();
    let args = Args::parse();
//

//    // Check if args.path is a file or a directory.
    // Check if args.path is a file or a directory.
//    check_compiler_path(args.single_file, &args.path)?;
    check_compiler_path(args.single_file, &args.path)?;
//

//    let db = &mut RootDatabase::builder().detect_corelib().build()?;
    let db = &mut RootDatabase::builder().detect_corelib().build()?;
//

//    let main_crate_ids = setup_project(db, Path::new(&args.path))?;
    let main_crate_ids = setup_project(db, Path::new(&args.path))?;
//

//    let mut reporter = DiagnosticsReporter::stderr();
    let mut reporter = DiagnosticsReporter::stderr();
//    if args.allow_warnings {
    if args.allow_warnings {
//        reporter = reporter.allow_warnings();
        reporter = reporter.allow_warnings();
//    }
    }
//    if reporter.check(db) {
    if reporter.check(db) {
//        anyhow::bail!("failed to compile: {}", args.path.display());
        anyhow::bail!("failed to compile: {}", args.path.display());
//    }
    }
//

//    let sierra_program = db
    let sierra_program = db
//        .get_sierra_program(main_crate_ids.clone())
        .get_sierra_program(main_crate_ids.clone())
//        .to_option()
        .to_option()
//        .with_context(|| "Compilation failed without any diagnostics.")?
        .with_context(|| "Compilation failed without any diagnostics.")?
//        .program
        .program
//        .clone();
        .clone();
//    let replacer = DebugReplacer { db };
    let replacer = DebugReplacer { db };
//    if args.available_gas.is_none() && sierra_program.requires_gas_counter() {
    if args.available_gas.is_none() && sierra_program.requires_gas_counter() {
//        anyhow::bail!("Program requires gas counter, please provide `--available-gas` argument.");
        anyhow::bail!("Program requires gas counter, please provide `--available-gas` argument.");
//    }
    }
//

//    let _contracts_info = get_contracts_info(db, main_crate_ids, &replacer)?;
    let _contracts_info = get_contracts_info(db, main_crate_ids, &replacer)?;
//    let sierra_program = replacer.apply(&sierra_program);
    let sierra_program = replacer.apply(&sierra_program);
//

//    let native_context = NativeContext::new();
    let native_context = NativeContext::new();
//

//    let debug_locations = {
    let debug_locations = {
//        let debug_info = DebugInfo::extract(db, &sierra_program)
        let debug_info = DebugInfo::extract(db, &sierra_program)
//            .map_err(|_| {
            .map_err(|_| {
//                let mut buffer = String::new();
                let mut buffer = String::new();
//                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(db));
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(db));
//                buffer
                buffer
//            })
            })
//            .unwrap();
            .unwrap();
//

//        DebugLocations::extract(native_context.context(), db, &debug_info)
        DebugLocations::extract(native_context.context(), db, &debug_info)
//    };
    };
//

//    // Compile the sierra program into a MLIR module.
    // Compile the sierra program into a MLIR module.
//    let native_module = native_context
    let native_module = native_context
//        .compile(&sierra_program, Some(debug_locations))
        .compile(&sierra_program, Some(debug_locations))
//        .unwrap();
        .unwrap();
//

//    let native_executor: NativeExecutor = match args.run_mode {
    let native_executor: NativeExecutor = match args.run_mode {
//        RunMode::Aot => {
        RunMode::Aot => {
//            AotNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
            AotNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
//        }
        }
//        RunMode::Jit => {
        RunMode::Jit => {
//            JitNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
            JitNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
//        }
        }
//    };
    };
//

//    let gas_metadata =
    let gas_metadata =
//        GasMetadata::new(&sierra_program, Some(MetadataComputationConfig::default())).unwrap();
        GasMetadata::new(&sierra_program, Some(MetadataComputationConfig::default())).unwrap();
//

//    let func = find_function(&sierra_program, "::main")?;
    let func = find_function(&sierra_program, "::main")?;
//

//    let initial_gas = gas_metadata
    let initial_gas = gas_metadata
//        .get_initial_available_gas(&func.id, args.available_gas.map(|x| x.try_into().unwrap()))
        .get_initial_available_gas(&func.id, args.available_gas.map(|x| x.try_into().unwrap()))
//        .with_context(|| "not enough gas to run")?;
        .with_context(|| "not enough gas to run")?;
//

//    let result = native_executor
    let result = native_executor
//        .invoke_dynamic(&func.id, &[], Some(initial_gas))
        .invoke_dynamic(&func.id, &[], Some(initial_gas))
//        .with_context(|| "Failed to run the function.")?;
        .with_context(|| "Failed to run the function.")?;
//

//    let run_result = result_to_runresult(&result)?;
    let run_result = result_to_runresult(&result)?;
//

//    match run_result {
    match run_result {
//        cairo_lang_runner::RunResultValue::Success(values) => {
        cairo_lang_runner::RunResultValue::Success(values) => {
//            println!("Run completed successfully, returning {values:?}")
            println!("Run completed successfully, returning {values:?}")
//        }
        }
//        cairo_lang_runner::RunResultValue::Panic(values) => {
        cairo_lang_runner::RunResultValue::Panic(values) => {
//            print!("Run panicked with [");
            print!("Run panicked with [");
//            for value in &values {
            for value in &values {
//                match as_cairo_short_string(value) {
                match as_cairo_short_string(value) {
//                    Some(as_string) => print!("{value} ('{as_string}'), "),
                    Some(as_string) => print!("{value} ('{as_string}'), "),
//                    None => print!("{value}, "),
                    None => print!("{value}, "),
//                }
                }
//            }
            }
//            println!("].")
            println!("].")
//        }
        }
//    }
    }
//    if let Some(gas) = result.remaining_gas {
    if let Some(gas) = result.remaining_gas {
//        println!("Remaining gas: {gas}");
        println!("Remaining gas: {gas}");
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

//pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
//    if path.is_file() {
    if path.is_file() {
//        if !single_file {
        if !single_file {
//            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
//        }
        }
//    } else if path.is_dir() {
    } else if path.is_dir() {
//        if single_file {
        if single_file {
//            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
//        }
        }
//    } else {
    } else {
//        anyhow::bail!("The given path does not exist.");
        anyhow::bail!("The given path does not exist.");
//    }
    }
//    Ok(())
    Ok(())
//}
}
//

//pub fn find_function<'a>(
pub fn find_function<'a>(
//    sierra_program: &'a Program,
    sierra_program: &'a Program,
//    name_suffix: &str,
    name_suffix: &str,
//) -> anyhow::Result<&'a Function> {
) -> anyhow::Result<&'a Function> {
//    if let Some(x) = sierra_program.funcs.iter().find(|f| {
    if let Some(x) = sierra_program.funcs.iter().find(|f| {
//        if let Some(name) = &f.id.debug_name {
        if let Some(name) = &f.id.debug_name {
//            name.ends_with(name_suffix)
            name.ends_with(name_suffix)
//        } else {
        } else {
//            false
            false
//        }
        }
//    }) {
    }) {
//        Ok(x)
        Ok(x)
//    } else {
    } else {
//        bail!("function {name_suffix} not found")
        bail!("function {name_suffix} not found")
//    }
    }
//}
}
//

//fn result_to_runresult(result: &ExecutionResult) -> anyhow::Result<RunResultValue> {
fn result_to_runresult(result: &ExecutionResult) -> anyhow::Result<RunResultValue> {
//    let is_success;
    let is_success;
//    let mut felts: Vec<Felt> = Vec::new();
    let mut felts: Vec<Felt> = Vec::new();
//

//    match &result.return_value {
    match &result.return_value {
//        outer_value @ JitValue::Enum {
        outer_value @ JitValue::Enum {
//            tag,
            tag,
//            value,
            value,
//            debug_name,
            debug_name,
//        } => {
        } => {
//            if debug_name
            if debug_name
//                .as_ref()
                .as_ref()
//                .expect("missing debug name")
                .expect("missing debug name")
//                .starts_with("core::panics::PanicResult::")
                .starts_with("core::panics::PanicResult::")
//            {
            {
//                is_success = *tag == 0;
                is_success = *tag == 0;
//

//                if !is_success {
                if !is_success {
//                    match &**value {
                    match &**value {
//                        JitValue::Struct { fields, .. } => {
                        JitValue::Struct { fields, .. } => {
//                            for field in fields {
                            for field in fields {
//                                let felt = jitvalue_to_felt(field);
                                let felt = jitvalue_to_felt(field);
//                                felts.extend(felt);
                                felts.extend(felt);
//                            }
                            }
//                        }
                        }
//                        _ => bail!("unsuported return value in cairo-native"),
                        _ => bail!("unsuported return value in cairo-native"),
//                    }
                    }
//                } else {
                } else {
//                    felts.extend(jitvalue_to_felt(value));
                    felts.extend(jitvalue_to_felt(value));
//                }
                }
//            } else {
            } else {
//                is_success = true;
                is_success = true;
//                felts.extend(jitvalue_to_felt(outer_value));
                felts.extend(jitvalue_to_felt(outer_value));
//            }
            }
//        }
        }
//        x => {
        x => {
//            is_success = true;
            is_success = true;
//            felts.extend(jitvalue_to_felt(x));
            felts.extend(jitvalue_to_felt(x));
//        }
        }
//    }
    }
//

//    let return_values = felts
    let return_values = felts
//        .into_iter()
        .into_iter()
//        .map(|x| x.to_bigint().into())
        .map(|x| x.to_bigint().into())
//        .collect_vec();
        .collect_vec();
//

//    Ok(match is_success {
    Ok(match is_success {
//        true => RunResultValue::Success(return_values),
        true => RunResultValue::Success(return_values),
//        false => RunResultValue::Panic(return_values),
        false => RunResultValue::Panic(return_values),
//    })
    })
//}
}
//

//fn jitvalue_to_felt(value: &JitValue) -> Vec<Felt> {
fn jitvalue_to_felt(value: &JitValue) -> Vec<Felt> {
//    match value {
    match value {
//        JitValue::Felt252(felt) => vec![*felt],
        JitValue::Felt252(felt) => vec![*felt],
//        JitValue::BoundedInt { value, .. } => vec![*value],
        JitValue::BoundedInt { value, .. } => vec![*value],
//        JitValue::Bytes31(bytes) => vec![Felt::from_bytes_le_slice(bytes)],
        JitValue::Bytes31(bytes) => vec![Felt::from_bytes_le_slice(bytes)],
//        JitValue::Array(fields) | JitValue::Struct { fields, .. } => {
        JitValue::Array(fields) | JitValue::Struct { fields, .. } => {
//            fields.iter().flat_map(jitvalue_to_felt).collect()
            fields.iter().flat_map(jitvalue_to_felt).collect()
//        }
        }
//        JitValue::Enum {
        JitValue::Enum {
//            value,
            value,
//            tag,
            tag,
//            debug_name,
            debug_name,
//        } => {
        } => {
//            if let Some(debug_name) = debug_name {
            if let Some(debug_name) = debug_name {
//                if debug_name == "core::bool" {
                if debug_name == "core::bool" {
//                    vec![(*tag == 1).into()]
                    vec![(*tag == 1).into()]
//                } else {
                } else {
//                    let mut felts = vec![(*tag).into()];
                    let mut felts = vec![(*tag).into()];
//                    felts.extend(jitvalue_to_felt(value));
                    felts.extend(jitvalue_to_felt(value));
//                    felts
                    felts
//                }
                }
//            } else {
            } else {
//                todo!()
                todo!()
//            }
            }
//        }
        }
//        JitValue::Uint8(x) => vec![(*x).into()],
        JitValue::Uint8(x) => vec![(*x).into()],
//        JitValue::Uint16(x) => vec![(*x).into()],
        JitValue::Uint16(x) => vec![(*x).into()],
//        JitValue::Uint32(x) => vec![(*x).into()],
        JitValue::Uint32(x) => vec![(*x).into()],
//        JitValue::Uint64(x) => vec![(*x).into()],
        JitValue::Uint64(x) => vec![(*x).into()],
//        JitValue::Uint128(x) => vec![(*x).into()],
        JitValue::Uint128(x) => vec![(*x).into()],
//        JitValue::Sint8(x) => vec![(*x).into()],
        JitValue::Sint8(x) => vec![(*x).into()],
//        JitValue::Sint16(x) => vec![(*x).into()],
        JitValue::Sint16(x) => vec![(*x).into()],
//        JitValue::Sint32(x) => vec![(*x).into()],
        JitValue::Sint32(x) => vec![(*x).into()],
//        JitValue::Sint64(x) => vec![(*x).into()],
        JitValue::Sint64(x) => vec![(*x).into()],
//        JitValue::Sint128(x) => vec![(*x).into()],
        JitValue::Sint128(x) => vec![(*x).into()],
//        JitValue::Null => vec![0.into()],
        JitValue::Null => vec![0.into()],
//        JitValue::EcPoint(_, _)
        JitValue::EcPoint(_, _)
//        | JitValue::EcState(_, _, _, _)
        | JitValue::EcState(_, _, _, _)
//        | JitValue::Secp256K1Point { .. }
        | JitValue::Secp256K1Point { .. }
//        | JitValue::Secp256R1Point { .. }
        | JitValue::Secp256R1Point { .. }
//        | JitValue::Felt252Dict { .. } => todo!(),
        | JitValue::Felt252Dict { .. } => todo!(),
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod tests {
mod tests {
//    use super::*;
    use super::*;
//    use cairo_felt::Felt252;
    use cairo_felt::Felt252;
//    use cairo_lang_sierra::ProgramParser;
    use cairo_lang_sierra::ProgramParser;
//

//    #[test]
    #[test]
//    fn test_check_compiler_path() {
    fn test_check_compiler_path() {
//        // Define file, folder, and invalid paths for testing
        // Define file, folder, and invalid paths for testing
//        let file_path = Path::new("src/bin/cairo-native-run.rs");
        let file_path = Path::new("src/bin/cairo-native-run.rs");
//        let folder_path = Path::new("src/bin");
        let folder_path = Path::new("src/bin");
//        let invalid_path = Path::new("src/non-existing-file.rs");
        let invalid_path = Path::new("src/non-existing-file.rs");
//

//        // Test when single_file is true and the path is a file
        // Test when single_file is true and the path is a file
//        assert!(check_compiler_path(true, file_path).is_ok());
        assert!(check_compiler_path(true, file_path).is_ok());
//

//        // Test when single_file is false and the path is a file
        // Test when single_file is false and the path is a file
//        assert!(check_compiler_path(false, file_path).is_err());
        assert!(check_compiler_path(false, file_path).is_err());
//

//        // Test when single_file is true and the path is a folder
        // Test when single_file is true and the path is a folder
//        assert!(check_compiler_path(true, folder_path).is_err());
        assert!(check_compiler_path(true, folder_path).is_err());
//

//        // Test when single_file is false and the path is a folder
        // Test when single_file is false and the path is a folder
//        assert!(check_compiler_path(false, folder_path).is_ok());
        assert!(check_compiler_path(false, folder_path).is_ok());
//

//        // Test when single_file is true and the path does not exist
        // Test when single_file is true and the path does not exist
//        assert!(check_compiler_path(true, invalid_path).is_err());
        assert!(check_compiler_path(true, invalid_path).is_err());
//

//        // Test when single_file is false and the path does not exist
        // Test when single_file is false and the path does not exist
//        assert!(check_compiler_path(false, invalid_path).is_err());
        assert!(check_compiler_path(false, invalid_path).is_err());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_find_function() {
    fn test_find_function() {
//        // Parse a simple program containing a function named "Func2"
        // Parse a simple program containing a function named "Func2"
//        let program = ProgramParser::new().parse("Func2@6() -> ();").unwrap();
        let program = ProgramParser::new().parse("Func2@6() -> ();").unwrap();
//

//        // Assert that the function "Func2" is found and returned correctly
        // Assert that the function "Func2" is found and returned correctly
//        assert_eq!(
        assert_eq!(
//            find_function(&program, "Func2").unwrap(),
            find_function(&program, "Func2").unwrap(),
//            program.funcs.first().unwrap()
            program.funcs.first().unwrap()
//        );
        );
//

//        // Assert that an error is returned when trying to find a non-existing function "Func3"
        // Assert that an error is returned when trying to find a non-existing function "Func3"
//        assert!(find_function(&program, "Func3").is_err());
        assert!(find_function(&program, "Func3").is_err());
//

//        // Assert that an error is returned when trying to find a function in an empty program
        // Assert that an error is returned when trying to find a function in an empty program
//        assert!(find_function(&ProgramParser::new().parse("").unwrap(), "Func2").is_err());
        assert!(find_function(&ProgramParser::new().parse("").unwrap(), "Func2").is_err());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_result_to_runresult_enum_nonpanic() {
    fn test_result_to_runresult_enum_nonpanic() {
//        // Tests the conversion of a non-panic enum result to a `RunResultValue::Success`.
        // Tests the conversion of a non-panic enum result to a `RunResultValue::Success`.
//        assert_eq!(
        assert_eq!(
//            result_to_runresult(&ExecutionResult {
            result_to_runresult(&ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: JitValue::Enum {
                return_value: JitValue::Enum {
//                    tag: 34,
                    tag: 34,
//                    value: JitValue::Array(vec![
                    value: JitValue::Array(vec![
//                        JitValue::Felt252(42.into()),
                        JitValue::Felt252(42.into()),
//                        JitValue::Uint8(100),
                        JitValue::Uint8(100),
//                        JitValue::Uint128(1000),
                        JitValue::Uint128(1000),
//                    ])
                    ])
//                    .into(),
                    .into(),
//                    debug_name: Some("debug_name".into()),
                    debug_name: Some("debug_name".into()),
//                },
                },
//                builtin_stats: Default::default(),
                builtin_stats: Default::default(),
//            })
            })
//            .unwrap(),
            .unwrap(),
//            RunResultValue::Success(vec![
            RunResultValue::Success(vec![
//                Felt252::from(34),
                Felt252::from(34),
//                Felt252::from(42),
                Felt252::from(42),
//                Felt252::from(100),
                Felt252::from(100),
//                Felt252::from(1000)
                Felt252::from(1000)
//            ])
            ])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_result_to_runresult_success() {
    fn test_result_to_runresult_success() {
//        // Tests the conversion of a success enum result to a `RunResultValue::Success`.
        // Tests the conversion of a success enum result to a `RunResultValue::Success`.
//        assert_eq!(
        assert_eq!(
//            result_to_runresult(&ExecutionResult {
            result_to_runresult(&ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: JitValue::Enum {
                return_value: JitValue::Enum {
//                    tag: 0,
                    tag: 0,
//                    value: JitValue::Uint64(24).into(),
                    value: JitValue::Uint64(24).into(),
//                    debug_name: Some("core::panics::PanicResult::Test".into()),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
//                },
                },
//                builtin_stats: Default::default(),
                builtin_stats: Default::default(),
//            })
            })
//            .unwrap(),
            .unwrap(),
//            RunResultValue::Success(vec![Felt252::from(24)])
            RunResultValue::Success(vec![Felt252::from(24)])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    #[should_panic(expected = "unsuported return value in cairo-native")]
    #[should_panic(expected = "unsuported return value in cairo-native")]
//    fn test_result_to_runresult_panic() {
    fn test_result_to_runresult_panic() {
//        // Tests the conversion with unsuported return value.
        // Tests the conversion with unsuported return value.
//        let _ = result_to_runresult(&ExecutionResult {
        let _ = result_to_runresult(&ExecutionResult {
//            remaining_gas: None,
            remaining_gas: None,
//            return_value: JitValue::Enum {
            return_value: JitValue::Enum {
//                tag: 10,
                tag: 10,
//                value: JitValue::Uint64(24).into(),
                value: JitValue::Uint64(24).into(),
//                debug_name: Some("core::panics::PanicResult::Test".into()),
                debug_name: Some("core::panics::PanicResult::Test".into()),
//            },
            },
//            builtin_stats: Default::default(),
            builtin_stats: Default::default(),
//        })
        })
//        .unwrap();
        .unwrap();
//    }
    }
//

//    #[test]
    #[test]
//    #[should_panic(expected = "missing debug name")]
    #[should_panic(expected = "missing debug name")]
//    fn test_result_to_runresult_missing_debug_name() {
    fn test_result_to_runresult_missing_debug_name() {
//        // Tests the conversion with no debug name.
        // Tests the conversion with no debug name.
//        let _ = result_to_runresult(&ExecutionResult {
        let _ = result_to_runresult(&ExecutionResult {
//            remaining_gas: None,
            remaining_gas: None,
//            return_value: JitValue::Enum {
            return_value: JitValue::Enum {
//                tag: 10,
                tag: 10,
//                value: JitValue::Uint64(24).into(),
                value: JitValue::Uint64(24).into(),
//                debug_name: None,
                debug_name: None,
//            },
            },
//            builtin_stats: Default::default(),
            builtin_stats: Default::default(),
//        })
        })
//        .unwrap();
        .unwrap();
//    }
    }
//

//    #[test]
    #[test]
//    fn test_result_to_runresult_return() {
    fn test_result_to_runresult_return() {
//        // Tests the conversion of a panic enum result with non-zero tag to a `RunResultValue::Panic`.
        // Tests the conversion of a panic enum result with non-zero tag to a `RunResultValue::Panic`.
//        assert_eq!(
        assert_eq!(
//            result_to_runresult(&ExecutionResult {
            result_to_runresult(&ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: JitValue::Enum {
                return_value: JitValue::Enum {
//                    tag: 10,
                    tag: 10,
//                    value: JitValue::Struct {
                    value: JitValue::Struct {
//                        fields: vec![
                        fields: vec![
//                            JitValue::Felt252(42.into()),
                            JitValue::Felt252(42.into()),
//                            JitValue::Uint8(100),
                            JitValue::Uint8(100),
//                            JitValue::Uint128(1000),
                            JitValue::Uint128(1000),
//                        ],
                        ],
//                        debug_name: Some("debug_name".into()),
                        debug_name: Some("debug_name".into()),
//                    }
                    }
//                    .into(),
                    .into(),
//                    debug_name: Some("core::panics::PanicResult::Test".into()),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
//                },
                },
//                builtin_stats: Default::default(),
                builtin_stats: Default::default(),
//            })
            })
//            .unwrap(),
            .unwrap(),
//            RunResultValue::Panic(vec![
            RunResultValue::Panic(vec![
//                Felt252::from(42),
                Felt252::from(42),
//                Felt252::from(100),
                Felt252::from(100),
//                Felt252::from(1000)
                Felt252::from(1000)
//            ])
            ])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_result_to_runresult_non_enum() {
    fn test_result_to_runresult_non_enum() {
//        // Tests the conversion of a non-enum result to a `RunResultValue::Success`.
        // Tests the conversion of a non-enum result to a `RunResultValue::Success`.
//        assert_eq!(
        assert_eq!(
//            result_to_runresult(&ExecutionResult {
            result_to_runresult(&ExecutionResult {
//                remaining_gas: None,
                remaining_gas: None,
//                return_value: JitValue::Uint8(10),
                return_value: JitValue::Uint8(10),
//                builtin_stats: Default::default(),
                builtin_stats: Default::default(),
//            })
            })
//            .unwrap(),
            .unwrap(),
//            RunResultValue::Success(vec![Felt252::from(10)])
            RunResultValue::Success(vec![Felt252::from(10)])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_felt252() {
    fn test_jitvalue_to_felt_felt252() {
//        let felt_value: Felt = 42.into();
        let felt_value: Felt = 42.into();
//

//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Felt252(felt_value)),
            jitvalue_to_felt(&JitValue::Felt252(felt_value)),
//            vec![felt_value]
            vec![felt_value]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_array() {
    fn test_jitvalue_to_felt_array() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Array(vec![
            jitvalue_to_felt(&JitValue::Array(vec![
//                JitValue::Felt252(42.into()),
                JitValue::Felt252(42.into()),
//                JitValue::Uint8(100),
                JitValue::Uint8(100),
//                JitValue::Uint128(1000),
                JitValue::Uint128(1000),
//            ])),
            ])),
//            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_struct() {
    fn test_jitvalue_to_felt_struct() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Struct {
            jitvalue_to_felt(&JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Felt252(42.into()),
                    JitValue::Felt252(42.into()),
//                    JitValue::Uint8(100),
                    JitValue::Uint8(100),
//                    JitValue::Uint128(1000)
                    JitValue::Uint128(1000)
//                ],
                ],
//                debug_name: Some("debug_name".into())
                debug_name: Some("debug_name".into())
//            }),
            }),
//            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_enum() {
    fn test_jitvalue_to_felt_enum() {
//        // With debug name
        // With debug name
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Enum {
            jitvalue_to_felt(&JitValue::Enum {
//                tag: 34,
                tag: 34,
//                value: JitValue::Array(vec![
                value: JitValue::Array(vec![
//                    JitValue::Felt252(42.into()),
                    JitValue::Felt252(42.into()),
//                    JitValue::Uint8(100),
                    JitValue::Uint8(100),
//                    JitValue::Uint128(1000),
                    JitValue::Uint128(1000),
//                ])
                ])
//                .into(),
                .into(),
//                debug_name: Some("debug_name".into())
                debug_name: Some("debug_name".into())
//            }),
            }),
//            vec![
            vec![
//                Felt::from(34),
                Felt::from(34),
//                Felt::from(42),
                Felt::from(42),
//                Felt::from(100),
                Felt::from(100),
//                Felt::from(1000)
                Felt::from(1000)
//            ]
            ]
//        );
        );
//

//        // With core::bool debug name and tag 1
        // With core::bool debug name and tag 1
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Enum {
            jitvalue_to_felt(&JitValue::Enum {
//                tag: 1,
                tag: 1,
//                value: JitValue::Uint128(1000).into(),
                value: JitValue::Uint128(1000).into(),
//                debug_name: Some("core::bool".into())
                debug_name: Some("core::bool".into())
//            }),
            }),
//            vec![Felt::ONE]
            vec![Felt::ONE]
//        );
        );
//

//        // With core::bool debug name and tag not 1
        // With core::bool debug name and tag not 1
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Enum {
            jitvalue_to_felt(&JitValue::Enum {
//                tag: 10,
                tag: 10,
//                value: JitValue::Uint128(1000).into(),
                value: JitValue::Uint128(1000).into(),
//                debug_name: Some("core::bool".into())
                debug_name: Some("core::bool".into())
//            }),
            }),
//            vec![Felt::ZERO]
            vec![Felt::ZERO]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_u8() {
    fn test_jitvalue_to_felt_u8() {
//        assert_eq!(jitvalue_to_felt(&JitValue::Uint8(10)), vec![Felt::from(10)]);
        assert_eq!(jitvalue_to_felt(&JitValue::Uint8(10)), vec![Felt::from(10)]);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_u16() {
    fn test_jitvalue_to_felt_u16() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Uint16(100)),
            jitvalue_to_felt(&JitValue::Uint16(100)),
//            vec![Felt::from(100)]
            vec![Felt::from(100)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_u32() {
    fn test_jitvalue_to_felt_u32() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Uint32(1000)),
            jitvalue_to_felt(&JitValue::Uint32(1000)),
//            vec![Felt::from(1000)]
            vec![Felt::from(1000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_u64() {
    fn test_jitvalue_to_felt_u64() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Uint64(10000)),
            jitvalue_to_felt(&JitValue::Uint64(10000)),
//            vec![Felt::from(10000)]
            vec![Felt::from(10000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_u128() {
    fn test_jitvalue_to_felt_u128() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Uint128(100000)),
            jitvalue_to_felt(&JitValue::Uint128(100000)),
//            vec![Felt::from(100000)]
            vec![Felt::from(100000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_sint8() {
    fn test_jitvalue_to_felt_sint8() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Sint8(-10)),
            jitvalue_to_felt(&JitValue::Sint8(-10)),
//            vec![Felt::from(-10)]
            vec![Felt::from(-10)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_sint16() {
    fn test_jitvalue_to_felt_sint16() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Sint16(-100)),
            jitvalue_to_felt(&JitValue::Sint16(-100)),
//            vec![Felt::from(-100)]
            vec![Felt::from(-100)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_sint32() {
    fn test_jitvalue_to_felt_sint32() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Sint32(-1000)),
            jitvalue_to_felt(&JitValue::Sint32(-1000)),
//            vec![Felt::from(-1000)]
            vec![Felt::from(-1000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_sint64() {
    fn test_jitvalue_to_felt_sint64() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Sint64(-10000)),
            jitvalue_to_felt(&JitValue::Sint64(-10000)),
//            vec![Felt::from(-10000)]
            vec![Felt::from(-10000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_sint128() {
    fn test_jitvalue_to_felt_sint128() {
//        assert_eq!(
        assert_eq!(
//            jitvalue_to_felt(&JitValue::Sint128(-100000)),
            jitvalue_to_felt(&JitValue::Sint128(-100000)),
//            vec![Felt::from(-100000)]
            vec![Felt::from(-100000)]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jitvalue_to_felt_null() {
    fn test_jitvalue_to_felt_null() {
//        assert_eq!(jitvalue_to_felt(&JitValue::Null), vec![Felt::ZERO]);
        assert_eq!(jitvalue_to_felt(&JitValue::Null), vec![Felt::ZERO]);
//    }
    }
//}
}
