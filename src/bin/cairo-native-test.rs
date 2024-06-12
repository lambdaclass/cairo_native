//use anyhow::{bail, Context};
use anyhow::{bail, Context};
//use cairo_felt::Felt252;
use cairo_felt::Felt252;
//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
//};
};
//use cairo_lang_filesystem::cfg::{Cfg, CfgSet};
use cairo_lang_filesystem::cfg::{Cfg, CfgSet};
//use cairo_lang_runner::{casm_run::format_next_item, RunResultValue};
use cairo_lang_runner::{casm_run::format_next_item, RunResultValue};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::gas::CostTokenType,
    extensions::gas::CostTokenType,
//    ids::FunctionId,
    ids::FunctionId,
//    program::{Function, Program},
    program::{Function, Program},
//};
};
//use cairo_lang_starknet::{contract::ContractInfo, starknet_plugin_suite};
use cairo_lang_starknet::{contract::ContractInfo, starknet_plugin_suite};
//use cairo_lang_test_plugin::{
use cairo_lang_test_plugin::{
//    compile_test_prepared_db,
    compile_test_prepared_db,
//    test_config::{PanicExpectation, TestExpectation},
    test_config::{PanicExpectation, TestExpectation},
//    test_plugin_suite, TestCompilation, TestConfig,
    test_plugin_suite, TestCompilation, TestConfig,
//};
};
//use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};
use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};
//use cairo_native::{
use cairo_native::{
//    context::NativeContext,
    context::NativeContext,
//    execution_result::ExecutionResult,
    execution_result::ExecutionResult,
//    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
//    metadata::gas::{GasMetadata, MetadataComputationConfig},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
//    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
//    values::JitValue,
    values::JitValue,
//};
};
//use clap::{Parser, ValueEnum};
use clap::{Parser, ValueEnum};
//use colored::Colorize;
use colored::Colorize;
//use itertools::Itertools;
use itertools::Itertools;
//use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::sec1::ToEncodedPoint;
//use k256::elliptic_curve::{generic_array::GenericArray, sec1::FromEncodedPoint};
use k256::elliptic_curve::{generic_array::GenericArray, sec1::FromEncodedPoint};
//use num_traits::ToPrimitive;
use num_traits::ToPrimitive;
//use sec1::point::Coordinates;
use sec1::point::Coordinates;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::{
use std::{
//    iter::once,
    iter::once,
//    path::{Path, PathBuf},
    path::{Path, PathBuf},
//    vec::IntoIter,
    vec::IntoIter,
//};
};
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

///// Compiles a Cairo project and runs all the functions marked as `#[test]`.
/// Compiles a Cairo project and runs all the functions marked as `#[test]`.
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
//    /// The filter for the tests, running only tests containing the filter string.
    /// The filter for the tests, running only tests containing the filter string.
//    #[arg(short, long, default_value_t = String::default())]
    #[arg(short, long, default_value_t = String::default())]
//    filter: String,
    filter: String,
//    /// Should we run ignored tests as well.
    /// Should we run ignored tests as well.
//    #[arg(long, default_value_t = false)]
    #[arg(long, default_value_t = false)]
//    include_ignored: bool,
    include_ignored: bool,
//    /// Should we run only the ignored tests.
    /// Should we run only the ignored tests.
//    #[arg(long, default_value_t = false)]
    #[arg(long, default_value_t = false)]
//    ignored: bool,
    ignored: bool,
//    /// Should we add the starknet plugin to run the tests.
    /// Should we add the starknet plugin to run the tests.
//    #[arg(long, default_value_t = false)]
    #[arg(long, default_value_t = false)]
//    starknet: bool,
    starknet: bool,
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
//    // Parse command-line arguments.
    // Parse command-line arguments.
//    let args = Args::parse();
    let args = Args::parse();
//

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

//    check_compiler_path(args.single_file, &args.path)?;
    check_compiler_path(args.single_file, &args.path)?;
//

//    let db = &mut {
    let db = &mut {
//        let mut b = RootDatabase::builder();
        let mut b = RootDatabase::builder();
//        b.detect_corelib();
        b.detect_corelib();
//        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
//        b.with_plugin_suite(test_plugin_suite());
        b.with_plugin_suite(test_plugin_suite());
//        if args.starknet {
        if args.starknet {
//            b.with_plugin_suite(starknet_plugin_suite());
            b.with_plugin_suite(starknet_plugin_suite());
//        }
        }
//

//        b.build()?
        b.build()?
//    };
    };
//

//    let main_crate_ids = setup_project(db, Path::new(&args.path))?;
    let main_crate_ids = setup_project(db, Path::new(&args.path))?;
//    let mut reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);
    let mut reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);
//    if args.allow_warnings {
    if args.allow_warnings {
//        reporter = reporter.allow_warnings();
        reporter = reporter.allow_warnings();
//    }
    }
//    if reporter.check(db) {
    if reporter.check(db) {
//        bail!("failed to compile: {}", args.path.display());
        bail!("failed to compile: {}", args.path.display());
//    }
    }
//

//    let db = db.snapshot();
    let db = db.snapshot();
//    let test_crate_ids = main_crate_ids.clone();
    let test_crate_ids = main_crate_ids.clone();
//

//    let build_test_compilation = compile_test_prepared_db(
    let build_test_compilation = compile_test_prepared_db(
//        &db,
        &db,
//        args.starknet,
        args.starknet,
//        main_crate_ids.clone(),
        main_crate_ids.clone(),
//        test_crate_ids.clone(),
        test_crate_ids.clone(),
//    )?;
    )?;
//

//    let (compiled, filtered_out) = filter_test_cases(
    let (compiled, filtered_out) = filter_test_cases(
//        build_test_compilation,
        build_test_compilation,
//        args.include_ignored,
        args.include_ignored,
//        args.ignored,
        args.ignored,
//        args.filter.clone(),
        args.filter.clone(),
//    );
    );
//

//    let TestsSummary {
    let TestsSummary {
//        passed,
        passed,
//        failed,
        failed,
//        ignored,
        ignored,
//        failed_run_results,
        failed_run_results,
//    } = run_tests(
    } = run_tests(
//        compiled.named_tests,
        compiled.named_tests,
//        compiled.sierra_program,
        compiled.sierra_program,
//        compiled.function_set_costs,
        compiled.function_set_costs,
//        compiled.contracts_info,
        compiled.contracts_info,
//        &args,
        &args,
//    )?;
    )?;
//

//    if failed.is_empty() {
    if failed.is_empty() {
//        println!(
        println!(
//            "test result: {}. {} passed; {} failed; {} ignored; {filtered_out} filtered out;",
            "test result: {}. {} passed; {} failed; {} ignored; {filtered_out} filtered out;",
//            "ok".bright_green(),
            "ok".bright_green(),
//            passed.len(),
            passed.len(),
//            failed.len(),
            failed.len(),
//            ignored.len()
            ignored.len()
//        );
        );
//    } else {
    } else {
//        println!("failures:");
        println!("failures:");
//        for (failure, run_result) in failed.iter().zip_eq(failed_run_results) {
        for (failure, run_result) in failed.iter().zip_eq(failed_run_results) {
//            print!("   {failure} - ");
            print!("   {failure} - ");
//            match run_result {
            match run_result {
//                RunResultValue::Success(_) => {
                RunResultValue::Success(_) => {
//                    println!("expected panic but finished successfully.");
                    println!("expected panic but finished successfully.");
//                }
                }
//                RunResultValue::Panic(values) => {
                RunResultValue::Panic(values) => {
//                    println!("{}", format_for_panic(values.into_iter()));
                    println!("{}", format_for_panic(values.into_iter()));
//                }
                }
//            }
            }
//        }
        }
//        println!();
        println!();
//        bail!(
        bail!(
//            "test result: {}. {} passed; {} failed; {} ignored",
            "test result: {}. {} passed; {} failed; {} ignored",
//            "FAILED".bright_red(),
            "FAILED".bright_red(),
//            passed.len(),
            passed.len(),
//            failed.len(),
            failed.len(),
//            ignored.len()
            ignored.len()
//        );
        );
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

///// Formats the given felts as a panic string.
/// Formats the given felts as a panic string.
//fn format_for_panic(mut felts: IntoIter<Felt252>) -> String {
fn format_for_panic(mut felts: IntoIter<Felt252>) -> String {
//    let mut items = Vec::new();
    let mut items = Vec::new();
//    while let Some(item) = format_next_item(&mut felts) {
    while let Some(item) = format_next_item(&mut felts) {
//        items.push(item.quote_if_string());
        items.push(item.quote_if_string());
//    }
    }
//    let panic_values_string = if let [item] = &items[..] {
    let panic_values_string = if let [item] = &items[..] {
//        item.clone()
        item.clone()
//    } else {
    } else {
//        format!("({})", items.join(", "))
        format!("({})", items.join(", "))
//    };
    };
//    format!("Panicked with {panic_values_string}.")
    format!("Panicked with {panic_values_string}.")
//}
}
//

///// Filter compiled test cases with user provided arguments.
/// Filter compiled test cases with user provided arguments.
/////
///
///// # Arguments
/// # Arguments
///// * `compiled` - Compiled test cases with metadata.
/// * `compiled` - Compiled test cases with metadata.
///// * `include_ignored` - Include ignored tests as well.
/// * `include_ignored` - Include ignored tests as well.
///// * `ignored` - Run ignored tests only.l
/// * `ignored` - Run ignored tests only.l
///// * `filter` - Include only tests containing the filter string.
/// * `filter` - Include only tests containing the filter string.
///// # Returns
/// # Returns
///// * (`TestCompilation`, `usize`) - The filtered test cases and the number of filtered out cases.
/// * (`TestCompilation`, `usize`) - The filtered test cases and the number of filtered out cases.
//pub fn filter_test_cases(
pub fn filter_test_cases(
//    compiled: TestCompilation,
    compiled: TestCompilation,
//    include_ignored: bool,
    include_ignored: bool,
//    ignored: bool,
    ignored: bool,
//    filter: String,
    filter: String,
//) -> (TestCompilation, usize) {
) -> (TestCompilation, usize) {
//    let total_tests_count = compiled.named_tests.len();
    let total_tests_count = compiled.named_tests.len();
//    let named_tests = compiled
    let named_tests = compiled
//        .named_tests
        .named_tests
//        .into_iter()
        .into_iter()
//        .filter(|(name, _)| name.contains(&filter));
        .filter(|(name, _)| name.contains(&filter));
//

//    let named_tests = if include_ignored {
    let named_tests = if include_ignored {
//        // enable the ignored tests
        // enable the ignored tests
//        named_tests
        named_tests
//            .into_iter()
            .into_iter()
//            .map(|(name, mut test)| {
            .map(|(name, mut test)| {
//                test.ignored = false;
                test.ignored = false;
//                (name, test)
                (name, test)
//            })
            })
//            .collect_vec()
            .collect_vec()
//    } else if ignored {
    } else if ignored {
//        // filter not ignored tests and enable the remaining ones
        // filter not ignored tests and enable the remaining ones
//        named_tests
        named_tests
//            .into_iter()
            .into_iter()
//            .map(|(name, mut test)| {
            .map(|(name, mut test)| {
//                test.ignored = !test.ignored;
                test.ignored = !test.ignored;
//                (name, test)
                (name, test)
//            })
            })
//            .filter(|(_, test)| !test.ignored)
            .filter(|(_, test)| !test.ignored)
//            .collect_vec()
            .collect_vec()
//    } else {
    } else {
//        named_tests.collect_vec()
        named_tests.collect_vec()
//    };
    };
//

//    let filtered_out = total_tests_count - named_tests.len();
    let filtered_out = total_tests_count - named_tests.len();
//    let tests = TestCompilation {
    let tests = TestCompilation {
//        named_tests,
        named_tests,
//        ..compiled
        ..compiled
//    };
    };
//    (tests, filtered_out)
    (tests, filtered_out)
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
//        bail!("test function not found")
        bail!("test function not found")
//    }
    }
//}
}
//

///// The status of a ran test.
/// The status of a ran test.
//enum TestStatus {
enum TestStatus {
//    Success,
    Success,
//    Fail(RunResultValue),
    Fail(RunResultValue),
//}
}
//

///// The result of a ran test.
/// The result of a ran test.
//struct TestResult {
struct TestResult {
//    /// The status of the run.
    /// The status of the run.
//    status: TestStatus,
    status: TestStatus,
//    /// The gas usage of the run if relevant.
    /// The gas usage of the run if relevant.
//    gas_usage: Option<i64>,
    gas_usage: Option<i64>,
//}
}
//

///// Summary data of the ran tests.
/// Summary data of the ran tests.
//pub struct TestsSummary {
pub struct TestsSummary {
//    passed: Vec<String>,
    passed: Vec<String>,
//    failed: Vec<String>,
    failed: Vec<String>,
//    ignored: Vec<String>,
    ignored: Vec<String>,
//    failed_run_results: Vec<RunResultValue>,
    failed_run_results: Vec<RunResultValue>,
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
//        JitValue::Enum { tag, value, .. } => {
        JitValue::Enum { tag, value, .. } => {
//            is_success = *tag == 0;
            is_success = *tag == 0;
//

//            if !is_success {
            if !is_success {
//                match &**value {
                match &**value {
//                    JitValue::Struct { fields, .. } => {
                    JitValue::Struct { fields, .. } => {
//                        for field in fields {
                        for field in fields {
//                            let felt = jitvalue_to_felt(field);
                            let felt = jitvalue_to_felt(field);
//                            felts.extend(felt);
                            felts.extend(felt);
//                        }
                        }
//                    }
                    }
//                    _ => bail!(
                    _ => bail!(
//                        "unsuported return value in cairo-native (inside enum): {:#?}",
                        "unsuported return value in cairo-native (inside enum): {:#?}",
//                        value
                        value
//                    ),
                    ),
//                }
                }
//            }
            }
//        }
        }
//        value => {
        value => {
//            is_success = true;
            is_success = true;
//            let felt = jitvalue_to_felt(value);
            let felt = jitvalue_to_felt(value);
//            felts.extend(felt);
            felts.extend(felt);
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
//    let mut felts = Vec::new();
    let mut felts = Vec::new();
//    match value {
    match value {
//        JitValue::Felt252(felt) => vec![*felt],
        JitValue::Felt252(felt) => vec![*felt],
//        JitValue::BoundedInt { value, .. } => vec![*value],
        JitValue::BoundedInt { value, .. } => vec![*value],
//        JitValue::Bytes31(_) => todo!(),
        JitValue::Bytes31(_) => todo!(),
//        JitValue::Array(values) => {
        JitValue::Array(values) => {
//            for value in values {
            for value in values {
//                let felt = jitvalue_to_felt(value);
                let felt = jitvalue_to_felt(value);
//                felts.extend(felt);
                felts.extend(felt);
//            }
            }
//            felts
            felts
//        }
        }
//        JitValue::Struct { fields, .. } => {
        JitValue::Struct { fields, .. } => {
//            for field in fields {
            for field in fields {
//                let felt = jitvalue_to_felt(field);
                let felt = jitvalue_to_felt(field);
//                felts.extend(felt);
                felts.extend(felt);
//            }
            }
//            felts
            felts
//        }
        }
//        JitValue::Enum { .. } => todo!(),
        JitValue::Enum { .. } => todo!(),
//        JitValue::Felt252Dict { value, .. } => {
        JitValue::Felt252Dict { value, .. } => {
//            for (key, value) in value {
            for (key, value) in value {
//                felts.push(*key);
                felts.push(*key);
//                let felt = jitvalue_to_felt(value);
                let felt = jitvalue_to_felt(value);
//                felts.extend(felt);
                felts.extend(felt);
//            }
            }
//

//            felts
            felts
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
//        JitValue::EcPoint(_, _) => todo!(),
        JitValue::EcPoint(_, _) => todo!(),
//        JitValue::EcState(_, _, _, _) => todo!(),
        JitValue::EcState(_, _, _, _) => todo!(),
//        JitValue::Secp256K1Point { .. } => todo!(),
        JitValue::Secp256K1Point { .. } => todo!(),
//        JitValue::Secp256R1Point { .. } => todo!(),
        JitValue::Secp256R1Point { .. } => todo!(),
//        JitValue::Null => vec![0.into()],
        JitValue::Null => vec![0.into()],
//    }
    }
//}
}
//

///// Runs the tests and process the results for a summary.
/// Runs the tests and process the results for a summary.
//fn run_tests(
fn run_tests(
//    named_tests: Vec<(String, TestConfig)>,
    named_tests: Vec<(String, TestConfig)>,
//    sierra_program: Program,
    sierra_program: Program,
//    function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
//    _contracts_info: OrderedHashMap<Felt252, ContractInfo>,
    _contracts_info: OrderedHashMap<Felt252, ContractInfo>,
//    args: &Args,
    args: &Args,
//) -> anyhow::Result<TestsSummary> {
) -> anyhow::Result<TestsSummary> {
//    let native_context = NativeContext::new();
    let native_context = NativeContext::new();
//

//    // Compile the sierra program into a MLIR module.
    // Compile the sierra program into a MLIR module.
//    let native_module = native_context
    let native_module = native_context
//        .compile_with_metadata(
        .compile_with_metadata(
//            &sierra_program,
            &sierra_program,
//            MetadataComputationConfig {
            MetadataComputationConfig {
//                function_set_costs: function_set_costs.clone(),
                function_set_costs: function_set_costs.clone(),
//                linear_ap_change_solver: true,
                linear_ap_change_solver: true,
//                linear_gas_solver: true,
                linear_gas_solver: true,
//            },
            },
//        )
        )
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

//    let gas_metadata = GasMetadata::new(
    let gas_metadata = GasMetadata::new(
//        &sierra_program,
        &sierra_program,
//        Some(MetadataComputationConfig {
        Some(MetadataComputationConfig {
//            function_set_costs,
            function_set_costs,
//            linear_ap_change_solver: true,
            linear_ap_change_solver: true,
//            linear_gas_solver: true,
            linear_gas_solver: true,
//        }),
        }),
//    )
    )
//    .unwrap();
    .unwrap();
//

//    println!("running {} tests", named_tests.len());
    println!("running {} tests", named_tests.len());
//    let mut wrapped_summary = Ok(TestsSummary {
    let mut wrapped_summary = Ok(TestsSummary {
//        passed: vec![],
        passed: vec![],
//        failed: vec![],
        failed: vec![],
//        ignored: vec![],
        ignored: vec![],
//        failed_run_results: vec![],
        failed_run_results: vec![],
//    });
    });
//    named_tests
    named_tests
//        .into_iter()
        .into_iter()
//        .map(
        .map(
//            |(name, test)| -> anyhow::Result<(String, Option<TestResult>)> {
            |(name, test)| -> anyhow::Result<(String, Option<TestResult>)> {
//                if test.ignored {
                if test.ignored {
//                    return Ok((name, None));
                    return Ok((name, None));
//                }
                }
//                tracing::trace!("running test {name:?}");
                tracing::trace!("running test {name:?}");
//

//                let func = find_function(&sierra_program, name.as_str())?;
                let func = find_function(&sierra_program, name.as_str())?;
//

//                let initial_gas = test.available_gas.map(|x| x.try_into().unwrap());
                let initial_gas = test.available_gas.map(|x| x.try_into().unwrap());
//

//                let result = native_executor
                let result = native_executor
//                    .invoke_dynamic_with_syscall_handler(
                    .invoke_dynamic_with_syscall_handler(
//                        &func.id,
                        &func.id,
//                        &[],
                        &[],
//                        initial_gas,
                        initial_gas,
//                        TestSyscallHandler,
                        TestSyscallHandler,
//                    )
                    )
//                    .with_context(|| format!("Failed to run the function `{}`.", name.as_str()))?;
                    .with_context(|| format!("Failed to run the function `{}`.", name.as_str()))?;
//

//                let run_result = result_to_runresult(&result)?;
                let run_result = result_to_runresult(&result)?;
//                Ok((
                Ok((
//                    name,
                    name,
//                    Some(TestResult {
                    Some(TestResult {
//                        status: match &run_result {
                        status: match &run_result {
//                            RunResultValue::Success(_) => match test.expectation {
                            RunResultValue::Success(_) => match test.expectation {
//                                TestExpectation::Success => TestStatus::Success,
                                TestExpectation::Success => TestStatus::Success,
//                                TestExpectation::Panics(_) => TestStatus::Fail(run_result),
                                TestExpectation::Panics(_) => TestStatus::Fail(run_result),
//                            },
                            },
//                            RunResultValue::Panic(value) => match test.expectation {
                            RunResultValue::Panic(value) => match test.expectation {
//                                TestExpectation::Success => TestStatus::Fail(run_result),
                                TestExpectation::Success => TestStatus::Fail(run_result),
//                                TestExpectation::Panics(panic_expectation) => {
                                TestExpectation::Panics(panic_expectation) => {
//                                    match panic_expectation {
                                    match panic_expectation {
//                                        PanicExpectation::Exact(expected) if value != &expected => {
                                        PanicExpectation::Exact(expected) if value != &expected => {
//                                            TestStatus::Fail(run_result)
                                            TestStatus::Fail(run_result)
//                                        }
                                        }
//                                        _ => TestStatus::Success,
                                        _ => TestStatus::Success,
//                                    }
                                    }
//                                }
                                }
//                            },
                            },
//                        },
                        },
//                        gas_usage: test
                        gas_usage: test
//                            .available_gas
                            .available_gas
//                            .zip(result.remaining_gas)
                            .zip(result.remaining_gas)
//                            .map(|(before, after)| {
                            .map(|(before, after)| {
//                                before.into_or_panic::<i64>() - after.to_i64().unwrap()
                                before.into_or_panic::<i64>() - after.to_i64().unwrap()
//                            })
                            })
//                            .or_else(|| {
                            .or_else(|| {
//                                gas_metadata
                                gas_metadata
//                                    .initial_required_gas(&func.id)
                                    .initial_required_gas(&func.id)
//                                    .map(|gas| gas.try_into().unwrap())
                                    .map(|gas| gas.try_into().unwrap())
//                            }),
                            }),
//                    }),
                    }),
//                ))
                ))
//            },
            },
//        )
        )
//        .for_each(|r| {
        .for_each(|r| {
//            let (name, status) = match r {
            let (name, status) = match r {
//                Ok((name, status)) => (name, status),
                Ok((name, status)) => (name, status),
//                Err(err) => {
                Err(err) => {
//                    wrapped_summary = Err(err);
                    wrapped_summary = Err(err);
//                    return;
                    return;
//                }
                }
//            };
            };
//            let summary = wrapped_summary.as_mut().unwrap();
            let summary = wrapped_summary.as_mut().unwrap();
//            let (res_type, status_str, gas_usage) = match status {
            let (res_type, status_str, gas_usage) = match status {
//                Some(TestResult {
                Some(TestResult {
//                    status: TestStatus::Success,
                    status: TestStatus::Success,
//                    gas_usage,
                    gas_usage,
//                }) => (&mut summary.passed, "ok".bright_green(), gas_usage),
                }) => (&mut summary.passed, "ok".bright_green(), gas_usage),
//                Some(TestResult {
                Some(TestResult {
//                    status: TestStatus::Fail(run_result),
                    status: TestStatus::Fail(run_result),
//                    gas_usage,
                    gas_usage,
//                }) => {
                }) => {
//                    summary.failed_run_results.push(run_result);
                    summary.failed_run_results.push(run_result);
//                    (&mut summary.failed, "fail".bright_red(), gas_usage)
                    (&mut summary.failed, "fail".bright_red(), gas_usage)
//                }
                }
//                None => (&mut summary.ignored, "ignored".bright_yellow(), None),
                None => (&mut summary.ignored, "ignored".bright_yellow(), None),
//            };
            };
//            if let Some(gas_usage) = gas_usage {
            if let Some(gas_usage) = gas_usage {
//                println!("test {name} ... {status_str} (gas usage est.: {gas_usage})");
                println!("test {name} ... {status_str} (gas usage est.: {gas_usage})");
//            } else {
            } else {
//                println!("test {name} ... {status_str}");
                println!("test {name} ... {status_str}");
//            }
            }
//            res_type.push(name);
            res_type.push(name);
//        });
        });
//    wrapped_summary
    wrapped_summary
//}
}
//

//pub struct TestSyscallHandler;
pub struct TestSyscallHandler;
//

//impl StarknetSyscallHandler for TestSyscallHandler {
impl StarknetSyscallHandler for TestSyscallHandler {
//    fn get_block_hash(
    fn get_block_hash(
//        &mut self,
        &mut self,
//        _block_number: u64,
        _block_number: u64,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info(
    fn get_execution_info(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info_v2(
    fn get_execution_info_v2(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn deploy(
    fn deploy(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _contract_address_salt: Felt,
        _contract_address_salt: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _deploy_from_zero: bool,
        _deploy_from_zero: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(Felt, Vec<Felt>)> {
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn library_call(
    fn library_call(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _function_selector: Felt,
        _function_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn call_contract(
    fn call_contract(
//        &mut self,
        &mut self,
//        _address: Felt,
        _address: Felt,
//        _entry_point_selector: Felt,
        _entry_point_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_read(
    fn storage_read(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_write(
    fn storage_write(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _value: Felt,
        _value: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn emit_event(
    fn emit_event(
//        &mut self,
        &mut self,
//        _keys: &[Felt],
        _keys: &[Felt],
//        _data: &[Felt],
        _data: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        _to_address: Felt,
        _to_address: Felt,
//        _payload: &[Felt],
        _payload: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn keccak(&mut self, input: &[u64], gas: &mut u128) -> SyscallResult<U256> {
    fn keccak(&mut self, input: &[u64], gas: &mut u128) -> SyscallResult<U256> {
//        let length = input.len();
        let length = input.len();
//

//        if length % 17 != 0 {
        if length % 17 != 0 {
//            let error_msg = b"Invalid keccak input size";
            let error_msg = b"Invalid keccak input size";
//            let felt_error = Felt::from_bytes_be_slice(error_msg);
            let felt_error = Felt::from_bytes_be_slice(error_msg);
//            return Err(vec![felt_error]);
            return Err(vec![felt_error]);
//        }
        }
//

//        let n_chunks = length / 17;
        let n_chunks = length / 17;
//        let mut state = [0u64; 25];
        let mut state = [0u64; 25];
//

//        for i in 0..n_chunks {
        for i in 0..n_chunks {
//            if *gas < KECCAK_ROUND_COST {
            if *gas < KECCAK_ROUND_COST {
//                let error_msg = b"Syscall out of gas";
                let error_msg = b"Syscall out of gas";
//                let felt_error = Felt::from_bytes_be_slice(error_msg);
                let felt_error = Felt::from_bytes_be_slice(error_msg);
//                return Err(vec![felt_error]);
                return Err(vec![felt_error]);
//            }
            }
//            const KECCAK_ROUND_COST: u128 = 180000;
            const KECCAK_ROUND_COST: u128 = 180000;
//            *gas -= KECCAK_ROUND_COST;
            *gas -= KECCAK_ROUND_COST;
//            let chunk = &input[i * 17..(i + 1) * 17]; //(request.input_start + i * 17)?;
            let chunk = &input[i * 17..(i + 1) * 17]; //(request.input_start + i * 17)?;
//            for (i, val) in chunk.iter().enumerate() {
            for (i, val) in chunk.iter().enumerate() {
//                state[i] ^= val;
                state[i] ^= val;
//            }
            }
//            keccak::f1600(&mut state)
            keccak::f1600(&mut state)
//        }
        }
//

//        // state[0] and state[1] conform the hash_high (u128)
        // state[0] and state[1] conform the hash_high (u128)
//        // state[2] and state[3] conform the hash_low (u128)
        // state[2] and state[3] conform the hash_low (u128)
//        SyscallResult::Ok(U256 {
        SyscallResult::Ok(U256 {
//            lo: state[2] as u128 | ((state[3] as u128) << 64),
            lo: state[2] as u128 | ((state[3] as u128) << 64),
//            hi: state[0] as u128 | ((state[1] as u128) << 64),
            hi: state[0] as u128 | ((state[1] as u128) << 64),
//        })
        })
//    }
    }
//

//    fn secp256k1_new(
    fn secp256k1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        // The following unwraps should be unreachable because the iterator we provide has the
        // The following unwraps should be unreachable because the iterator we provide has the
//        // expected number of bytes.
        // expected number of bytes.
//        let point = k256::ProjectivePoint::from_encoded_point(
        let point = k256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_affine_coordinates(
            &k256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        );
        );
//

//        if bool::from(point.is_some()) {
        if bool::from(point.is_some()) {
//            Ok(Some(Secp256k1Point { x, y }))
            Ok(Some(Secp256k1Point { x, y }))
//        } else {
        } else {
//            Ok(None)
            Ok(None)
//        }
        }
//    }
    }
//

//    fn secp256k1_add(
    fn secp256k1_add(
//        &mut self,
        &mut self,
//        p0: Secp256k1Point,
        p0: Secp256k1Point,
//        p1: Secp256k1Point,
        p1: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // The inner unwraps should be unreachable because the iterator we provide has the expected
//        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
//        // they'll be provided by secp256 syscalls.
        // they'll be provided by secp256 syscalls.
//        let p0 = k256::ProjectivePoint::from_encoded_point(
        let p0 = k256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_affine_coordinates(
            &k256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p0.x.hi
                    p0.x.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p0.x.lo.to_be_bytes()),
                        .chain(p0.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p0.y.hi
                    p0.y.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p0.y.lo.to_be_bytes()),
                        .chain(p0.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//        let p1 = k256::ProjectivePoint::from_encoded_point(
        let p1 = k256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_affine_coordinates(
            &k256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p1.x.hi
                    p1.x.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p1.x.lo.to_be_bytes()),
                        .chain(p1.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p1.y.hi
                    p1.y.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p1.y.lo.to_be_bytes()),
                        .chain(p1.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//

//        let p = p0 + p1;
        let p = p0 + p1;
//

//        let p = p.to_encoded_point(false);
        let p = p.to_encoded_point(false);
//        let (x, y) = match p.coordinates() {
        let (x, y) = match p.coordinates() {
//            Coordinates::Uncompressed { x, y } => (x, y),
            Coordinates::Uncompressed { x, y } => (x, y),
//            _ => {
            _ => {
//                // This should be unreachable because we explicitly asked for the uncompressed
                // This should be unreachable because we explicitly asked for the uncompressed
//                // encoding.
                // encoding.
//                unreachable!()
                unreachable!()
//            }
            }
//        };
        };
//

//        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // The following two unwraps should be safe because the array always has 32 bytes. The other
//        // four are definitely safe because the slicing guarantees its length to be the right one.
        // four are definitely safe because the slicing guarantees its length to be the right one.
//        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
//        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
//        Ok(Secp256k1Point {
        Ok(Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//            },
            },
//        })
        })
//    }
    }
//

//    fn secp256k1_mul(
    fn secp256k1_mul(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        m: U256,
        m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // The inner unwrap should be unreachable because the iterator we provide has the expected
//        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
//        // they'll be provided by secp256 syscalls.
        // they'll be provided by secp256 syscalls.
//        let p = k256::ProjectivePoint::from_encoded_point(
        let p = k256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_affine_coordinates(
            &k256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//        let m: k256::Scalar = k256::elliptic_curve::ScalarPrimitive::from_slice(&{
        let m: k256::Scalar = k256::elliptic_curve::ScalarPrimitive::from_slice(&{
//            let mut buf = [0u8; 32];
            let mut buf = [0u8; 32];
//            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
//            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
//            buf
            buf
//        })
        })
//        .map_err(|_| {
        .map_err(|_| {
//            vec![Felt::from_bytes_be(
            vec![Felt::from_bytes_be(
//                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
//            )]
            )]
//        })?
        })?
//        .into();
        .into();
//

//        let p = p * m;
        let p = p * m;
//

//        let p = p.to_encoded_point(false);
        let p = p.to_encoded_point(false);
//        let (x, y) = match p.coordinates() {
        let (x, y) = match p.coordinates() {
//            Coordinates::Uncompressed { x, y } => (x, y),
            Coordinates::Uncompressed { x, y } => (x, y),
//            _ => {
            _ => {
//                // This should be unreachable because we explicitly asked for the uncompressed
                // This should be unreachable because we explicitly asked for the uncompressed
//                // encoding.
                // encoding.
//                unreachable!()
                unreachable!()
//            }
            }
//        };
        };
//

//        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // The following two unwraps should be safe because the array always has 32 bytes. The other
//        // four are definitely safe because the slicing guarantees its length to be the right one.
        // four are definitely safe because the slicing guarantees its length to be the right one.
//        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
//        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
//        Ok(Secp256k1Point {
        Ok(Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//            },
            },
//        })
        })
//    }
    }
//

//    fn secp256k1_get_point_from_x(
    fn secp256k1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // The inner unwrap should be unreachable because the iterator we provide has the expected
//        // number of bytes. The outer unwrap depends on the encoding format, which should be valid
        // number of bytes. The outer unwrap depends on the encoding format, which should be valid
//        // since it's hardcoded..
        // since it's hardcoded..
//        let point = k256::ProjectivePoint::from_encoded_point(
        let point = k256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_bytes(
            &k256::EncodedPoint::from_bytes(
//                k256::CompressedPoint::from_exact_iter(
                k256::CompressedPoint::from_exact_iter(
//                    once(0x02 | y_parity as u8)
                    once(0x02 | y_parity as u8)
//                        .chain(x.hi.to_be_bytes())
                        .chain(x.hi.to_be_bytes())
//                        .chain(x.lo.to_be_bytes()),
                        .chain(x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//            )
            )
//            .unwrap(),
            .unwrap(),
//        );
        );
//

//        if bool::from(point.is_some()) {
        if bool::from(point.is_some()) {
//            // This unwrap has already been checked in the `if` expression's condition.
            // This unwrap has already been checked in the `if` expression's condition.
//            let p = point.unwrap();
            let p = point.unwrap();
//

//            let p = p.to_encoded_point(false);
            let p = p.to_encoded_point(false);
//            let y = match p.coordinates() {
            let y = match p.coordinates() {
//                Coordinates::Uncompressed { y, .. } => y,
                Coordinates::Uncompressed { y, .. } => y,
//                _ => {
                _ => {
//                    // This should be unreachable because we explicitly asked for the uncompressed
                    // This should be unreachable because we explicitly asked for the uncompressed
//                    // encoding.
                    // encoding.
//                    unreachable!()
                    unreachable!()
//                }
                }
//            };
            };
//

//            // The following unwrap should be safe because the array always has 32 bytes. The other
            // The following unwrap should be safe because the array always has 32 bytes. The other
//            // two are definitely safe because the slicing guarantees its length to be the right
            // two are definitely safe because the slicing guarantees its length to be the right
//            // one.
            // one.
//            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            let y: [u8; 32] = y.as_slice().try_into().unwrap();
//            Ok(Some(Secp256k1Point {
            Ok(Some(Secp256k1Point {
//                x,
                x,
//                y: U256 {
                y: U256 {
//                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//                },
                },
//            }))
            }))
//        } else {
        } else {
//            Ok(None)
            Ok(None)
//        }
        }
//    }
    }
//

//    fn secp256k1_get_xy(
    fn secp256k1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        Ok((p.x, p.y))
        Ok((p.x, p.y))
//    }
    }
//

//    fn secp256r1_new(
    fn secp256r1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        // The following unwraps should be unreachable because the iterator we provide has the
        // The following unwraps should be unreachable because the iterator we provide has the
//        // expected number of bytes.
        // expected number of bytes.
//        let point = p256::ProjectivePoint::from_encoded_point(
        let point = p256::ProjectivePoint::from_encoded_point(
//            &k256::EncodedPoint::from_affine_coordinates(
            &k256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
                    x.hi.to_be_bytes().into_iter().chain(x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
                    y.hi.to_be_bytes().into_iter().chain(y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        );
        );
//

//        if bool::from(point.is_some()) {
        if bool::from(point.is_some()) {
//            Ok(Some(Secp256r1Point { x, y }))
            Ok(Some(Secp256r1Point { x, y }))
//        } else {
        } else {
//            Ok(None)
            Ok(None)
//        }
        }
//    }
    }
//

//    fn secp256r1_add(
    fn secp256r1_add(
//        &mut self,
        &mut self,
//        p0: Secp256r1Point,
        p0: Secp256r1Point,
//        p1: Secp256r1Point,
        p1: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // The inner unwraps should be unreachable because the iterator we provide has the expected
//        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
//        // they'll be provided by secp256 syscalls.
        // they'll be provided by secp256 syscalls.
//        let p0 = p256::ProjectivePoint::from_encoded_point(
        let p0 = p256::ProjectivePoint::from_encoded_point(
//            &p256::EncodedPoint::from_affine_coordinates(
            &p256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p0.x.hi
                    p0.x.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p0.x.lo.to_be_bytes()),
                        .chain(p0.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p0.y.hi
                    p0.y.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p0.y.lo.to_be_bytes()),
                        .chain(p0.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//        let p1 = p256::ProjectivePoint::from_encoded_point(
        let p1 = p256::ProjectivePoint::from_encoded_point(
//            &p256::EncodedPoint::from_affine_coordinates(
            &p256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p1.x.hi
                    p1.x.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p1.x.lo.to_be_bytes()),
                        .chain(p1.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p1.y.hi
                    p1.y.hi
//                        .to_be_bytes()
                        .to_be_bytes()
//                        .into_iter()
                        .into_iter()
//                        .chain(p1.y.lo.to_be_bytes()),
                        .chain(p1.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//

//        let p = p0 + p1;
        let p = p0 + p1;
//

//        let p = p.to_encoded_point(false);
        let p = p.to_encoded_point(false);
//        let (x, y) = match p.coordinates() {
        let (x, y) = match p.coordinates() {
//            Coordinates::Uncompressed { x, y } => (x, y),
            Coordinates::Uncompressed { x, y } => (x, y),
//            _ => {
            _ => {
//                // This should be unreachable because we explicitly asked for the uncompressed
                // This should be unreachable because we explicitly asked for the uncompressed
//                // encoding.
                // encoding.
//                unreachable!()
                unreachable!()
//            }
            }
//        };
        };
//

//        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // The following two unwraps should be safe because the array always has 32 bytes. The other
//        // four are definitely safe because the slicing guarantees its length to be the right one.
        // four are definitely safe because the slicing guarantees its length to be the right one.
//        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
//        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
//        Ok(Secp256r1Point {
        Ok(Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//            },
            },
//        })
        })
//    }
    }
//

//    fn secp256r1_mul(
    fn secp256r1_mul(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        m: U256,
        m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // The inner unwrap should be unreachable because the iterator we provide has the expected
//        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
//        // they'll be provided by secp256 syscalls.
        // they'll be provided by secp256 syscalls.
//        let p = p256::ProjectivePoint::from_encoded_point(
        let p = p256::ProjectivePoint::from_encoded_point(
//            &p256::EncodedPoint::from_affine_coordinates(
            &p256::EncodedPoint::from_affine_coordinates(
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
                    p.x.hi.to_be_bytes().into_iter().chain(p.x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                &GenericArray::from_exact_iter(
                &GenericArray::from_exact_iter(
//                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
                    p.y.hi.to_be_bytes().into_iter().chain(p.y.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//                false,
                false,
//            ),
            ),
//        )
        )
//        .unwrap();
        .unwrap();
//        let m: p256::Scalar = p256::elliptic_curve::ScalarPrimitive::from_slice(&{
        let m: p256::Scalar = p256::elliptic_curve::ScalarPrimitive::from_slice(&{
//            let mut buf = [0u8; 32];
            let mut buf = [0u8; 32];
//            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
            buf[0..16].copy_from_slice(&m.hi.to_be_bytes());
//            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
            buf[16..32].copy_from_slice(&m.lo.to_be_bytes());
//            buf
            buf
//        })
        })
//        .map_err(|_| {
        .map_err(|_| {
//            vec![Felt::from_bytes_be(
            vec![Felt::from_bytes_be(
//                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
//            )]
            )]
//        })?
        })?
//        .into();
        .into();
//

//        let p = p * m;
        let p = p * m;
//

//        let p = p.to_encoded_point(false);
        let p = p.to_encoded_point(false);
//        let (x, y) = match p.coordinates() {
        let (x, y) = match p.coordinates() {
//            Coordinates::Uncompressed { x, y } => (x, y),
            Coordinates::Uncompressed { x, y } => (x, y),
//            _ => {
            _ => {
//                // This should be unreachable because we explicitly asked for the uncompressed
                // This should be unreachable because we explicitly asked for the uncompressed
//                // encoding.
                // encoding.
//                unreachable!()
                unreachable!()
//            }
            }
//        };
        };
//

//        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // The following two unwraps should be safe because the array always has 32 bytes. The other
//        // four are definitely safe because the slicing guarantees its length to be the right one.
        // four are definitely safe because the slicing guarantees its length to be the right one.
//        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
//        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
//        Ok(Secp256r1Point {
        Ok(Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(x[16..32].try_into().unwrap()),
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//            },
            },
//        })
        })
//    }
    }
//

//    fn secp256r1_get_point_from_x(
    fn secp256r1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        let point = p256::ProjectivePoint::from_encoded_point(
        let point = p256::ProjectivePoint::from_encoded_point(
//            &p256::EncodedPoint::from_bytes(
            &p256::EncodedPoint::from_bytes(
//                p256::CompressedPoint::from_exact_iter(
                p256::CompressedPoint::from_exact_iter(
//                    once(0x02 | y_parity as u8)
                    once(0x02 | y_parity as u8)
//                        .chain(x.hi.to_be_bytes())
                        .chain(x.hi.to_be_bytes())
//                        .chain(x.lo.to_be_bytes()),
                        .chain(x.lo.to_be_bytes()),
//                )
                )
//                .unwrap(),
                .unwrap(),
//            )
            )
//            .unwrap(),
            .unwrap(),
//        );
        );
//

//        if bool::from(point.is_some()) {
        if bool::from(point.is_some()) {
//            let p = point.unwrap();
            let p = point.unwrap();
//

//            let p = p.to_encoded_point(false);
            let p = p.to_encoded_point(false);
//            let y = match p.coordinates() {
            let y = match p.coordinates() {
//                Coordinates::Uncompressed { y, .. } => y,
                Coordinates::Uncompressed { y, .. } => y,
//                _ => unreachable!(),
                _ => unreachable!(),
//            };
            };
//

//            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            let y: [u8; 32] = y.as_slice().try_into().unwrap();
//            Ok(Some(Secp256r1Point {
            Ok(Some(Secp256r1Point {
//                x,
                x,
//                y: U256 {
                y: U256 {
//                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    hi: u128::from_be_bytes(y[0..16].try_into().unwrap()),
//                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                    lo: u128::from_be_bytes(y[16..32].try_into().unwrap()),
//                },
                },
//            }))
            }))
//        } else {
        } else {
//            Ok(None)
            Ok(None)
//        }
        }
//    }
    }
//

//    fn secp256r1_get_xy(
    fn secp256r1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        Ok((p.x, p.y))
        Ok((p.x, p.y))
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
//

//    #[test]
    #[test]
//    fn test_secp256k1_get_xy() {
    fn test_secp256k1_get_xy() {
//        let p = Secp256k1Point {
        let p = Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: 331229800296699308591929724809569456681,
                hi: 331229800296699308591929724809569456681,
//                lo: 240848751772479376198639683648735950585,
                lo: 240848751772479376198639683648735950585,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 75181762170223969696219813306313470806,
                hi: 75181762170223969696219813306313470806,
//                lo: 134255467439736302886468555755295925874,
                lo: 134255467439736302886468555755295925874,
//            },
            },
//        };
        };
//

//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler.secp256k1_get_xy(p, &mut 10).unwrap(),
            test_syscall_handler.secp256k1_get_xy(p, &mut 10).unwrap(),
//            (
            (
//                U256 {
                U256 {
//                    hi: 331229800296699308591929724809569456681,
                    hi: 331229800296699308591929724809569456681,
//                    lo: 240848751772479376198639683648735950585,
                    lo: 240848751772479376198639683648735950585,
//                },
                },
//                U256 {
                U256 {
//                    hi: 75181762170223969696219813306313470806,
                    hi: 75181762170223969696219813306313470806,
//                    lo: 134255467439736302886468555755295925874,
                    lo: 134255467439736302886468555755295925874,
//                }
                }
//            )
            )
//        )
        )
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_secp256k1_new() {
    fn test_secp256k1_secp256k1_new() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 {
        let x = U256 {
//            hi: 97179038819393695679,
            hi: 97179038819393695679,
//            lo: 330631467365974629050427735731901850225,
            lo: 330631467365974629050427735731901850225,
//        };
        };
//        let y = U256 {
        let y = U256 {
//            hi: 26163136114030451075775058782541084873,
            hi: 26163136114030451075775058782541084873,
//            lo: 68974579539311638391577168388077592842,
            lo: 68974579539311638391577168388077592842,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler.secp256k1_new(x, y, &mut 10).unwrap(),
            test_syscall_handler.secp256k1_new(x, y, &mut 10).unwrap(),
//            Some(Secp256k1Point { x, y })
            Some(Secp256k1Point { x, y })
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_secp256k1_new_none() {
    fn test_secp256k1_secp256k1_new_none() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 {
        let x = U256 {
//            hi: 97179038819393695679,
            hi: 97179038819393695679,
//            lo: 330631467365974629050427735731901850225,
            lo: 330631467365974629050427735731901850225,
//        };
        };
//        let y = U256 { hi: 0, lo: 0 };
        let y = U256 { hi: 0, lo: 0 };
//

//        assert!(test_syscall_handler
        assert!(test_syscall_handler
//            .secp256k1_new(x, y, &mut 10)
            .secp256k1_new(x, y, &mut 10)
//            .unwrap()
            .unwrap()
//            .is_none());
            .is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_ssecp256k1_add() {
    fn test_secp256k1_ssecp256k1_add() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let p1 = Secp256k1Point {
        let p1 = Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: 161825202758953104525843685720298294023,
                hi: 161825202758953104525843685720298294023,
//                lo: 3468390537006497937951914270391801752,
                lo: 3468390537006497937951914270391801752,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 96009999919712310848645357523629574312,
                hi: 96009999919712310848645357523629574312,
//                lo: 336417762351022071123394393598455764152,
                lo: 336417762351022071123394393598455764152,
//            },
            },
//        };
        };
//

//        let p2 = p1;
        let p2 = p1;
//

//        // 2 * P1
        // 2 * P1
//        let p3 = test_syscall_handler.secp256k1_add(p1, p2, &mut 10).unwrap();
        let p3 = test_syscall_handler.secp256k1_add(p1, p2, &mut 10).unwrap();
//

//        let p1_double = Secp256k1Point {
        let p1_double = Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: 263210499965038831386353541518668627160,
                hi: 263210499965038831386353541518668627160,
//                lo: 122909745026270932982812610085084241637,
                lo: 122909745026270932982812610085084241637,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 35730324229579385338853513728577301230,
                hi: 35730324229579385338853513728577301230,
//                lo: 329597642124196932058042157271922763050,
                lo: 329597642124196932058042157271922763050,
//            },
            },
//        };
        };
//        assert_eq!(p3, p1_double);
        assert_eq!(p3, p1_double);
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256k1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
                .secp256k1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
//                .unwrap(),
                .unwrap(),
//            p1_double
            p1_double
//        );
        );
//

//        // 3 * P1
        // 3 * P1
//        let three_p1 = Secp256k1Point {
        let three_p1 = Secp256k1Point {
//            x: U256 {
            x: U256 {
//                hi: 331229800296699308591929724809569456681,
                hi: 331229800296699308591929724809569456681,
//                lo: 240848751772479376198639683648735950585,
                lo: 240848751772479376198639683648735950585,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 75181762170223969696219813306313470806,
                hi: 75181762170223969696219813306313470806,
//                lo: 134255467439736302886468555755295925874,
                lo: 134255467439736302886468555755295925874,
//            },
            },
//        };
        };
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler.secp256k1_add(p1, p3, &mut 10).unwrap(),
            test_syscall_handler.secp256k1_add(p1, p3, &mut 10).unwrap(),
//            three_p1
            three_p1
//        );
        );
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256k1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
                .secp256k1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
//                .unwrap(),
                .unwrap(),
//            three_p1
            three_p1
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_get_point_from_x_false_yparity() {
    fn test_secp256k1_get_point_from_x_false_yparity() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256k1_get_point_from_x(
                .secp256k1_get_point_from_x(
//                    U256 {
                    U256 {
//                        hi: 97179038819393695679,
                        hi: 97179038819393695679,
//                        lo: 330631467365974629050427735731901850225,
                        lo: 330631467365974629050427735731901850225,
//                    },
                    },
//                    false,
                    false,
//                    &mut 10
                    &mut 10
//                )
                )
//                .unwrap()
                .unwrap()
//                .unwrap(),
                .unwrap(),
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 {
                x: U256 {
//                    hi: 97179038819393695679,
                    hi: 97179038819393695679,
//                    lo: 330631467365974629050427735731901850225,
                    lo: 330631467365974629050427735731901850225,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: 26163136114030451075775058782541084873,
                    hi: 26163136114030451075775058782541084873,
//                    lo: 68974579539311638391577168388077592842
                    lo: 68974579539311638391577168388077592842
//                },
                },
//            }
            }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_get_point_from_x_true_yparity() {
    fn test_secp256k1_get_point_from_x_true_yparity() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256k1_get_point_from_x(
                .secp256k1_get_point_from_x(
//                    U256 {
                    U256 {
//                        hi: 97179038819393695679,
                        hi: 97179038819393695679,
//                        lo: 330631467365974629050427735731901850225,
                        lo: 330631467365974629050427735731901850225,
//                    },
                    },
//                    true,
                    true,
//                    &mut 10
                    &mut 10
//                )
                )
//                .unwrap()
                .unwrap()
//                .unwrap(),
                .unwrap(),
//            Secp256k1Point {
            Secp256k1Point {
//                x: U256 {
                x: U256 {
//                    hi: 97179038819393695679,
                    hi: 97179038819393695679,
//                    lo: 330631467365974629050427735731901850225,
                    lo: 330631467365974629050427735731901850225,
//                },
                },
//                y: U256 {
                y: U256 {
//                    hi: 314119230806908012387599548649227126582,
                    hi: 314119230806908012387599548649227126582,
//                    lo: 271307787381626825071797439039395650341
                    lo: 271307787381626825071797439039395650341
//                },
                },
//            }
            }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256k1_get_point_from_x_none() {
    fn test_secp256k1_get_point_from_x_none() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        assert!(test_syscall_handler
        assert!(test_syscall_handler
//            .secp256k1_get_point_from_x(U256 { hi: 0, lo: 0 }, true, &mut 10)
            .secp256k1_get_point_from_x(U256 { hi: 0, lo: 0 }, true, &mut 10)
//            .unwrap()
            .unwrap()
//            .is_none());
            .is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_new() {
    fn test_secp256r1_new() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 {
        let x = U256 {
//            hi: 97179038819393695679,
            hi: 97179038819393695679,
//            lo: 330631467365974629050427735731901850225,
            lo: 330631467365974629050427735731901850225,
//        };
        };
//        let y = U256 {
        let y = U256 {
//            hi: 118910939004298029402109603132816090461,
            hi: 118910939004298029402109603132816090461,
//            lo: 111045440647474106186537215379882575585,
            lo: 111045440647474106186537215379882575585,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256r1_new(x, y, &mut 10)
                .secp256r1_new(x, y, &mut 10)
//                .unwrap()
                .unwrap()
//                .unwrap(),
                .unwrap(),
//            Secp256r1Point { x, y }
            Secp256r1Point { x, y }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_new_none() {
    fn test_secp256r1_new_none() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 { hi: 0, lo: 0 };
        let x = U256 { hi: 0, lo: 0 };
//        let y = U256 { hi: 0, lo: 0 };
        let y = U256 { hi: 0, lo: 0 };
//

//        assert!(test_syscall_handler
        assert!(test_syscall_handler
//            .secp256r1_new(x, y, &mut 10)
            .secp256r1_new(x, y, &mut 10)
//            .unwrap()
            .unwrap()
//            .is_none());
            .is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_add() {
    fn test_secp256r1_add() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let p1 = Secp256r1Point {
        let p1 = Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: 97179038819393695679,
                hi: 97179038819393695679,
//                lo: 330631467365974629050427735731901850225,
                lo: 330631467365974629050427735731901850225,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 118910939004298029402109603132816090461,
                hi: 118910939004298029402109603132816090461,
//                lo: 111045440647474106186537215379882575585,
                lo: 111045440647474106186537215379882575585,
//            },
            },
//        };
        };
//

//        let p2 = p1;
        let p2 = p1;
//

//        // 2 * P1
        // 2 * P1
//        let p3 = test_syscall_handler.secp256r1_add(p1, p2, &mut 10).unwrap();
        let p3 = test_syscall_handler.secp256r1_add(p1, p2, &mut 10).unwrap();
//

//        let p1_double = Secp256r1Point {
        let p1_double = Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: 280079427190737520201067412903899817878,
                hi: 280079427190737520201067412903899817878,
//                lo: 309339945874468445579793098896656960879,
                lo: 309339945874468445579793098896656960879,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 84249534056490759701994051847937833933,
                hi: 84249534056490759701994051847937833933,
//                lo: 231570843221643745062297421862629788481,
                lo: 231570843221643745062297421862629788481,
//            },
            },
//        };
        };
//        assert_eq!(p3, p1_double);
        assert_eq!(p3, p1_double);
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256r1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
                .secp256r1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
//                .unwrap(),
                .unwrap(),
//            p1_double
            p1_double
//        );
        );
//

//        // 3 * P1
        // 3 * P1
//        let three_p1 = Secp256r1Point {
        let three_p1 = Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: 23850518908906170876551962912581992002,
                hi: 23850518908906170876551962912581992002,
//                lo: 195259625777021303662291420857740525307,
                lo: 195259625777021303662291420857740525307,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 178681203065513270100417145499857169664,
                hi: 178681203065513270100417145499857169664,
//                lo: 282344931843342117515389970197013120959,
                lo: 282344931843342117515389970197013120959,
//            },
            },
//        };
        };
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler.secp256r1_add(p1, p3, &mut 10).unwrap(),
            test_syscall_handler.secp256r1_add(p1, p3, &mut 10).unwrap(),
//            three_p1
            three_p1
//        );
        );
//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256r1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
                .secp256r1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
//                .unwrap(),
                .unwrap(),
//            three_p1
            three_p1
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_get_point_from_x_true_yparity() {
    fn test_secp256r1_get_point_from_x_true_yparity() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 {
        let x = U256 {
//            hi: 97179038819393695679,
            hi: 97179038819393695679,
//            lo: 330631467365974629050427735731901850225,
            lo: 330631467365974629050427735731901850225,
//        };
        };
//

//        let y = U256 {
        let y = U256 {
//            hi: 118910939004298029402109603132816090461,
            hi: 118910939004298029402109603132816090461,
//            lo: 111045440647474106186537215379882575585,
            lo: 111045440647474106186537215379882575585,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256r1_get_point_from_x(x, true, &mut 10)
                .secp256r1_get_point_from_x(x, true, &mut 10)
//                .unwrap()
                .unwrap()
//                .unwrap(),
                .unwrap(),
//            Secp256r1Point { x, y }
            Secp256r1Point { x, y }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_get_point_from_x_false_yparity() {
    fn test_secp256r1_get_point_from_x_false_yparity() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 {
        let x = U256 {
//            hi: 97179038819393695679,
            hi: 97179038819393695679,
//            lo: 330631467365974629050427735731901850225,
            lo: 330631467365974629050427735731901850225,
//        };
        };
//

//        let y = U256 {
        let y = U256 {
//            hi: 221371427837412271565447410779117722274,
            hi: 221371427837412271565447410779117722274,
//            lo: 229236926352692519791101729645429586206,
            lo: 229236926352692519791101729645429586206,
//        };
        };
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler
            test_syscall_handler
//                .secp256r1_get_point_from_x(x, false, &mut 10)
                .secp256r1_get_point_from_x(x, false, &mut 10)
//                .unwrap()
                .unwrap()
//                .unwrap(),
                .unwrap(),
//            Secp256r1Point { x, y }
            Secp256r1Point { x, y }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_get_point_from_x_none() {
    fn test_secp256r1_get_point_from_x_none() {
//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        let x = U256 { hi: 0, lo: 10 };
        let x = U256 { hi: 0, lo: 10 };
//

//        assert!(test_syscall_handler
        assert!(test_syscall_handler
//            .secp256r1_get_point_from_x(x, true, &mut 10)
            .secp256r1_get_point_from_x(x, true, &mut 10)
//            .unwrap()
            .unwrap()
//            .is_none());
            .is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_secp256r1_get_xy() {
    fn test_secp256r1_get_xy() {
//        let p = Secp256r1Point {
        let p = Secp256r1Point {
//            x: U256 {
            x: U256 {
//                hi: 97179038819393695679,
                hi: 97179038819393695679,
//                lo: 330631467365974629050427735731901850225,
                lo: 330631467365974629050427735731901850225,
//            },
            },
//            y: U256 {
            y: U256 {
//                hi: 221371427837412271565447410779117722274,
                hi: 221371427837412271565447410779117722274,
//                lo: 229236926352692519791101729645429586206,
                lo: 229236926352692519791101729645429586206,
//            },
            },
//        };
        };
//

//        let mut test_syscall_handler = TestSyscallHandler {};
        let mut test_syscall_handler = TestSyscallHandler {};
//

//        assert_eq!(
        assert_eq!(
//            test_syscall_handler.secp256r1_get_xy(p, &mut 10).unwrap(),
            test_syscall_handler.secp256r1_get_xy(p, &mut 10).unwrap(),
//            (
            (
//                U256 {
                U256 {
//                    hi: 97179038819393695679,
                    hi: 97179038819393695679,
//                    lo: 330631467365974629050427735731901850225,
                    lo: 330631467365974629050427735731901850225,
//                },
                },
//                U256 {
                U256 {
//                    hi: 221371427837412271565447410779117722274,
                    hi: 221371427837412271565447410779117722274,
//                    lo: 229236926352692519791101729645429586206,
                    lo: 229236926352692519791101729645429586206,
//                }
                }
//            )
            )
//        )
        )
//    }
    }
//}
}
