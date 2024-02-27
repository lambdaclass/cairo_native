use anyhow::{bail, Context};
use cairo_felt::Felt252;
use cairo_lang_compiler::{
    db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
};
use cairo_lang_filesystem::cfg::{Cfg, CfgSet};
use cairo_lang_runner::{casm_run::format_next_item, RunResultValue};
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{Function, Program},
};
use cairo_lang_starknet::{contract::ContractInfo, starknet_plugin_suite};
use cairo_lang_test_plugin::{
    compile_test_prepared_db,
    test_config::{PanicExpectation, TestExpectation},
    test_plugin_suite, TestCompilation, TestConfig,
};
use cairo_lang_utils::{casts::IntoOrPanic, ordered_hash_map::OrderedHashMap};
use cairo_native::{
    context::NativeContext,
    execution_result::ExecutionResult,
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    values::JitValue,
    OptLevel,
};
use clap::{Parser, ValueEnum};
use colored::Colorize;
use itertools::Itertools;
use num_traits::ToPrimitive;
use starknet_types_core::felt::Felt;
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    vec::IntoIter,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Clone, Debug, ValueEnum)]
enum RunMode {
    Aot,
    Jit,
}

/// Compiles a Cairo project and runs all the functions marked as `#[test]`.
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
    /// The filter for the tests, running only tests containing the filter string.
    #[arg(short, long, default_value_t = String::default())]
    filter: String,
    /// Should we run ignored tests as well.
    #[arg(long, default_value_t = false)]
    include_ignored: bool,
    /// Should we run only the ignored tests.
    #[arg(long, default_value_t = false)]
    ignored: bool,
    /// Should we add the starknet plugin to run the tests.
    #[arg(long, default_value_t = false)]
    starknet: bool,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
}

fn main() -> anyhow::Result<()> {
    // Parse command-line arguments.
    let args = Args::parse();

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    check_compiler_path(args.single_file, &args.path)?;

    let db = &mut {
        let mut b = RootDatabase::builder();
        b.detect_corelib();
        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
        b.with_plugin_suite(test_plugin_suite());
        if args.starknet {
            b.with_plugin_suite(starknet_plugin_suite());
        }

        b.build()?
    };

    let main_crate_ids = setup_project(db, Path::new(&args.path))?;
    let mut reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);
    if args.allow_warnings {
        reporter = reporter.allow_warnings();
    }
    if reporter.check(db) {
        bail!("failed to compile: {}", args.path.display());
    }

    let db = db.snapshot();
    let test_crate_ids = main_crate_ids.clone();

    let build_test_compilation = compile_test_prepared_db(
        &db,
        args.starknet,
        main_crate_ids.clone(),
        test_crate_ids.clone(),
    )?;

    let (compiled, filtered_out) = filter_test_cases(
        build_test_compilation,
        args.include_ignored,
        args.ignored,
        args.filter.clone(),
    );

    let TestsSummary {
        passed,
        failed,
        ignored,
        failed_run_results,
    } = run_tests(
        compiled.named_tests,
        compiled.sierra_program,
        compiled.function_set_costs,
        compiled.contracts_info,
        &args,
    )?;

    if failed.is_empty() {
        println!(
            "test result: {}. {} passed; {} failed; {} ignored; {filtered_out} filtered out;",
            "ok".bright_green(),
            passed.len(),
            failed.len(),
            ignored.len()
        );
    } else {
        println!("failures:");
        for (failure, run_result) in failed.iter().zip_eq(failed_run_results) {
            print!("   {failure} - ");
            match run_result {
                RunResultValue::Success(_) => {
                    println!("expected panic but finished successfully.");
                }
                RunResultValue::Panic(values) => {
                    println!("{}", format_for_panic(values.into_iter()));
                }
            }
        }
        println!();
        bail!(
            "test result: {}. {} passed; {} failed; {} ignored",
            "FAILED".bright_red(),
            passed.len(),
            failed.len(),
            ignored.len()
        );
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

/// Formats the given felts as a panic string.
fn format_for_panic(mut felts: IntoIter<Felt252>) -> String {
    let mut items = Vec::new();
    while let Some(item) = format_next_item(&mut felts) {
        items.push(item.quote_if_string());
    }
    let panic_values_string = if let [item] = &items[..] {
        item.clone()
    } else {
        format!("({})", items.join(", "))
    };
    format!("Panicked with {panic_values_string}.")
}

/// Filter compiled test cases with user provided arguments.
///
/// # Arguments
/// * `compiled` - Compiled test cases with metadata.
/// * `include_ignored` - Include ignored tests as well.
/// * `ignored` - Run ignored tests only.l
/// * `filter` - Include only tests containing the filter string.
/// # Returns
/// * (`TestCompilation`, `usize`) - The filtered test cases and the number of filtered out cases.
pub fn filter_test_cases(
    compiled: TestCompilation,
    include_ignored: bool,
    ignored: bool,
    filter: String,
) -> (TestCompilation, usize) {
    let total_tests_count = compiled.named_tests.len();
    let named_tests = compiled
        .named_tests
        .into_iter()
        .map(|(func, mut test)| {
            // Un-ignoring all the tests in `include-ignored` mode.
            if include_ignored {
                test.ignored = false;
            }
            (func, test)
        })
        .filter(|(name, _)| name.contains(&filter))
        // Filtering unignored tests in `ignored` mode
        .filter(|(_, test)| !ignored || test.ignored)
        .collect_vec();
    let filtered_out = total_tests_count - named_tests.len();
    let tests = TestCompilation {
        named_tests,
        ..compiled
    };
    (tests, filtered_out)
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
        bail!("test function not found")
    }
}

/// The status of a ran test.
enum TestStatus {
    Success,
    Fail(RunResultValue),
}

/// The result of a ran test.
struct TestResult {
    /// The status of the run.
    status: TestStatus,
    /// The gas usage of the run if relevant.
    gas_usage: Option<i64>,
}

/// Summary data of the ran tests.
pub struct TestsSummary {
    passed: Vec<String>,
    failed: Vec<String>,
    ignored: Vec<String>,
    failed_run_results: Vec<RunResultValue>,
}

fn result_to_runresult(result: &ExecutionResult) -> anyhow::Result<RunResultValue> {
    let is_success;
    let mut felts: Vec<Felt> = Vec::new();

    match &result.return_value {
        JitValue::Enum { tag, value, .. } => {
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
            }
        }
        _ => bail!("unsuported return value in cairo-native"),
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
        JitValue::Enum { .. } => todo!(),
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

/// Runs the tests and process the results for a summary.
fn run_tests(
    named_tests: Vec<(String, TestConfig)>,
    sierra_program: Program,
    function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    _contracts_info: OrderedHashMap<Felt252, ContractInfo>,
    args: &Args,
) -> anyhow::Result<TestsSummary> {
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile_with_metadata(
            &sierra_program,
            MetadataComputationConfig {
                function_set_costs: function_set_costs.clone(),
                linear_ap_change_solver: true,
                linear_gas_solver: true,
            },
        )
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

    let gas_metadata = GasMetadata::new(
        &sierra_program,
        Some(MetadataComputationConfig {
            function_set_costs,
            linear_ap_change_solver: true,
            linear_gas_solver: true,
        }),
    )
    .unwrap();

    println!("running {} tests", named_tests.len());
    let wrapped_summary = Mutex::new(Ok(TestsSummary {
        passed: vec![],
        failed: vec![],
        ignored: vec![],
        failed_run_results: vec![],
    }));
    named_tests
        .into_iter()
        .map(
            |(name, test)| -> anyhow::Result<(String, Option<TestResult>)> {
                if test.ignored {
                    return Ok((name, None));
                }
                tracing::trace!("running test {name:?}");

                let func = find_function(&sierra_program, name.as_str())?;

                let initial_gas = gas_metadata
                    .get_initial_available_gas(
                        &func.id,
                        test.available_gas.map(|x| x.try_into().unwrap()),
                    )
                    .with_context(|| "not enough gas to run")?;

                let result = native_executor
                    .invoke_dynamic(&func.id, &[], Some(initial_gas), None)
                    .with_context(|| format!("Failed to run the function `{}`.", name.as_str()))?;

                let run_result = result_to_runresult(&result)?;
                Ok((
                    name,
                    Some(TestResult {
                        status: match &run_result {
                            RunResultValue::Success(_) => match test.expectation {
                                TestExpectation::Success => TestStatus::Success,
                                TestExpectation::Panics(_) => TestStatus::Fail(run_result),
                            },
                            RunResultValue::Panic(value) => match test.expectation {
                                TestExpectation::Success => TestStatus::Fail(run_result),
                                TestExpectation::Panics(panic_expectation) => {
                                    match panic_expectation {
                                        PanicExpectation::Exact(expected) if value != &expected => {
                                            TestStatus::Fail(run_result)
                                        }
                                        _ => TestStatus::Success,
                                    }
                                }
                            },
                        },
                        gas_usage: test
                            .available_gas
                            .zip(result.remaining_gas)
                            .map(|(before, after)| {
                                before.into_or_panic::<i64>() - after.to_i64().unwrap()
                            })
                            .or_else(|| {
                                gas_metadata
                                    .initial_required_gas(&func.id)
                                    .map(|gas| gas.try_into().unwrap())
                            }),
                    }),
                ))
            },
        )
        .for_each(|r| {
            let mut wrapped_summary = wrapped_summary.lock().unwrap();
            if wrapped_summary.is_err() {
                return;
            }
            let (name, status) = match r {
                Ok((name, status)) => (name, status),
                Err(err) => {
                    *wrapped_summary = Err(err);
                    return;
                }
            };
            let summary = wrapped_summary.as_mut().unwrap();
            let (res_type, status_str, gas_usage) = match status {
                Some(TestResult {
                    status: TestStatus::Success,
                    gas_usage,
                }) => (&mut summary.passed, "ok".bright_green(), gas_usage),
                Some(TestResult {
                    status: TestStatus::Fail(run_result),
                    gas_usage,
                }) => {
                    summary.failed_run_results.push(run_result);
                    (&mut summary.failed, "fail".bright_red(), gas_usage)
                }
                None => (&mut summary.ignored, "ignored".bright_yellow(), None),
            };
            if let Some(gas_usage) = gas_usage {
                println!("test {name} ... {status_str} (gas usage est.: {gas_usage})");
            } else {
                println!("test {name} ... {status_str}");
            }
            res_type.push(name);
        });
    wrapped_summary.into_inner().unwrap()
}
