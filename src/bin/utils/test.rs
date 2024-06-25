use super::{find_function, format_for_panic, result_to_runresult, RunArgs, RunMode};
use anyhow::Context;
use cairo_lang_runner::RunResultValue;
use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::{extensions::gas::CostTokenType, ids::FunctionId};
use cairo_lang_test_plugin::TestCompilation;
use cairo_lang_test_plugin::{
    test_config::{PanicExpectation, TestExpectation},
    TestConfig,
};
use cairo_lang_utils::casts::IntoOrPanic;
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use cairo_native::starknet_stub::StubSyscallHandler;
use cairo_native::{
    context::NativeContext,
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
};
use colored::Colorize;
use itertools::Itertools;
use num_traits::ToPrimitive;
#[cfg(feature = "scarb")]
use scarb_metadata::{PackageMetadata, TargetMetadata};
use std::sync::Mutex;

/// Summary data of the ran tests.
pub struct TestsSummary {
    pub passed: Vec<String>,
    pub failed: Vec<String>,
    pub ignored: Vec<String>,
    pub failed_run_results: Vec<RunResultValue>,
}

/// The result of a ran test.
struct TestResult {
    /// The status of the run.
    status: TestStatus,
    /// The gas usage of the run if relevant.
    gas_usage: Option<i64>,
}

/// The status of a ran test.
enum TestStatus {
    Success,
    Fail(RunResultValue),
}

/// Find all testable targets in the Scarb package.
#[cfg(feature = "scarb")]
pub fn find_testable_targets(package: &PackageMetadata) -> Vec<&TargetMetadata> {
    package
        .targets
        .iter()
        .filter(|target| target.kind == "test")
        .collect()
}

/// Filter compiled test cases with user provided arguments.
///
/// # Arguments
/// * `compiled` - Compiled test cases with metadata.
/// * `include_ignored` - Include ignored tests as well.
/// * `ignored` - Run ignored tests only.
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
        // Filtering unignored tests in `ignored` mode
        .filter(|(_, test)| !ignored || test.ignored || include_ignored)
        .map(|(func, mut test)| {
            // Un-ignoring all the tests in `include-ignored` and `ignored` mode.
            if include_ignored || ignored {
                test.ignored = false;
            }
            (func, test)
        })
        .filter(|(name, _)| name.contains(&filter))
        .collect_vec();
    let filtered_out = total_tests_count - named_tests.len();
    let tests = TestCompilation {
        named_tests,
        ..compiled
    };
    (tests, filtered_out)
}

/// Display the summary of the ran tests.
pub fn display_tests_summary(summary: &TestsSummary, filtered_out: usize) {
    if summary.failed.is_empty() {
        println!(
            "test result: {}. {} passed; {} failed; {} ignored; {filtered_out} filtered out;",
            "ok".bright_green(),
            summary.passed.len(),
            summary.failed.len(),
            summary.ignored.len()
        );
    } else {
        println!("failures:");
        for (failure, run_result) in summary
            .failed
            .iter()
            .zip_eq(summary.failed_run_results.clone())
        {
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
    }
}

/// Runs the tests and process the results for a summary.
pub fn run_tests(
    named_tests: Vec<(String, TestConfig)>,
    sierra_program: Program,
    function_set_costs: OrderedHashMap<FunctionId, OrderedHashMap<CostTokenType, i32>>,
    args: RunArgs,
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

    let native_executor: NativeExecutor = match args.run_mode {
        RunMode::Aot => {
            AotNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
        }
        RunMode::Jit => {
            JitNativeExecutor::from_native_module(native_module, args.opt_level.into()).into()
        }
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

                let initial_gas = test.available_gas.map(|x| x.try_into().unwrap());

                let result = native_executor
                    .invoke_dynamic_with_syscall_handler(
                        &func.id,
                        &[],
                        initial_gas,
                        &mut StubSyscallHandler::default(),
                    )
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
