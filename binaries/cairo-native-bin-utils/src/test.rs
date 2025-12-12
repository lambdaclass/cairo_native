use super::{find_function, format_for_panic, result_to_runresult, RunArgs};
use anyhow::Context;
use cairo_lang_runner::{RunResultValue, SierraCasmRunner};
use cairo_lang_sierra::{extensions::gas::CostTokenType, ids::FunctionId, program::Program};
use cairo_lang_sierra_to_casm::metadata::MetadataComputationConfig;
use cairo_lang_sierra_type_size::ProgramRegistryInfo;
use cairo_lang_starknet::contract::ContractInfo;
use cairo_lang_test_plugin::{
    test_config::{PanicExpectation, TestExpectation},
    TestConfig,
};
use cairo_lang_test_plugin::{TestCompilation, TestCompilationMetadata};
use cairo_lang_utils::{
    casts::IntoOrPanic, ordered_hash_map::OrderedHashMap, small_ordered_map::SmallOrderedMap,
};
use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, metadata::gas::GasMetadata,
    starknet_stub::StubSyscallHandler,
};
use colored::Colorize;
use itertools::Itertools;
use num_traits::ToPrimitive;
#[cfg(feature = "scarb")]
use scarb_metadata::{PackageMetadata, TargetMetadata};
use starknet_types_core::felt::Felt;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Summary data of the ran tests.
pub struct TestsSummary {
    pub passed: Vec<String>,
    pub failed: Vec<String>,
    pub ignored: Vec<String>,
    pub mismatch: Vec<String>,
    pub failed_run_results: Vec<RunResultValue>,
    pub mismatch_reason: Vec<String>,
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
    Mismatch(String),
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
    let total_tests_count = compiled.metadata.named_tests.len();
    let named_tests = compiled
        .metadata
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
        metadata: TestCompilationMetadata {
            named_tests,
            ..compiled.metadata
        },
        ..compiled
    };
    (tests, filtered_out)
}

/// Display the summary of the ran tests.
pub fn display_tests_summary(summary: &TestsSummary, filtered_out: usize) {
    println!();

    if !summary.failed.is_empty() || !summary.mismatch.is_empty() {
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
        for (test_name, mismatch_reason) in summary
            .mismatch
            .iter()
            .zip_eq(summary.mismatch_reason.clone())
        {
            println!("   {test_name} - {mismatch_reason}");
        }
        println!();
    }

    println!(
        "test result: {}. {} passed; {} failed; {} ignored; {} filtered out;",
        if summary.failed.is_empty() && summary.mismatch.is_empty() {
            "ok".bright_green()
        } else {
            "FAILED".bright_red()
        },
        summary.passed.len(),
        summary.failed.len() + summary.mismatch.len(),
        summary.ignored.len(),
        filtered_out
    );
    println!();
}

/// Runs the tests and process the results for a summary.
pub fn run_tests(
    named_tests: Vec<(String, TestConfig)>,
    sierra_program: Program,
    function_set_costs: OrderedHashMap<FunctionId, SmallOrderedMap<CostTokenType, i32>>,
    contracts_info: OrderedHashMap<Felt, ContractInfo>,
    args: RunArgs,
) -> anyhow::Result<TestsSummary> {
    let runner = if args.compare_with_vm {
        Some(
            SierraCasmRunner::new(
                sierra_program.clone(),
                Some(MetadataComputationConfig {
                    function_set_costs: function_set_costs.clone(),
                    linear_gas_solver: true,
                    linear_ap_change_solver: true,
                    skip_non_linear_solver_comparisons: false,
                    compute_runtime_costs: false,
                }),
                contracts_info.clone(),
                None,
            )
            .with_context(|| "Failed setting up runner.")?,
        )
    } else {
        None
    };

    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile(&sierra_program, false, Some(Default::default()), None)
        .unwrap();

    let native_executor = Arc::new(AotNativeExecutor::from_native_module(
        native_module,
        args.opt_level.into(),
    )?);

    let gas_metadata = GasMetadata::new(
        &sierra_program,
        &ProgramRegistryInfo::new(&sierra_program)?,
        Some(MetadataComputationConfig {
            function_set_costs,
            linear_ap_change_solver: true,
            linear_gas_solver: true,
            skip_non_linear_solver_comparisons: false,
            compute_runtime_costs: false,
        }),
    )
    .unwrap();

    println!("running {} tests", named_tests.len());
    let wrapped_summary = Mutex::new(Ok(TestsSummary {
        passed: vec![],
        failed: vec![],
        ignored: vec![],
        mismatch: vec![],
        failed_run_results: vec![],
        mismatch_reason: vec![],
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

                let syscall_handler = &mut StubSyscallHandler {
                    contracts_info: contracts_info.clone(),
                    executor: Some(native_executor.clone()),
                    ..Default::default()
                };
                let native_result = native_executor.invoke_dynamic_with_syscall_handler(
                    &func.id,
                    &[],
                    initial_gas,
                    syscall_handler,
                )?;
                let run_result = result_to_runresult(&native_result)?;

                let gas_usage = test
                    .available_gas
                    .zip(native_result.remaining_gas)
                    .map(|(before, after)| before.into_or_panic::<i64>() - after.to_i64().unwrap())
                    .or_else(|| {
                        gas_metadata
                            .initial_required_gas(&func.id)
                            .unwrap()
                            .map(|gas| gas.try_into().unwrap())
                    });

                if let Some(runner) = &runner {
                    let vm_result = runner
                        .run_function_with_starknet_context(
                            func,
                            vec![],
                            test.available_gas,
                            Default::default(),
                        )
                        .with_context(|| {
                            format!("Failed to run the function `{}`.", name.as_str())
                        })?;

                    let vm_builtins: HashMap<&str, usize> = vm_result
                        .used_resources
                        .basic_resources
                        .filter_unused_builtins()
                        .builtin_instance_counter
                        .iter()
                        .map(|(k, v)| (k.to_str(), *v))
                        .collect();

                    let native_builtins = {
                        let mut native_builtins = HashMap::new();
                        native_builtins
                            .insert("range_check", native_result.builtin_stats.range_check);
                        native_builtins.insert("pedersen", native_result.builtin_stats.pedersen);
                        native_builtins.insert("bitwise", native_result.builtin_stats.bitwise);
                        native_builtins.insert("ec_op", native_result.builtin_stats.ec_op);
                        native_builtins.insert("poseidon", native_result.builtin_stats.poseidon);
                        // we don't include the segment arena builtin, as its not included in the VM output either.
                        native_builtins
                            .insert("range_check96", native_result.builtin_stats.range_check96);
                        native_builtins.insert("add_mod", native_result.builtin_stats.add_mod);
                        native_builtins.insert("mul_mod", native_result.builtin_stats.mul_mod);

                        for builtin_name in vm_builtins.keys() {
                            native_builtins.entry(builtin_name).or_default();
                        }
                        native_builtins
                    };

                    for (builtin_name, native_counter) in native_builtins {
                        let vm_counter = vm_builtins.get(builtin_name).cloned().unwrap_or_default();

                        if native_counter != vm_counter {
                            return Ok((
                                name,
                                Some(TestResult {
                                    status: TestStatus::Mismatch(format!(
                                        "{} builtin mismatch: expected {}, got {}",
                                        builtin_name, vm_counter, native_counter
                                    )),
                                    gas_usage,
                                }),
                            ));
                        }
                    }
                }

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
                        gas_usage,
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
                Some(TestResult {
                    status: TestStatus::Mismatch(reason),
                    gas_usage,
                }) => {
                    summary.mismatch_reason.push(reason);
                    (&mut summary.mismatch, "mismatch".bright_red(), gas_usage)
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
