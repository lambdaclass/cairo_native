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
    starknet::{Secp256k1Point, Secp256r1Point, StarknetSyscallHandler, SyscallResult, U256},
    values::JitValue,
};
use clap::{Parser, ValueEnum};
use colored::Colorize;
use itertools::Itertools;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::{generic_array::GenericArray, sec1::FromEncodedPoint};
use num_traits::ToPrimitive;
use sec1::point::Coordinates;
use starknet_types_core::felt::Felt;
use std::{
    iter::once,
    path::{Path, PathBuf},
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
        .filter(|(name, _)| name.contains(&filter));

    let named_tests = if include_ignored {
        // enable the ignored tests
        named_tests
            .into_iter()
            .map(|(name, mut test)| {
                test.ignored = false;
                (name, test)
            })
            .collect_vec()
    } else if ignored {
        // filter not ignored tests and enable the remaining ones
        named_tests
            .into_iter()
            .map(|(name, mut test)| {
                test.ignored = !test.ignored;
                (name, test)
            })
            .filter(|(_, test)| !test.ignored)
            .collect_vec()
    } else {
        named_tests.collect_vec()
    };

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
                    _ => bail!(
                        "unsuported return value in cairo-native (inside enum): {:#?}",
                        value
                    ),
                }
            }
        }
        value => {
            is_success = true;
            let felt = jitvalue_to_felt(value);
            felts.extend(felt);
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
        JitValue::Felt252(felt) => vec![*felt],
        JitValue::BoundedInt { value, .. } => vec![*value],
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
        JitValue::Felt252Dict { value, .. } => {
            for (key, value) in value {
                felts.push(*key);
                let felt = jitvalue_to_felt(value);
                felts.extend(felt);
            }

            felts
        }
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
    let mut wrapped_summary = Ok(TestsSummary {
        passed: vec![],
        failed: vec![],
        ignored: vec![],
        failed_run_results: vec![],
    });
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
                        TestSyscallHandler,
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
            let (name, status) = match r {
                Ok((name, status)) => (name, status),
                Err(err) => {
                    wrapped_summary = Err(err);
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
    wrapped_summary
}

pub struct TestSyscallHandler;

impl StarknetSyscallHandler for TestSyscallHandler {
    fn get_block_hash(
        &mut self,
        _block_number: u64,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        unimplemented!()
    }

    fn get_execution_info(
        &mut self,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
        unimplemented!()
    }

    fn get_execution_info_v2(
        &mut self,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
        unimplemented!()
    }

    fn deploy(
        &mut self,
        _class_hash: Felt,
        _contract_address_salt: Felt,
        _calldata: &[Felt],
        _deploy_from_zero: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        unimplemented!()
    }

    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
        unimplemented!()
    }

    fn library_call(
        &mut self,
        _class_hash: Felt,
        _function_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        unimplemented!()
    }

    fn call_contract(
        &mut self,
        _address: Felt,
        _entry_point_selector: Felt,
        _calldata: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        unimplemented!()
    }

    fn storage_read(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        unimplemented!()
    }

    fn storage_write(
        &mut self,
        _address_domain: u32,
        _address: Felt,
        _value: Felt,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn emit_event(
        &mut self,
        _keys: &[Felt],
        _data: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn send_message_to_l1(
        &mut self,
        _to_address: Felt,
        _payload: &[Felt],
        _remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        unimplemented!()
    }

    fn keccak(&mut self, input: &[u64], gas: &mut u128) -> SyscallResult<U256> {
        let length = input.len();

        if length % 17 != 0 {
            let error_msg = b"Invalid keccak input size";
            let felt_error = Felt::from_bytes_be_slice(error_msg);
            return Err(vec![felt_error]);
        }

        let n_chunks = length / 17;
        let mut state = [0u64; 25];

        for i in 0..n_chunks {
            if *gas < KECCAK_ROUND_COST {
                let error_msg = b"Syscall out of gas";
                let felt_error = Felt::from_bytes_be_slice(error_msg);
                return Err(vec![felt_error]);
            }
            const KECCAK_ROUND_COST: u128 = 180000;
            *gas -= KECCAK_ROUND_COST;
            let chunk = &input[i * 17..(i + 1) * 17]; //(request.input_start + i * 17)?;
            for (i, val) in chunk.iter().enumerate() {
                state[i] ^= val;
            }
            keccak::f1600(&mut state)
        }

        // state[0] and state[1] conform the hash_high (u128)
        // state[2] and state[3] conform the hash_low (u128)
        SyscallResult::Ok(U256 {
            lo: state[2] as u128 | ((state[3] as u128) << 64),
            hi: state[0] as u128 | ((state[1] as u128) << 64),
        })
    }

    fn secp256k1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        // The following unwraps should be unreachable because the iterator we provide has the
        // expected number of bytes.
        let point = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    x.lo.to_be_bytes().into_iter().chain(x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    y.lo.to_be_bytes().into_iter().chain(y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        );

        if bool::from(point.is_some()) {
            Ok(Some(Secp256k1Point { x, y }))
        } else {
            Ok(None)
        }
    }

    fn secp256k1_add(
        &mut self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p0 = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p0.x.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p0.y.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let p1 = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p1.x.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p1.y.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();

        let p = p0 + p1;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256k1Point {
            x: U256 {
                lo: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    fn secp256k1_mul(
        &mut self,
        p: Secp256k1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256k1Point> {
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p.x.lo.to_be_bytes().into_iter().chain(p.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p.y.lo.to_be_bytes().into_iter().chain(p.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let m: k256::Scalar = k256::elliptic_curve::ScalarPrimitive::from_slice(&{
            let mut buf = [0u8; 32];
            buf[0..16].copy_from_slice(&m.lo.to_be_bytes());
            buf[16..32].copy_from_slice(&m.hi.to_be_bytes());
            buf
        })
        .map_err(|_| {
            vec![Felt::from_bytes_be(
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
            )]
        })?
        .into();

        let p = p * m;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256k1Point {
            x: U256 {
                lo: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    fn secp256k1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>> {
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the encoding format, which should be valid
        // since it's hardcoded..
        let point = k256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_bytes(
                k256::CompressedPoint::from_exact_iter(
                    once(0x02 | y_parity as u8)
                        .chain(x.lo.to_be_bytes())
                        .chain(x.hi.to_be_bytes()),
                )
                .unwrap(),
            )
            .unwrap(),
        );

        if bool::from(point.is_some()) {
            // This unwrap has already been checked in the `if` expression's condition.
            let p = point.unwrap();

            let p = p.to_encoded_point(false);
            let y = match p.coordinates() {
                Coordinates::Uncompressed { y, .. } => y,
                _ => {
                    // This should be unreachable because we explicitly asked for the uncompressed
                    // encoding.
                    unreachable!()
                }
            };

            // The following unwrap should be safe because the array always has 32 bytes. The other
            // two are definitely safe because the slicing guarantees its length to be the right
            // one.
            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            Ok(Some(Secp256k1Point {
                x,
                y: U256 {
                    lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                },
            }))
        } else {
            Ok(None)
        }
    }

    fn secp256k1_get_xy(
        &mut self,
        p: Secp256k1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        Ok((p.x, p.y))
    }

    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        // The following unwraps should be unreachable because the iterator we provide has the
        // expected number of bytes.
        let point = p256::ProjectivePoint::from_encoded_point(
            &k256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    x.lo.to_be_bytes().into_iter().chain(x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    y.lo.to_be_bytes().into_iter().chain(y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        );

        if bool::from(point.is_some()) {
            Ok(Some(Secp256r1Point { x, y }))
        } else {
            Ok(None)
        }
    }

    fn secp256r1_add(
        &mut self,
        p0: Secp256r1Point,
        p1: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        // The inner unwraps should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwraps depend on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p0 = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p0.x.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p0.y.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p0.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let p1 = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p1.x.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p1.y.lo
                        .to_be_bytes()
                        .into_iter()
                        .chain(p1.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();

        let p = p0 + p1;

        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256r1Point {
            x: U256 {
                lo: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    fn secp256r1_mul(
        &mut self,
        p: Secp256r1Point,
        m: U256,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Secp256r1Point> {
        // The inner unwrap should be unreachable because the iterator we provide has the expected
        // number of bytes. The outer unwrap depends on the felt values, which should be valid since
        // they'll be provided by secp256 syscalls.
        let p = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_affine_coordinates(
                &GenericArray::from_exact_iter(
                    p.x.lo.to_be_bytes().into_iter().chain(p.x.hi.to_be_bytes()),
                )
                .unwrap(),
                &GenericArray::from_exact_iter(
                    p.y.lo.to_be_bytes().into_iter().chain(p.y.hi.to_be_bytes()),
                )
                .unwrap(),
                false,
            ),
        )
        .unwrap();
        let m: p256::Scalar = p256::elliptic_curve::ScalarPrimitive::from_slice(&{
            let mut buf = [0u8; 32];
            buf[0..16].copy_from_slice(&m.lo.to_be_bytes());
            buf[16..32].copy_from_slice(&m.hi.to_be_bytes());
            buf
        })
        .map_err(|_| {
            vec![Felt::from_bytes_be(
                b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0invalid scalar",
            )]
        })?
        .into();

        let p = p * m;
        let p = p.to_encoded_point(false);
        let (x, y) = match p.coordinates() {
            Coordinates::Uncompressed { x, y } => (x, y),
            _ => {
                // This should be unreachable because we explicitly asked for the uncompressed
                // encoding.
                unreachable!()
            }
        };

        // The following two unwraps should be safe because the array always has 32 bytes. The other
        // four are definitely safe because the slicing guarantees its length to be the right one.
        let x: [u8; 32] = x.as_slice().try_into().unwrap();
        let y: [u8; 32] = y.as_slice().try_into().unwrap();
        Ok(Secp256r1Point {
            x: U256 {
                lo: u128::from_be_bytes(x[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(x[16..32].try_into().unwrap()),
            },
            y: U256 {
                lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
            },
        })
    }

    fn secp256r1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256r1Point>> {
        let point = p256::ProjectivePoint::from_encoded_point(
            &p256::EncodedPoint::from_bytes(
                p256::CompressedPoint::from_exact_iter(
                    once(0x02 | y_parity as u8)
                        .chain(x.lo.to_be_bytes())
                        .chain(x.hi.to_be_bytes()),
                )
                .unwrap(),
            )
            .unwrap(),
        );

        if bool::from(point.is_some()) {
            let p = point.unwrap();

            let p = p.to_encoded_point(false);
            let y = match p.coordinates() {
                Coordinates::Uncompressed { y, .. } => y,
                _ => unreachable!(),
            };

            let y: [u8; 32] = y.as_slice().try_into().unwrap();
            Ok(Some(Secp256r1Point {
                x,
                y: U256 {
                    lo: u128::from_be_bytes(y[0..16].try_into().unwrap()),
                    hi: u128::from_be_bytes(y[16..32].try_into().unwrap()),
                },
            }))
        } else {
            Ok(None)
        }
    }

    fn secp256r1_get_xy(
        &mut self,
        p: Secp256r1Point,
        _remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        Ok((p.x, p.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secp256k1_get_xy() {
        let p = Secp256k1Point {
            x: U256 {
                hi: 331229800296699308591929724809569456681,
                lo: 240848751772479376198639683648735950585,
            },
            y: U256 {
                hi: 75181762170223969696219813306313470806,
                lo: 134255467439736302886468555755295925874,
            },
        };

        let mut test_syscall_handler = TestSyscallHandler {};

        assert_eq!(
            test_syscall_handler.secp256k1_get_xy(p, &mut 10).unwrap(),
            (
                U256 {
                    hi: 331229800296699308591929724809569456681,
                    lo: 240848751772479376198639683648735950585,
                },
                U256 {
                    hi: 75181762170223969696219813306313470806,
                    lo: 134255467439736302886468555755295925874,
                }
            )
        )
    }

    #[test]
    fn test_secp256k1_secp256k1_new() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 {
            hi: 97179038819393695679,
            lo: 330631467365974629050427735731901850225,
        };
        let y = U256 {
            hi: 26163136114030451075775058782541084873,
            lo: 68974579539311638391577168388077592842,
        };

        assert_eq!(
            test_syscall_handler.secp256k1_new(x, y, &mut 10).unwrap(),
            Some(Secp256k1Point { x, y })
        );
    }

    #[test]
    fn test_secp256k1_secp256k1_new_none() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 {
            hi: 97179038819393695679,
            lo: 330631467365974629050427735731901850225,
        };
        let y = U256 { hi: 0, lo: 0 };

        assert!(test_syscall_handler
            .secp256k1_new(x, y, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256k1_ssecp256k1_add() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let p1 = Secp256k1Point {
            x: U256 {
                hi: 161825202758953104525843685720298294023,
                lo: 3468390537006497937951914270391801752,
            },
            y: U256 {
                hi: 96009999919712310848645357523629574312,
                lo: 336417762351022071123394393598455764152,
            },
        };

        let p2 = p1;

        // 2 * P1
        let p3 = test_syscall_handler.secp256k1_add(p1, p2, &mut 10).unwrap();

        let p1_double = Secp256k1Point {
            x: U256 {
                hi: 263210499965038831386353541518668627160,
                lo: 122909745026270932982812610085084241637,
            },
            y: U256 {
                hi: 35730324229579385338853513728577301230,
                lo: 329597642124196932058042157271922763050,
            },
        };
        assert_eq!(p3, p1_double);
        assert_eq!(
            test_syscall_handler
                .secp256k1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
                .unwrap(),
            p1_double
        );

        // 3 * P1
        let three_p1 = Secp256k1Point {
            x: U256 {
                hi: 331229800296699308591929724809569456681,
                lo: 240848751772479376198639683648735950585,
            },
            y: U256 {
                hi: 75181762170223969696219813306313470806,
                lo: 134255467439736302886468555755295925874,
            },
        };
        assert_eq!(
            test_syscall_handler.secp256k1_add(p1, p3, &mut 10).unwrap(),
            three_p1
        );
        assert_eq!(
            test_syscall_handler
                .secp256k1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
                .unwrap(),
            three_p1
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_false_yparity() {
        let mut test_syscall_handler = TestSyscallHandler {};

        assert_eq!(
            test_syscall_handler
                .secp256k1_get_point_from_x(
                    U256 {
                        hi: 97179038819393695679,
                        lo: 330631467365974629050427735731901850225,
                    },
                    false,
                    &mut 10
                )
                .unwrap()
                .unwrap(),
            Secp256k1Point {
                x: U256 {
                    hi: 97179038819393695679,
                    lo: 330631467365974629050427735731901850225,
                },
                y: U256 {
                    hi: 26163136114030451075775058782541084873,
                    lo: 68974579539311638391577168388077592842
                },
            }
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_true_yparity() {
        let mut test_syscall_handler = TestSyscallHandler {};

        assert_eq!(
            test_syscall_handler
                .secp256k1_get_point_from_x(
                    U256 {
                        hi: 97179038819393695679,
                        lo: 330631467365974629050427735731901850225,
                    },
                    true,
                    &mut 10
                )
                .unwrap()
                .unwrap(),
            Secp256k1Point {
                x: U256 {
                    hi: 97179038819393695679,
                    lo: 330631467365974629050427735731901850225,
                },
                y: U256 {
                    hi: 314119230806908012387599548649227126582,
                    lo: 271307787381626825071797439039395650341
                },
            }
        );
    }

    #[test]
    fn test_secp256k1_get_point_from_x_none() {
        let mut test_syscall_handler = TestSyscallHandler {};

        assert!(test_syscall_handler
            .secp256k1_get_point_from_x(U256 { hi: 0, lo: 0 }, true, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_new() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 {
            hi: 97179038819393695679,
            lo: 330631467365974629050427735731901850225,
        };
        let y = U256 {
            hi: 118910939004298029402109603132816090461,
            lo: 111045440647474106186537215379882575585,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_new(x, y, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_new_none() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 { hi: 0, lo: 0 };
        let y = U256 { hi: 0, lo: 0 };

        assert!(test_syscall_handler
            .secp256r1_new(x, y, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_add() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let p1 = Secp256r1Point {
            x: U256 {
                hi: 97179038819393695679,
                lo: 330631467365974629050427735731901850225,
            },
            y: U256 {
                hi: 118910939004298029402109603132816090461,
                lo: 111045440647474106186537215379882575585,
            },
        };

        let p2 = p1;

        // 2 * P1
        let p3 = test_syscall_handler.secp256r1_add(p1, p2, &mut 10).unwrap();

        let p1_double = Secp256r1Point {
            x: U256 {
                hi: 280079427190737520201067412903899817878,
                lo: 309339945874468445579793098896656960879,
            },
            y: U256 {
                hi: 84249534056490759701994051847937833933,
                lo: 231570843221643745062297421862629788481,
            },
        };
        assert_eq!(p3, p1_double);
        assert_eq!(
            test_syscall_handler
                .secp256r1_mul(p1, U256 { hi: 0, lo: 2 }, &mut 10)
                .unwrap(),
            p1_double
        );

        // 3 * P1
        let three_p1 = Secp256r1Point {
            x: U256 {
                hi: 23850518908906170876551962912581992002,
                lo: 195259625777021303662291420857740525307,
            },
            y: U256 {
                hi: 178681203065513270100417145499857169664,
                lo: 282344931843342117515389970197013120959,
            },
        };
        assert_eq!(
            test_syscall_handler.secp256r1_add(p1, p3, &mut 10).unwrap(),
            three_p1
        );
        assert_eq!(
            test_syscall_handler
                .secp256r1_mul(p1, U256 { hi: 0, lo: 3 }, &mut 10)
                .unwrap(),
            three_p1
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_true_yparity() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 {
            hi: 97179038819393695679,
            lo: 330631467365974629050427735731901850225,
        };

        let y = U256 {
            hi: 118910939004298029402109603132816090461,
            lo: 111045440647474106186537215379882575585,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_get_point_from_x(x, true, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_false_yparity() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 {
            hi: 97179038819393695679,
            lo: 330631467365974629050427735731901850225,
        };

        let y = U256 {
            hi: 221371427837412271565447410779117722274,
            lo: 229236926352692519791101729645429586206,
        };

        assert_eq!(
            test_syscall_handler
                .secp256r1_get_point_from_x(x, false, &mut 10)
                .unwrap()
                .unwrap(),
            Secp256r1Point { x, y }
        );
    }

    #[test]
    fn test_secp256r1_get_point_from_x_none() {
        let mut test_syscall_handler = TestSyscallHandler {};

        let x = U256 { hi: 0, lo: 10 };

        assert!(test_syscall_handler
            .secp256r1_get_point_from_x(x, true, &mut 10)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_secp256r1_get_xy() {
        let p = Secp256r1Point {
            x: U256 {
                hi: 97179038819393695679,
                lo: 330631467365974629050427735731901850225,
            },
            y: U256 {
                hi: 221371427837412271565447410779117722274,
                lo: 229236926352692519791101729645429586206,
            },
        };

        let mut test_syscall_handler = TestSyscallHandler {};

        assert_eq!(
            test_syscall_handler.secp256r1_get_xy(p, &mut 10).unwrap(),
            (
                U256 {
                    hi: 97179038819393695679,
                    lo: 330631467365974629050427735731901850225,
                },
                U256 {
                    hi: 221371427837412271565447410779117722274,
                    lo: 229236926352692519791101729645429586206,
                }
            )
        )
    }
}
