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
    debug_info::{DebugInfo, DebugLocations},
    execution_result::ExecutionResult,
    executor::{AotNativeExecutor, JitNativeExecutor, NativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    values::JitValue,
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

    let debug_locations = {
        let debug_info = DebugInfo::extract(db, &sierra_program)
            .map_err(|_| {
                let mut buffer = String::new();
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(db));
                buffer
            })
            .unwrap();

        DebugLocations::extract(native_context.context(), db, &debug_info)
    };

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
            if debug_name
                .as_ref()
                .expect("missing debug name")
                .starts_with("core::panics::PanicResult::")
            {
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
    match value {
        JitValue::Felt252(felt) => vec![*felt],
        JitValue::BoundedInt { value, .. } => vec![*value],
        JitValue::Bytes31(bytes) => vec![Felt::from_bytes_le_slice(bytes)],
        JitValue::Array(fields) | JitValue::Struct { fields, .. } => {
            fields.iter().flat_map(jitvalue_to_felt).collect()
        }
        JitValue::Enum {
            value,
            tag,
            debug_name,
        } => {
            if let Some(debug_name) = debug_name {
                if debug_name == "core::bool" {
                    vec![(*tag == 1).into()]
                } else {
                    let mut felts = vec![(*tag).into()];
                    felts.extend(jitvalue_to_felt(value));
                    felts
                }
            } else {
                todo!()
            }
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
        JitValue::Null => vec![0.into()],
        JitValue::Felt252Dict {
            value,
            debug_name: _,
        } => {
            let mut result = Vec::new();
            for (k, v) in value.iter() {
                result.push(*k);
                result.append(&mut jitvalue_to_felt(v))
            }
            result
        }
        JitValue::EcPoint(x, y) => {
            vec![*x, *y]
        }
        JitValue::EcState(a, b, c, d) => {
            vec![*a, *b, *c, *d]
        }
        JitValue::Secp256K1Point { x, y } => {
            vec![x.0.into(), x.1.into(), y.0.into(), y.1.into()]
        }
        JitValue::Secp256R1Point { x, y } => {
            vec![x.0.into(), x.1.into(), y.0.into(), y.1.into()]
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use cairo_felt::Felt252;
    use cairo_lang_sierra::ProgramParser;

    #[test]
    fn test_check_compiler_path() {
        // Define file, folder, and invalid paths for testing
        let file_path = Path::new("src/bin/cairo-native-run.rs");
        let folder_path = Path::new("src/bin");
        let invalid_path = Path::new("src/non-existing-file.rs");

        // Test when single_file is true and the path is a file
        assert!(check_compiler_path(true, file_path).is_ok());

        // Test when single_file is false and the path is a file
        assert!(check_compiler_path(false, file_path).is_err());

        // Test when single_file is true and the path is a folder
        assert!(check_compiler_path(true, folder_path).is_err());

        // Test when single_file is false and the path is a folder
        assert!(check_compiler_path(false, folder_path).is_ok());

        // Test when single_file is true and the path does not exist
        assert!(check_compiler_path(true, invalid_path).is_err());

        // Test when single_file is false and the path does not exist
        assert!(check_compiler_path(false, invalid_path).is_err());
    }

    #[test]
    fn test_find_function() {
        // Parse a simple program containing a function named "Func2"
        let program = ProgramParser::new().parse("Func2@6() -> ();").unwrap();

        // Assert that the function "Func2" is found and returned correctly
        assert_eq!(
            find_function(&program, "Func2").unwrap(),
            program.funcs.first().unwrap()
        );

        // Assert that an error is returned when trying to find a non-existing function "Func3"
        assert!(find_function(&program, "Func3").is_err());

        // Assert that an error is returned when trying to find a function in an empty program
        assert!(find_function(&ProgramParser::new().parse("").unwrap(), "Func2").is_err());
    }

    #[test]
    fn test_result_to_runresult_enum_nonpanic() {
        // Tests the conversion of a non-panic enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 34,
                    value: JitValue::Array(vec![
                        JitValue::Felt252(42.into()),
                        JitValue::Uint8(100),
                        JitValue::Uint128(1000),
                    ])
                    .into(),
                    debug_name: Some("debug_name".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![
                Felt252::from(34),
                Felt252::from(42),
                Felt252::from(100),
                Felt252::from(1000)
            ])
        );
    }

    #[test]
    fn test_result_to_runresult_success() {
        // Tests the conversion of a success enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 0,
                    value: JitValue::Uint64(24).into(),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![Felt252::from(24)])
        );
    }

    #[test]
    #[should_panic(expected = "unsuported return value in cairo-native")]
    fn test_result_to_runresult_panic() {
        // Tests the conversion with unsuported return value.
        let _ = result_to_runresult(&ExecutionResult {
            remaining_gas: None,
            return_value: JitValue::Enum {
                tag: 10,
                value: JitValue::Uint64(24).into(),
                debug_name: Some("core::panics::PanicResult::Test".into()),
            },
            builtin_stats: Default::default(),
        })
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "missing debug name")]
    fn test_result_to_runresult_missing_debug_name() {
        // Tests the conversion with no debug name.
        let _ = result_to_runresult(&ExecutionResult {
            remaining_gas: None,
            return_value: JitValue::Enum {
                tag: 10,
                value: JitValue::Uint64(24).into(),
                debug_name: None,
            },
            builtin_stats: Default::default(),
        })
        .unwrap();
    }

    #[test]
    fn test_result_to_runresult_return() {
        // Tests the conversion of a panic enum result with non-zero tag to a `RunResultValue::Panic`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 10,
                    value: JitValue::Struct {
                        fields: vec![
                            JitValue::Felt252(42.into()),
                            JitValue::Uint8(100),
                            JitValue::Uint128(1000),
                        ],
                        debug_name: Some("debug_name".into()),
                    }
                    .into(),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Panic(vec![
                Felt252::from(42),
                Felt252::from(100),
                Felt252::from(1000)
            ])
        );
    }

    #[test]
    fn test_result_to_runresult_non_enum() {
        // Tests the conversion of a non-enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Uint8(10),
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![Felt252::from(10)])
        );
    }

    #[test]
    fn test_jitvalue_to_felt_felt252() {
        let felt_value: Felt = 42.into();

        assert_eq!(
            jitvalue_to_felt(&JitValue::Felt252(felt_value)),
            vec![felt_value]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_array() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Array(vec![
                JitValue::Felt252(42.into()),
                JitValue::Uint8(100),
                JitValue::Uint128(1000),
            ])),
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_struct() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Struct {
                fields: vec![
                    JitValue::Felt252(42.into()),
                    JitValue::Uint8(100),
                    JitValue::Uint128(1000)
                ],
                debug_name: Some("debug_name".into())
            }),
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_enum() {
        // With debug name
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 34,
                value: JitValue::Array(vec![
                    JitValue::Felt252(42.into()),
                    JitValue::Uint8(100),
                    JitValue::Uint128(1000),
                ])
                .into(),
                debug_name: Some("debug_name".into())
            }),
            vec![
                Felt::from(34),
                Felt::from(42),
                Felt::from(100),
                Felt::from(1000)
            ]
        );

        // With core::bool debug name and tag 1
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 1,
                value: JitValue::Uint128(1000).into(),
                debug_name: Some("core::bool".into())
            }),
            vec![Felt::ONE]
        );

        // With core::bool debug name and tag not 1
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 10,
                value: JitValue::Uint128(1000).into(),
                debug_name: Some("core::bool".into())
            }),
            vec![Felt::ZERO]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u8() {
        assert_eq!(jitvalue_to_felt(&JitValue::Uint8(10)), vec![Felt::from(10)]);
    }

    #[test]
    fn test_jitvalue_to_felt_u16() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint16(100)),
            vec![Felt::from(100)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u32() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint32(1000)),
            vec![Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u64() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint64(10000)),
            vec![Felt::from(10000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u128() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint128(100000)),
            vec![Felt::from(100000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint8() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint8(-10)),
            vec![Felt::from(-10)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint16() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint16(-100)),
            vec![Felt::from(-100)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint32() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint32(-1000)),
            vec![Felt::from(-1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint64() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint64(-10000)),
            vec![Felt::from(-10000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint128() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint128(-100000)),
            vec![Felt::from(-100000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_null() {
        assert_eq!(jitvalue_to_felt(&JitValue::Null), vec![Felt::ZERO]);
    }

    /// Check if subsequence is present in sequence
    fn is_subsequence<T: PartialEq>(subsequence: &[T], mut sequence: &[T]) -> bool {
        for search in subsequence {
            if let Some(index) = sequence.iter().position(|element| search == element) {
                sequence = &sequence[index + 1..];
            } else {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_jitvalue_to_felt_felt252_dict() {
        let result = jitvalue_to_felt(&JitValue::Felt252Dict {
            value: HashMap::from([
                (Felt::ONE, JitValue::Felt252(Felt::from(101))),
                (
                    Felt::TWO,
                    JitValue::Array(Vec::from([
                        JitValue::Felt252(Felt::from(201)),
                        JitValue::Felt252(Felt::from(202)),
                    ])),
                ),
            ]),
            debug_name: None,
        });

        let first_dict_entry = vec![Felt::from(1), Felt::from(101)];
        let second_dict_entry = vec![Felt::from(2), Felt::from(201), Felt::from(202)];

        // Check that the two Key, value pairs are in the result
        assert!(is_subsequence(&first_dict_entry, &result));
        assert!(is_subsequence(&second_dict_entry, &result));
    }
    #[test]
    fn test_jitvalue_to_felt_ec_point() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::EcPoint(Felt::ONE, Felt::TWO,)),
            vec![Felt::ONE, Felt::TWO,]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_ec_state() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::EcState(
                Felt::ONE,
                Felt::TWO,
                Felt::THREE,
                Felt::from(4)
            )),
            vec![Felt::ONE, Felt::TWO, Felt::THREE, Felt::from(4)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_secp256_k1_point() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Secp256K1Point {
                x: (1, 2),
                y: (3, 4)
            }),
            vec![Felt::ONE, Felt::TWO, Felt::THREE, Felt::from(4)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_secp256_r1_point() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Secp256R1Point {
                x: (1, 2),
                y: (3, 4)
            }),
            vec![Felt::ONE, Felt::TWO, Felt::THREE, Felt::from(4)]
        );
    }
}
