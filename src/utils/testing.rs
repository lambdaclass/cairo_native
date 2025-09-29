#![cfg(any(test, feature = "testing"))]

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_filesystem::db::init_dev_corelib;
#[cfg(test)]
use cairo_lang_sierra::program::FunctionSignature;
use cairo_lang_sierra::{program::Program, ProgramParser};
use cairo_lang_starknet::{compile::compile_contract_in_prepared_db, starknet_plugin_suite};
#[cfg(test)]
use std::{fmt::Formatter, io::Write};
use std::{fs, path::Path, sync::Arc};

use crate::{
    context::NativeContext, execution_result::ExecutionResult, executor::JitNativeExecutor,
    starknet_stub::StubSyscallHandler, utils::*, values::Value,
};
use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter, project::setup_project,
};
use cairo_lang_starknet_classes::contract_class::ContractClass;
use std::env::var;

#[macro_export]
macro_rules! load_cairo {
        ( $( $program:tt )+ ) => {
            $crate::utils::testing::load_cairo_str(stringify!($($program)+))
        };
    }
#[macro_export]
macro_rules! load_starknet {
        ( $( $program:tt )+ ) => {
            $crate::utils::testing::load_starknet_str(stringify!($($program)+))
        };
    }
#[macro_export]
macro_rules! load_starknet_contract {
        ( $( $program:tt )+ ) => {
            $crate::utils::testing::load_starknet_contract_str(stringify!($($program)+))
        };
    }

// Helper macros for faster testing.
#[macro_export]
macro_rules! jit_struct {
        ($($y:expr),* $(,)? ) => {
            $crate::values::Value::Struct {
                fields: vec![$($y), *],
                debug_name: None
            }
        };
    }
#[macro_export]
macro_rules! jit_enum {
    ( $tag:expr, $value:expr ) => {
        $crate::values::Value::Enum {
            tag: $tag,
            value: Box::new($value),
            debug_name: None,
        }
    };
}
#[macro_export]
macro_rules! jit_dict {
        ( $($key:expr $(=>)+ $value:expr),* $(,)? ) => {
            $crate::values::Value::Felt252Dict {
                value: {
                    let mut map = std::collections::HashMap::new();
                    $(map.insert($key.into(), $value.into());)*
                    map
                },
                debug_name: None,
            }
        };
    }
#[macro_export]
macro_rules! jit_panic {
        ( $($value:expr)? ) => {
            $crate::jit_enum!(1, $crate::jit_struct!(
                $crate::jit_struct!(),
                [$($value), *].into()
            ))
        };
    }

/// Compile a cairo program found at the given path to sierra.
pub fn cairo_to_sierra(program: &Path) -> crate::error::Result<Arc<Program>> {
    if program
        .extension()
        .map(|x| {
            x.to_ascii_lowercase()
                .to_string_lossy()
                .eq_ignore_ascii_case("cairo")
        })
        .unwrap_or(false)
    {
        cairo_lang_compiler::compile_cairo_project_at_path(
            program,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    } else {
        let source = std::fs::read_to_string(program)?;
        ProgramParser::new()
            .parse(&source)
            .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    }
    .map(Arc::new)
}

pub fn load_cairo_str(program_str: &str) -> (String, Program) {
    compile_program(program_str, RootDatabase::default())
}

pub fn load_starknet_str(program_str: &str) -> (String, Program) {
    compile_program(
        program_str,
        RootDatabase::builder()
            .with_default_plugin_suite(starknet_plugin_suite())
            .build()
            .unwrap(),
    )
}

pub fn load_starknet_contract_str(program_str: &str) -> (String, ContractClass) {
    compile_contract(
        program_str,
        RootDatabase::builder()
            .with_default_plugin_suite(starknet_plugin_suite())
            .build()
            .unwrap(),
    )
}

pub(crate) fn compile_contract(program_str: &str, mut db: RootDatabase) -> (String, ContractClass) {
    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    fs::write(&mut program_file, program_str).unwrap();

    init_dev_corelib(
        &mut db,
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );
    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
    let contract = compile_contract_in_prepared_db(
        &db,
        None,
        main_crate_ids,
        CompilerConfig {
            diagnostics_reporter: DiagnosticsReporter::stderr(),
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let module_name = program_file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();
    (module_name.to_string(), contract)
}

pub(crate) fn compile_program(program_str: &str, mut db: RootDatabase) -> (String, Program) {
    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    fs::write(&mut program_file, program_str).unwrap();

    init_dev_corelib(
        &mut db,
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );
    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
    let sierra_program_with_dbg = compile_prepared_db(
        &db,
        main_crate_ids,
        CompilerConfig {
            diagnostics_reporter: DiagnosticsReporter::stderr(),
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let module_name = program_file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();
    (module_name.to_string(), sierra_program_with_dbg.program)
}

pub fn run_program(
    program: &(String, Program),
    entry_point: &str,
    args: &[Value],
) -> ExecutionResult {
    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
    let program = &program.1;

    let entry_point_id = &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
        .expect("Test program entry point not found.")
        .id;

    let context = NativeContext::new();

    let module = context
        .compile(program, false, Some(Default::default()), None)
        .expect("Could not compile test program to MLIR.");

    let executor = JitNativeExecutor::from_native_module(module, OptLevel::Less).unwrap();
    executor
        .invoke_dynamic_with_syscall_handler(
            entry_point_id,
            args,
            Some(u64::MAX),
            &mut StubSyscallHandler::default(),
        )
        .unwrap()
}

#[track_caller]
pub fn run_program_assert_output(
    program: &(String, Program),
    entry_point: &str,
    args: &[Value],
    output: Value,
) {
    let result = run_program(program, entry_point, args);
    assert_eq!(result.return_value, output);
}

// ==============================
// == TESTS: get_integer_layout
// ==============================
/// Ensures that the host's `u8` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u8() {
    assert_eq!(get_integer_layout(8).align(), 1);
}

/// Ensures that the host's `u16` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u16() {
    assert_eq!(get_integer_layout(16).align(), 2);
}

/// Ensures that the host's `u32` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u32() {
    assert_eq!(get_integer_layout(32).align(), 4);
}

/// Ensures that the host's `u64` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u64() {
    assert_eq!(get_integer_layout(64).align(), 8);
}

/// Ensures that the host's `u128` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u128() {
    assert_eq!(get_integer_layout(128).align(), 16);
}

/// Ensures that the host's `u256` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u256() {
    assert_eq!(get_integer_layout(256).align(), 16);
}

/// Ensures that the host's `u512` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_u512() {
    assert_eq!(get_integer_layout(512).align(), 16);
}

/// Ensures that the host's `Felt` is compatible with its compiled counterpart.
#[test]
fn test_alignment_compatibility_felt() {
    assert_eq!(get_integer_layout(252).align(), 16);
}

// ==============================
// == TESTS: find_entry_point
// ==============================
#[test]
fn test_find_entry_point_with_empty_program() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![],
    };
    let entry_point = find_entry_point(&program, "entry_point");
    assert!(entry_point.is_none());
}

#[test]
fn test_entry_point_not_found() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![GenFunction {
            id: FunctionId {
                id: 0,
                debug_name: Some("not_entry_point".into()),
            },
            signature: FunctionSignature {
                ret_types: vec![],
                param_types: vec![],
            },
            params: vec![],
            entry_point: StatementIdx(0),
        }],
    };
    let entry_point = find_entry_point(&program, "entry_point");
    assert!(entry_point.is_none());
}

#[test]
fn test_entry_point_found() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![GenFunction {
            id: FunctionId {
                id: 0,
                debug_name: Some("entry_point".into()),
            },
            signature: FunctionSignature {
                ret_types: vec![],
                param_types: vec![],
            },
            params: vec![],
            entry_point: StatementIdx(0),
        }],
    };
    let entry_point = find_entry_point(&program, "entry_point");
    assert!(entry_point.is_some());
    assert_eq!(entry_point.unwrap().id.id, 0);
}

// ====================================
// == TESTS: find_entry_point_by_idx
// ====================================
#[test]
fn test_find_entry_point_by_idx_with_empty_program() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![],
    };
    let entry_point = find_entry_point_by_idx(&program, 0);
    assert!(entry_point.is_none());
}

#[test]
fn test_entry_point_not_found_by_id() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![GenFunction {
            id: FunctionId {
                id: 0,
                debug_name: Some("some_name".into()),
            },
            signature: FunctionSignature {
                ret_types: vec![],
                param_types: vec![],
            },
            params: vec![],
            entry_point: StatementIdx(0),
        }],
    };
    let entry_point = find_entry_point_by_idx(&program, 1);
    assert!(entry_point.is_none());
}

#[test]
fn test_entry_point_found_by_id() {
    let program = Program {
        type_declarations: vec![],
        libfunc_declarations: vec![],
        statements: vec![],
        funcs: vec![GenFunction {
            id: FunctionId {
                id: 15,
                debug_name: Some("some_name".into()),
            },
            signature: FunctionSignature {
                ret_types: vec![],
                param_types: vec![],
            },
            params: vec![],
            entry_point: StatementIdx(0),
        }],
    };
    let entry_point = find_entry_point_by_idx(&program, 15);
    assert!(entry_point.is_some());
    assert_eq!(entry_point.unwrap().id.id, 15);
}

#[test]
fn decode_error_message() {
    // Checkout [issue 795](https://github.com/lambdaclass/cairo_native/issues/795) for context.
    assert_eq!(
            super::decode_error_message(&[
                97, 114, 103, 101, 110, 116, 47, 109, 117, 108, 116, 105, 99, 97, 108, 108, 45,
                102, 97, 105, 108, 101, 100, 3, 232, 78, 97, 116, 105, 118, 101, 32, 101, 120, 101,
                99, 117, 116, 105, 111, 110, 32, 101, 114, 114, 111, 114, 58, 32, 69, 114, 114,
                111, 114, 32, 97, 116, 32, 112, 99, 61, 48, 58, 49, 48, 52, 58, 10, 71, 111, 116,
                32, 97, 110, 32, 101, 120, 99, 101, 112, 116, 105, 111, 110, 32, 119, 104, 105,
                108, 101, 32, 101, 120, 101, 99, 117, 116, 105, 110, 103, 32, 97, 32, 104, 105,
                110, 116, 58, 32, 69, 114, 114, 111, 114, 32, 97, 116, 32, 112, 99, 61, 48, 58, 49,
                56, 52, 58, 10, 71, 111, 116, 32, 97, 110, 32, 101, 120, 99, 101, 112, 116, 105,
                111, 110, 32, 119, 104, 105, 108, 101, 32, 101, 120, 101, 99, 117, 116, 105, 110,
                103, 32, 97, 32, 104, 105, 110, 116, 58, 32, 69, 120, 99, 101, 101, 100, 101, 100,
                32, 116, 104, 101, 32, 109, 97, 120, 105, 109, 117, 109, 32, 110, 117, 109, 98,
                101, 114, 32, 111, 102, 32, 101, 118, 101, 110, 116, 115, 44, 32, 110, 117, 109,
                98, 101, 114, 32, 101, 118, 101, 110, 116, 115, 58, 32, 49, 48, 48, 49, 44, 32,
                109, 97, 120, 32, 110, 117, 109, 98, 101, 114, 32, 101, 118, 101, 110, 116, 115,
                58, 32, 49, 48, 48, 48, 46, 10, 67, 97, 105, 114, 111, 32, 116, 114, 97, 99, 101,
                98, 97, 99, 107, 32, 40, 109, 111, 115, 116, 32, 114, 101, 99, 101, 110, 116, 32,
                99, 97, 108, 108, 32, 108, 97, 115, 116, 41, 58, 10, 85, 110, 107, 110, 111, 119,
                110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58, 49, 52,
                51, 52, 41, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97, 116, 105,
                111, 110, 32, 40, 112, 99, 61, 48, 58, 49, 51, 57, 53, 41, 10, 85, 110, 107, 110,
                111, 119, 110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48,
                58, 57, 53, 51, 41, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97,
                116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58, 51, 51, 57, 41, 10, 10, 67, 97,
                105, 114, 111, 32, 116, 114, 97, 99, 101, 98, 97, 99, 107, 32, 40, 109, 111, 115,
                116, 32, 114, 101, 99, 101, 110, 116, 32, 99, 97, 108, 108, 32, 108, 97, 115, 116,
                41, 58, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97, 116, 105, 111,
                110, 32, 40, 112, 99, 61, 48, 58, 49, 54, 55, 56, 41, 10, 85, 110, 107, 110, 111,
                119, 110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58,
                49, 54, 54, 52, 41, 10
            ]),
            "argent/multicall-failed\x03\\xe8Native execution error: Error at pc=0:104:\nGot an exception while executing a hint: Error at pc=0:184:\nGot an exception while executing a hint: Exceeded the maximum number of events, number events: 1001, max number events: 1000.\nCairo traceback (most recent call last):\nUnknown location (pc=0:1434)\nUnknown location (pc=0:1395)\nUnknown location (pc=0:953)\nUnknown location (pc=0:339)\n\nCairo traceback (most recent call last):\nUnknown location (pc=0:1678)\nUnknown location (pc=0:1664)\n",
        );
}

// ==============================
// == TESTS: felt252_str
// ==============================
#[test]
#[should_panic(expected = "value must be a digit number")]
fn test_felt252_str_invalid_input() {
    let value = "not_a_number";
    felt252_str(value);
}

#[test]
fn test_felt252_str_positive_number() {
    let value = "123";
    let result = felt252_str(value);
    assert_eq!(result, 123.into());
}

#[test]
fn test_felt252_str_negative_number() {
    let value = "-123";
    let result = felt252_str(value);
    assert_eq!(
        result,
        Felt::from_dec_str(
            "3618502788666131213697322783095070105623107215331596699973092056135872020358"
        )
        .unwrap()
    );
}

#[test]
fn test_felt252_str_zero() {
    let value = "0";
    let result = felt252_str(value);
    assert_eq!(result, Felt::ZERO);
}

// ==============================
// == TESTS: felt252_short_str
// ==============================
#[test]
fn test_felt252_short_str_short_numeric_string() {
    let value = "12345";
    let result = felt252_short_str(value);
    assert_eq!(result, 211295614005u64.into());
}

#[test]
fn test_felt252_short_str_short_string_with_non_numeric_characters() {
    let value = "hello";
    let result = felt252_short_str(value);
    assert_eq!(result, 448378203247u64.into());
}

#[test]
#[should_panic]
fn test_felt252_short_str_long_numeric_string() {
    felt252_short_str("1234567890123456789012345678901234567890");
}

#[test]
fn test_felt252_short_str_empty_string() {
    let value = "";
    let result = felt252_short_str(value);
    assert_eq!(result, Felt::ZERO);
}

#[test]
fn test_felt252_short_str_string_with_non_ascii_characters() {
    let value = "h€llø";
    let result = felt252_short_str(value);
    assert_eq!(result, 6843500.into());
}

// ==============================
// == TESTS: debug_with
// ==============================
#[test]
fn test_debug_with_empty_closure() {
    let closure = |_f: &mut Formatter| -> fmt::Result { Ok(()) };
    let debug_wrapper = debug_with(closure);
    assert_eq!(format!("{:?}", debug_wrapper), "");
}

#[test]
#[should_panic]
fn test_debug_with_error_closure() {
    let closure = |_f: &mut Formatter| -> Result<(), fmt::Error> { Err(fmt::Error) };
    let debug_wrapper = debug_with(closure);
    let _ = format!("{:?}", debug_wrapper);
}

#[test]
fn test_debug_with_simple_closure() {
    let closure = |f: &mut fmt::Formatter| write!(f, "Hello, world!");
    let debug_wrapper = debug_with(closure);
    assert_eq!(format!("{:?}", debug_wrapper), "Hello, world!");
}

#[test]
fn test_debug_with_complex_closure() {
    let closure = |f: &mut fmt::Formatter| write!(f, "Name: William, Age: {}", 28);
    let debug_wrapper = debug_with(closure);
    assert_eq!(format!("{:?}", debug_wrapper), "Name: William, Age: 28");
}

#[test]
fn test_generate_function_name_debug_name() {
    let function_id = FunctionId {
        id: 123,
        debug_name: Some("function_name".into()),
    };

    assert_eq!(
        generate_function_name(&function_id, false),
        "function_name(f123)"
    );
}

#[test]
fn test_generate_function_name_debug_name_for_contract_executor() {
    let function_id = FunctionId {
        id: 123,
        debug_name: Some("function_name".into()),
    };

    assert_eq!(generate_function_name(&function_id, true), "f123");
}

#[test]
fn test_generate_function_name_without_debug_name() {
    let function_id = FunctionId {
        id: 123,
        debug_name: None,
    };

    assert_eq!(generate_function_name(&function_id, false), "f123");
}

#[test]
fn test_cairo_to_sierra_path() {
    // Define the path to the cairo program.
    let program_path = Path::new("programs/examples/hello.cairo");
    // Compile the cairo program to sierra.
    let sierra_program = cairo_to_sierra(program_path).unwrap();

    // Define the entry point function for comparison.
    let entry_point = "hello::hello::greet";
    // Find the function ID of the entry point function in the sierra program.
    let entry_point_id = find_function_id(&sierra_program, entry_point).unwrap();

    // Assert that the debug name of the entry point function matches the expected value.
    assert_eq!(
        entry_point_id.debug_name,
        Some("hello::hello::greet".into())
    );
}

#[test]
fn test_cairo_to_sierra_source() {
    // Define the content of the cairo program as a string.
    let content = "type u8 = u8;";

    // Create a named temporary file and write the content to it.
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(content.as_bytes()).unwrap();
    // Get the path of the temporary file.
    let file_path = file.path().to_path_buf();

    // Compile the cairo program to sierra using the path of the temporary file.
    let sierra_program = cairo_to_sierra(&file_path).unwrap();

    // Assert that the sierra program has no library function declarations, statements, or functions.
    assert!(sierra_program.libfunc_declarations.is_empty());
    assert!(sierra_program.statements.is_empty());
    assert!(sierra_program.funcs.is_empty());

    // Assert that the debug name of the first type declaration matches the expected value.
    assert_eq!(sierra_program.type_declarations.len(), 1);
    assert_eq!(
        sierra_program.type_declarations[0].id.debug_name,
        Some("u8".into())
    );
}
