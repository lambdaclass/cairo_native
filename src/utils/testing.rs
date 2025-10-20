#![cfg(any(test, feature = "testing"))]

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_filesystem::db::init_dev_corelib;
use cairo_lang_sierra::{program::Program, ProgramParser};
use cairo_lang_starknet::{compile::compile_contract_in_prepared_db, starknet_plugin_suite};
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
