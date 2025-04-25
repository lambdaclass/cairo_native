use std::{path::Path, sync::Arc, u64};

use cairo_lang_compiler::{
    db::RootDatabase,
    diagnostics::DiagnosticsReporter,
    project::{check_compiler_path, setup_project},
};
use cairo_lang_filesystem::{
    cfg::{Cfg, CfgSet},
    ids::CrateId,
};
use cairo_lang_runner::{casm_run::format_for_panic, RunResultValue};
use cairo_lang_sierra_generator::replace_ids::replace_sierra_ids_in_program;
use cairo_lang_starknet::starknet_plugin_suite;
use cairo_lang_test_plugin::{
    compile_test_prepared_db,
    test_config::{PanicExpectation, TestExpectation},
    test_plugin_suite, TestCompilation, TestsCompilationConfig,
};
use common::value_to_felt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sierra_emu::{run_program, EntryPoint, ProgramTrace, Value};
use tracing::info;
use tracing_test::traced_test;

mod common;

#[traced_test]
#[test]
#[ignore = "sierra-emu is not fully implemented yet"]
fn test_corelib() {
    let compiler_path = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("corelib");

    check_compiler_path(false, &compiler_path)
        .expect("Couldn't find the corelib in the given path");

    let db = &mut {
        let mut b = RootDatabase::builder();
        b.detect_corelib();
        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
        b.with_default_plugin_suite(test_plugin_suite());
        b.with_default_plugin_suite(starknet_plugin_suite());

        b.build().unwrap()
    };

    let main_crate_ids = setup_project(db, &compiler_path).unwrap();

    let db = db.snapshot();
    let test_crate_ids = main_crate_ids.clone();
    let test_config = TestsCompilationConfig {
        starknet: false,
        add_statements_functions: true,
        add_statements_code_locations: true,
        contract_declarations: None,
        contract_crate_ids: None,
        executable_crate_ids: None,
    };

    let diag_reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);

    let filtered_tests = vec![
        "core::test::dict_test::test_array_from_squash_dict",
        "core::test::hash_test::test_blake2s",
        "core::test::testing_test::test_get_unspent_gas",
    ];

    let compiled = compile_tests(
        &db,
        test_config,
        test_crate_ids,
        diag_reporter,
        Some(&filtered_tests),
    );

    run_tests(compiled);
}

/// Runs the tests and process the results for a summary.
pub fn run_tests(compiled: TestCompilation) {
    let program = Arc::new(compiled.sierra_program.program);

    let success = compiled
        .metadata
        .named_tests
        .into_par_iter()
        .filter(|(_, test)| !test.ignored)
        .map(|(name, test)| {
            let trace = run_program(
                program.clone(),
                EntryPoint::String(name.clone()),
                vec![],
                u64::MAX,
            );
            let run_result = trace_to_run_result(trace);

            assert_test_expectation(name, test.expectation, run_result)
        })
        .all(|r| r);

    assert!(success, "Some tests from the corelib have failed");
}

fn trace_to_run_result(trace: ProgramTrace) -> RunResultValue {
    let return_value = trace.return_value();
    let mut felts = Vec::new();

    let is_success = match &return_value {
        outer_value @ Value::Enum {
            self_ty,
            index,
            payload,
            ..
        } => {
            let debug_name = self_ty.debug_name.as_ref().expect("missing debug name");
            if debug_name.starts_with("core::panics::PanicResult::")
                || debug_name.starts_with("Enum<ut@core::panics::PanicResult::")
            {
                let is_success = *index == 0;

                if !is_success {
                    match &**payload {
                        Value::Struct(fields) => {
                            for field in fields {
                                let felt = value_to_felt(&field);
                                felts.extend(felt);
                            }
                        }
                        _ => panic!("unsuported return value in cairo-native"),
                    }
                } else {
                    felts.extend(value_to_felt(&*payload));
                }

                is_success
            } else {
                felts.extend(value_to_felt(&outer_value));
                true
            }
        }
        x => {
            felts.extend(value_to_felt(&x));
            true
        }
    };

    let return_values = felts.into_iter().map(|x| x.to_bigint().into()).collect();

    match is_success {
        true => RunResultValue::Success(return_values),
        false => RunResultValue::Panic(return_values),
    }
}

fn assert_test_expectation(
    name: String,
    expectation: TestExpectation,
    result: RunResultValue,
) -> bool {
    let mut success = true;

    match result {
        RunResultValue::Success(r) => {
            if let TestExpectation::Panics(_) = expectation {
                let err_msg = format_for_panic(r.into_iter());
                info!("test {} ... FAILED: {}", name, err_msg);
                success = false;
            }
            info!("test {} ... OK", name);
        }
        RunResultValue::Panic(e) => match expectation {
            TestExpectation::Success => {
                let err_msg = format_for_panic(e.into_iter());
                info!("test {} ... FAILED: {}", name, err_msg);
                success = false;
            }
            TestExpectation::Panics(panic_expect) => {
                if let PanicExpectation::Exact(expected) = panic_expect {
                    if expected != e {
                        let err_msg = format_for_panic(e.into_iter());
                        info!("test {} ... FAILED: {}", name, err_msg);
                        success = false;
                    }
                    info!("test {} ... OK", name);
                }
            }
        },
    }

    success
}

fn compile_tests(
    db: &RootDatabase,
    test_config: TestsCompilationConfig,
    test_crate_ids: Vec<CrateId>,
    diag_reporter: DiagnosticsReporter<'_>,
    with_filtered_tests: Option<&[&str]>,
) -> TestCompilation {
    let mut compiled =
        compile_test_prepared_db(db, test_config, test_crate_ids.clone(), diag_reporter).unwrap();
    compiled.sierra_program.program =
        replace_sierra_ids_in_program(db, &compiled.sierra_program.program);

    if let Some(compilation_filter) = with_filtered_tests {
        let should_skip_test = |name: &str| -> bool {
            compilation_filter
                .iter()
                .any(|filter| name.contains(filter))
        };

        // Ignore matching test cases.
        compiled
            .metadata
            .named_tests
            .iter_mut()
            .for_each(|(test, case)| {
                if should_skip_test(test) {
                    case.ignored = true
                }
            });
    }

    compiled
}
