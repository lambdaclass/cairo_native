use crate::common::{compare_outputs, get_compiled_program, run_native_program, run_vm_program};
use cairo_native::starknet::DummySyscallHandler;

#[test]
fn enum_init() {
    let program = &get_compiled_program("test_data_artifacts/programs/enum_init");

    let result_vm = run_vm_program(&program, "run_test", vec![], None).unwrap();
    let result_native = run_native_program(
        &program,
        "run_test",
        &[],
        None,
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn enum_match() {
    let program = &get_compiled_program("test_data_artifacts/programs/enum_match");

    let result_vm = run_vm_program(&program, "match_a", vec![], None).unwrap();
    let result_native = run_native_program(
        &program,
        "match_a",
        &[],
        None,
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("match_a").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();

    let result_vm = run_vm_program(&program, "match_b", vec![], None).unwrap();
    let result_native = run_native_program(
        &program,
        "match_b",
        &[],
        None,
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("match_b").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
