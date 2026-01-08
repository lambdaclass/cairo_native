//! To avoid generating lot of test executables, this is the single entry point of all tests.

use cairo_native::{
    starknet_stub::StubSyscallHandler,
    utils::testing::{load_contract, load_program, run_contract_with_native, run_with_native},
    Value,
};

pub mod common;
pub mod tests;

#[test]
fn fibonacci_function_10() {
    let versioned_program = load_program("fibonacci");

    let execution = run_with_native(
        versioned_program,
        "fibonacci::fibonacci::fibonacci",
        &[Value::Felt252(10.into())],
        None,
        None::<&mut StubSyscallHandler>,
    );

    assert_eq!(
        execution.return_value,
        Value::Enum {
            tag: 0,
            value: Value::Struct {
                fields: vec![Value::Felt252(55.into()),],
                debug_name: None
            }
            .into(),
            debug_name: None
        }
    )
}

#[test]
fn fibonacci_contract_10() {
    let contract_class = load_contract("fibonacci");

    let execution = run_contract_with_native(
        contract_class,
        "fibonacci",
        &[10.into()],
        None,
        &mut StubSyscallHandler::default(),
    );

    assert_eq!(execution.return_values, vec![55.into()]);
    assert!(!execution.failure_flag);
    assert_eq!(execution.error_msg, None);
}
