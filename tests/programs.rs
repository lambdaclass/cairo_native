mod common;
use crate::common::{felt, get_result_success, run_native_program, run_vm_program};
use common::load_cairo;
use serde_json::json;

const GAS: usize = usize::MAX;

#[test]
fn fib() {
    let (source, program, runner) = load_cairo! {
        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
            match n {
                0 => a,
                _ => fib(b, a + b, n - 1),
            }
        }

        fn run_test() -> felt252 {
            fib(0, 1, 10)
        }

    };

    let result_vm = run_vm_program(
        &(source.clone(), program.clone(), runner),
        "run_test",
        &[],
        Some(GAS),
    )
    .unwrap();

    let vm_results = get_result_success(result_vm.value);
    let fib_result = &vm_results[0];

    let result = run_native_program(&(source, program), "run_test", json!([null, GAS]));
    assert_eq!(result, json!([null, GAS, [0, [felt(fib_result)]]]));
}

#[test]
fn factorial() {
    let (source, program, runner) = load_cairo! {
        fn factorial(value: felt252, n: felt252) -> felt252 {
            if (n == 1) {
                value
            } else {
                factorial(value * n, n - 1)
            }
        }

        fn run_test() -> felt252 {
            factorial(1, 10)
        }
    };

    let result_vm = run_vm_program(
        &(source.clone(), program.clone(), runner),
        "run_test",
        &[],
        Some(GAS),
    )
    .unwrap();

    let vm_results = get_result_success(result_vm.value);
    let fib_result = &vm_results[0];

    let result = run_native_program(&(source, program), "run_test", json!([null, GAS]));
    assert_eq!(result, json!([null, GAS, [0, [felt(fib_result)]]]));
}
