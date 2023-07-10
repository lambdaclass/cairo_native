mod common;
use crate::common::{felt, get_result_success, run_native_program, run_vm_program, GAS};
use cairo_felt::Felt252;
use cairo_lang_runner::Arg;
use common::load_cairo;
use serde_json::json;
use pretty_assertions::assert_eq;

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
    let vm_result = &vm_results[0];

    let result = run_native_program(&(source, program), "run_test", json!([null, GAS]));
    assert_eq!(result, json!([null, GAS, [0, [felt(vm_result)]]]));
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
    let vm_result = &vm_results[0];

    let result = run_native_program(&(source, program), "run_test", json!([null, GAS]));
    assert_eq!(result, json!([null, GAS, [0, [felt(vm_result)]]]));
}

#[test]
fn logistic_map() {
    let (source, program, runner) = load_cairo! {
        fn iterate_map(r: felt252, x: felt252) -> felt252 {
            r * x * -x
        }

        fn run_test() -> felt252 {
            // Initial value.
            let mut x = 1234567890123456789012345678901234567890;

            // Iterate the map.
            let mut i = 1000;
            loop {
                x = iterate_map(4, x);

                if i == 0 {
                    break x;
                }

                i = i - 1;
            }
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
fn pedersen() {
    let (source, program, runner) = load_cairo! {
        use hash::pedersen;

        fn run_test(a: felt252, b: felt252) -> felt252 {
            pedersen(a, b)
        }
    };

    let result_vm = run_vm_program(
        &(source.clone(), program.clone(), runner),
        "run_test",
        &[Arg::Value(Felt252::new(2)), Arg::Value(Felt252::new(4))],
        Some(GAS),
    )
    .unwrap();

    let vm_results = get_result_success(result_vm.value);
    let vm_result = &vm_results[0];

    let result = run_native_program(
        &(source, program),
        "run_test",
        json!([null, felt("2"), felt("4")]),
    );
    assert_eq!(result, json!([null, felt(vm_result)]));
}
