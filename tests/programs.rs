use crate::common::{
    any_felt252, felt, feltn, get_run_result, load_cairo, run_native_program, run_vm_program,
};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::invoke::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use num_traits::Num;
use proptest::prelude::*;
use serde_json::json;

mod common;
mod starknet;

const GAS: usize = usize::MAX;

lazy_static! {
    pub static ref FACTORIAL: (String, Program, SierraCasmRunner) = load_cairo! {
        fn factorial(value: felt252, n: felt252) -> felt252 {
            if (n == 1) {
                value
            } else {
                factorial(value * n, n - 1)
            }
        }

        fn run_test(n: felt252) -> felt252 {
            factorial(1, n)
        }
    };

    pub static ref FIB: (String, Program, SierraCasmRunner) = load_cairo! {
        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
            match n {
                0 => a,
                _ => fib(b, a + b, n - 1),
            }
        }

        fn run_test(n: felt252) -> felt252 {
            fib(0, 1, n)
        }
    };

    pub static ref LOGISTIC_MAP: (String, Program, SierraCasmRunner) = load_cairo! {
        fn iterate_map(r: felt252, x: felt252) -> felt252 {
            r * x * -x
        }

        // good default: 1000
        fn run_test(mut i: felt252) -> felt252 {
            // Initial value.
            let mut x = 1234567890123456789012345678901234567890;

            // Iterate the map.
            loop {
                x = iterate_map(4, x);

                if i == 0 {
                    break x;
                }

                i = i - 1;
            }
        }
    };

    pub static ref PEDERSEN: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::pedersen::pedersen;

        fn run_test(a: felt252, b: felt252) -> felt252 {
            pedersen(a, b)
        }
    };

    pub static ref POSEIDON: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::poseidon::hades_permutation;

        fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
            hades_permutation(a, b, c)
        }
    };
}

#[test]
fn fib() {
    let result_vm =
        run_vm_program(&FIB, "run_test", &[Arg::Value(Felt252::new(10))], Some(GAS)).unwrap();

    let result_native = run_native_program(&FIB, "run_test", &[JITValue::Felt252(10.into())]);

    compare_outputs(
        &FIB.1,
        &FIB.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn logistic_map() {
    let result_vm = run_vm_program(
        &LOGISTIC_MAP,
        "run_test",
        &[Arg::Value(Felt252::new(1000))],
        Some(GAS),
    )
    .unwrap();

    let result_native =
        run_native_program(&LOGISTIC_MAP, "run_test", &[JITValue::Felt252(1000.into())]);

    compare_outputs(
        &LOGISTIC_MAP.1,
        &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn pedersen() {
    let result_vm = run_vm_program(
        &PEDERSEN,
        "run_test",
        &[
            Arg::Value(
                Felt252::from_str_radix(
                    "2163739901324492107409690946633517860331020929182861814098856895601180685",
                    10,
                )
                .unwrap(),
            ),
            Arg::Value(
                Felt252::from_str_radix(
                    "2392090257937917229310563411601744459500735555884672871108624696010915493156",
                    10,
                )
                .unwrap(),
            ),
        ],
        Some(GAS),
    )
    .unwrap();

    let result_native = run_native_program(
        &PEDERSEN,
        "run_test",
        &[
            JITValue::felt_str(
                "2163739901324492107409690946633517860331020929182861814098856895601180685",
            ),
            JITValue::felt_str(
                "2392090257937917229310563411601744459500735555884672871108624696010915493156",
            ),
        ],
    );

    compare_outputs(
        &PEDERSEN.1,
        &PEDERSEN.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn factorial() {
    let result_vm = run_vm_program(
        &FACTORIAL,
        "run_test",
        &[Arg::Value(Felt252::new(13))],
        Some(GAS),
    )
    .unwrap();
    let result_native = run_native_program(&FACTORIAL, "run_test", &[JITValue::Felt252(13.into())]);

    compare_outputs(
        &FACTORIAL.1,
        &FACTORIAL.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

proptest! {
    #[test]
    fn fib_proptest(n in 0..100i32) {
        let result_vm = run_vm_program(
            &FIB,
            "run_test",
            &[Arg::Value(Felt252::new(n))],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(&FIB, "run_test", &[JITValue::Felt252(n.into())]);

        compare_outputs(
            &FIB.1,
            &FIB.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn logistic_map_proptest(n in 100..110i32) {
        let result_vm = run_vm_program(
            &LOGISTIC_MAP,
            "run_test",
            &[Arg::Value(Felt252::new(n))],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(&LOGISTIC_MAP, "run_test", &[JITValue::Felt252(n.into())]);

        compare_outputs(
            &LOGISTIC_MAP.1,
            &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn factorial_proptest(n in 1..100i32) {
        let result_vm = run_vm_program(
            &FACTORIAL,
            "run_test",
            &[Arg::Value(Felt252::new(n))],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(&FACTORIAL, "run_test", &[JITValue::Felt252(n.into())]);

        compare_outputs(
            &FACTORIAL.1,
            &FACTORIAL.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn pedersen_proptest(a in any_felt252(), b in any_felt252()) {
        let result_vm = run_vm_program(
            &PEDERSEN,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone())],
            Some(GAS),
        )
        .unwrap();

        let result_native = run_native_program(&PEDERSEN, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b)]);

        compare_outputs(
            &PEDERSEN.1,
            &PEDERSEN.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn poseidon_proptest(a in any_felt252(), b in any_felt252(), c in any_felt252()) {
        let result_vm = run_vm_program(
            &POSEIDON,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone()), Arg::Value(c.clone())],
            Some(GAS),
        )
        .unwrap();

        let result_native = run_native_program(&PEDERSEN, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b), JITValue::Felt252(c)]);

        compare_outputs(
            &POSEIDON.1,
            &POSEIDON.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
