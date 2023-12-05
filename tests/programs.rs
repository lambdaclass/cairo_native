use crate::common::{any_felt252, load_cairo, run_native_or_vm_program};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use num_traits::Num;
use proptest::prelude::*;
use std::borrow::Borrow;
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
    let program = &FIB;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(10))]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(10.into())]),
        None,
        None,
        None,
    )
    .right()
    .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn logistic_map() {
    let program = &LOGISTIC_MAP;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(1000))]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(1000.into())]),
        None,
        None,
        None,
    )
    .right()
    .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn pedersen() {
    let program = &PEDERSEN;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[
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
        ]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[
            JITValue::felt_str(
                "2163739901324492107409690946633517860331020929182861814098856895601180685",
            ),
            JITValue::felt_str(
                "2392090257937917229310563411601744459500735555884672871108624696010915493156",
            ),
        ]),
        None,
        None,
        None,
    )
    .right()
    .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn factorial() {
    let program = &FACTORIAL;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(13))]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(13.into())]),
        None,
        None,
        None,
    )
    .right()
    .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

proptest! {
    #[test]
    fn fib_proptest(n in 0..100i32) {

        let program = &FIB;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(n))]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(n.into())]), None, None, None).right().unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
        .unwrap();
    }

    #[test]
    fn logistic_map_proptest(n in 100..110i32) {
let program = &LOGISTIC_MAP;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(n))]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(n.into())]), None, None, None).right().unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
        .unwrap();
    }

    #[test]
    fn factorial_proptest(n in 1..100i32) {
        let program = &FACTORIAL;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(n))]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(n.into())]), None, None, None).right().unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
        .unwrap();
    }

    #[test]
    fn pedersen_proptest(a in any_felt252(), b in any_felt252()) {


        let program = &FIB;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(a.clone()), Arg::Value(b.clone())]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(a), JITValue::Felt252(b)]), None, None, None).right().unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
        .unwrap();
    }

    #[test]
    fn poseidon_proptest(a in any_felt252(), b in any_felt252(), c in any_felt252()) {

        let program = &POSEIDON;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(a.clone()), Arg::Value(b.clone())]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(a), JITValue::Felt252(b)]), None, None, None).right().unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
        .unwrap();
    }
}
