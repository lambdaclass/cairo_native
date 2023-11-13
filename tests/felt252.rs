use crate::common::{any_felt252, load_cairo, run_native_program, run_vm_program};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::invoke::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {

    static ref FELT252_ADD: (String, Program, SierraCasmRunner) = load_cairo! {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            lhs + rhs
        }
    };

    static ref FELT252_SUB: (String, Program, SierraCasmRunner) = load_cairo! {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            lhs - rhs
        }
    };

    static ref FELT252_MUL: (String, Program, SierraCasmRunner) = load_cairo! {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            lhs * rhs
        }
    };

    // TODO: Add test program for `felt252_div`.

    // TODO: Add test program for `felt252_add_const`.
    // TODO: Add test program for `felt252_sub_const`.
    // TODO: Add test program for `felt252_mul_const`.
    // TODO: Add test program for `felt252_div_const`.

    static ref FELT252_CONST: (String, Program, SierraCasmRunner) = load_cairo! {
        fn run_test() -> (felt252, felt252, felt252, felt252) {
            (0, 1, -2, -1)
        }
    };

    static ref FELT252_IS_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
        fn run_test(x: felt252) -> felt252 {
            match x {
                0 => 1,
                _ => 0,
            }
        }
    };
}

proptest! {
    #[test]
    fn felt_add_proptest(a in any_felt252(), b in any_felt252()) {
        let program = &FELT252_ADD;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b)]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn felt_sub_proptest(a in any_felt252(), b in any_felt252()) {
        let program = &FELT252_SUB;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b)]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn felt_mul_proptest(a in any_felt252(), b in any_felt252()) {
        let program = &FELT252_MUL;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b)]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
