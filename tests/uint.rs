use crate::common::{
    any_felt252, felt, feltn, get_result_success, load_cairo, run_native_program, run_vm_program,
};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use common::compare_outputs;
use lazy_static::lazy_static;
use num_traits::Num;
use proptest::prelude::*;
use serde_json::json;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref U8_OVERFLOWING_ADD: (String, Program) = load_cairo! {
        fn run_test(lhs: u8, rhs: u8) -> u8 {
            lhs + rhs
        }
    };
    static ref U8_OVERFLOWING_SUB: (String, Program) = load_cairo! {
        fn run_test(lhs: u8, rhs: u8) -> u8 {
            lhs - rhs
        }
    };
    static ref U8_SAFE_DIVMOD: (String, Program) = load_cairo! {
        fn run_test(lhs: u8, rhs: u8) -> (u8, u8) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }
    };
    static ref U8_EQUAL: (String, Program) = load_cairo! {
        fn run_test(lhs: u8, rhs: u8) -> bool {
            lhs == rhs
        }
    };
    static ref U8_IS_ZERO: (String, Program) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u8_is_zero(a: u8) -> IsZeroResult<u8> implicits() nopanic;

        fn run_test(value: u8) -> bool {
            match u8_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }
    };
    static ref U8_SQRT: (String, Program) = load_cairo! {
        use core::integer::u8_sqrt;

        fn run_test(value: u8) -> u8 {
            u8_sqrt(value)
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
        let result_native = run_native_program(program, "run_test", json!([feltn(a.to_bigint()), feltn(b.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
            true,
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
        let result_native = run_native_program(program, "run_test", json!([feltn(a.to_bigint()), feltn(b.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
            true,
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
        let result_native = run_native_program(program, "run_test", json!([feltn(a.to_bigint()), feltn(b.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
            true,
        )?;
    }
}
