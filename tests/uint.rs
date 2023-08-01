use crate::common::{feltn, load_cairo, run_native_program, run_vm_program};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;
use serde_json::json;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref U8_OVERFLOWING_ADD: (String, Program, SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> u8 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u8 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_OVERFLOWING_SUB: (String, Program, SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> u8 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u8 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_SAFE_DIVMOD: (String, Program, SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> (u8, u8) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u8, u8) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_EQUAL: (String, Program, SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_IS_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u8_is_zero(a: u8) -> IsZeroResult<u8> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u8) -> bool {
            match u8_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U8_SQRT: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::integer::u8_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u8) -> u8 {
            u8_sqrt(value)
        }

        fn run_test(value: felt252) -> u8 {
            program(value.try_into().unwrap())
        }
    };
}

proptest! {
    #[test]
    fn u8_overflowing_add_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let program = &U8_OVERFLOWING_ADD;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.into()), Arg::Value(b.into())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a), feltn(b)]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
        )?;
    }

    #[test]
    fn u8_overflowing_sub_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let program = &U8_OVERFLOWING_SUB;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.into()), Arg::Value(b.into())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a), feltn(b)]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
        )?;
    }

    #[test]
    fn u8_safe_divmod_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let program = &U8_SAFE_DIVMOD;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.into()), Arg::Value(b.into())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a), feltn(b)]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
        )?;
    }

    #[test]
    fn u8_equal_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let program = &U8_EQUAL;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.into()), Arg::Value(b.into())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a), feltn(b)]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
        )?;
    }

    #[test]
    fn u8_is_zero_proptest(a in 0..u8::MAX) {
        let program = &U8_IS_ZERO;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.into())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a)]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
            true,
        )?;
    }
}
