use crate::common::{compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::{starknet::DummySyscallHandler, Value};
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    static ref FELT252_TO_BOOL: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x == 1
        }

        fn run_test(a: felt252) -> bool {
            felt_to_bool(a)
        }
    };
    static ref BOOL_NOT: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x.try_into().unwrap() == 1_u8
        }

        fn program(a: bool) -> bool {
            !a
        }

        fn run_test(a: felt252) -> bool {
            program(felt_to_bool(a))
        }
    };
    static ref BOOL_AND: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x.try_into().unwrap() == 1_u8
        }

        fn program(a: bool, b: bool) -> bool {
            a && b
        }

        fn run_test(a: felt252, b: felt252) -> bool {
            program(felt_to_bool(a), felt_to_bool(b))
        }
    };
    static ref BOOL_OR: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x.try_into().unwrap() == 1_u8
        }

        fn program(a: bool, b: bool) -> bool {
            a || b
        }

        fn run_test(a: felt252, b: felt252) -> bool {
            program(felt_to_bool(a), felt_to_bool(b))
        }
    };
    static ref BOOL_XOR: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x.try_into().unwrap() == 1_u8
        }

        fn program(a: bool, b: bool) -> bool {
            a ^ b
        }

        fn run_test(a: felt252, b: felt252) -> bool {
            program(felt_to_bool(a), felt_to_bool(b))
        }
    };
    static ref BOOL_TO_FELT252: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use core::bool_to_felt252;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn felt_to_bool(x: felt252) -> bool {
            x.try_into().unwrap() == 1_u8
        }

        fn program(a: bool) -> felt252 {
            bool_to_felt252(a)
        }

        fn run_test(a: felt252) -> felt252 {
            program(felt_to_bool(a))
        }
    };
}

#[test]
fn felt252_to_bool_bug() {
    let program = &FELT252_TO_BOOL;
    let a = true;
    let result_vm = run_vm_program(
        program,
        "run_test",
        &[Arg::Value(Felt::from(a))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[Value::Felt252(a.into())],
        Some(DEFAULT_GAS as u128),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();

    let a = false;
    let result_vm = run_vm_program(
        program,
        "run_test",
        &[Arg::Value(Felt::from(a))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[Value::Felt252(a.into())],
        Some(DEFAULT_GAS as u128),
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

proptest! {
    #[test]
    fn bool_to_felt252_proptest(a: bool) {
        let program = &BOOL_TO_FELT252;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(Felt::from(a)),
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into())],
            Some(DEFAULT_GAS as u128),
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
    fn bool_not_proptest(a: bool) {
        let program = &BOOL_NOT;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(Felt::from(a)),
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into())],
            Some(DEFAULT_GAS as u128),
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
    fn bool_and_proptest(a: bool, b: bool) {
        let program = &BOOL_AND;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS as u128),
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
    fn bool_or_proptest(a: bool, b: bool) {
        let program = &BOOL_OR;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS as u128),
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
    fn bool_xor_proptest(a: bool, b: bool) {
        let program = &BOOL_XOR;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS as u128),
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
}
