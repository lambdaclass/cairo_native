use crate::common::{load_cairo, run_native_or_vm_program};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;
use std::borrow::Borrow;
use std::ops::Deref;

mod common;

const GAS: usize = usize::MAX;

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

// Since comparing a felt to 1 to create boolean (uses felt252_is_zero and felt sub,add) has a bug,
// we'll be using use u8 on other tests until this is fixed. The bug may be in felt subtraction.
#[ignore = "TODO: comparing a felt252 == 1 will lead to wrong results"]
#[test]
fn felt252_to_bool_bug() {
    let program = &FELT252_TO_BOOL;
    let a = true;

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(a))]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(a.into())]),
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

    let a = false;

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(a))]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left().unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(a.into())]),
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
    fn bool_to_felt252_proptest(a: bool) {
        let program = &BOOL_TO_FELT252;


   let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(a))]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap();


    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(a.into())]), None, None, None).right().unwrap();

        compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_not_proptest(a: bool) {
        let program = &BOOL_NOT;

   let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(Felt252::new(a))]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap();


    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[JITValue::Felt252(a.into())]), None, None, None).right().unwrap();

        compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_and_proptest(a: bool, b: bool) {
        let program = &BOOL_AND;


   let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[
            Arg::Value(Felt252::new(a)),
            Arg::Value(Felt252::new(b))
        ]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some( &[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();


        compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_or_proptest(a: bool, b: bool) {
        let program = &BOOL_OR;

   let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());
    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[
            Arg::Value(Felt252::new(a)),
            Arg::Value(Felt252::new(b))
        ]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some( &[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();


        compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_xor_proptest(a: bool, b: bool) {
        let program = &BOOL_XOR;


   let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[
            Arg::Value(Felt252::new(a)),
            Arg::Value(Felt252::new(b))
        ]),
        Some(sierra_casm_runner),
        Some(GAS),
    ).left().unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some( &[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();


        compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }
}
