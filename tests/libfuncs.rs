mod common;

use crate::common::{compare_outputs, load_cairo, run_native_or_vm_program};
use std::borrow::Borrow;

#[test]
fn enum_init() {
    let program = load_cairo! {
        enum MySmallEnum {
            A: felt252,
        }

        enum MyEnum {
            A: felt252,
            B: u8,
            C: u16,
            D: u32,
            E: u64,
        }

        fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
            (
                MySmallEnum::A(-1),
                MyEnum::A(5678),
                MyEnum::B(90),
                MyEnum::C(9012),
                MyEnum::D(34567890),
                MyEnum::E(1234567890123456),
            )
        }
    };

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[]),
        Some(sierra_casm_runner),
        None,
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "run_test", Some(&[]), None, None, None)
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
fn enum_match() {
    let program = load_cairo! {
        enum MyEnum {
            A: felt252,
            B: u8,
            C: u16,
            D: u32,
            E: u64,
        }

        fn match_a() -> felt252 {
            let x = MyEnum::A(5);
            match x {
                MyEnum::A(x) => x,
                MyEnum::B(_) => 0,
                MyEnum::C(_) => 1,
                MyEnum::D(_) => 2,
                MyEnum::E(_) => 3,
            }
        }

        fn match_b() -> u8 {
            let x = MyEnum::B(5_u8);
            match x {
                MyEnum::A(_) => 0_u8,
                MyEnum::B(x) => x,
                MyEnum::C(_) => 1_u8,
                MyEnum::D(_) => 2_u8,
                MyEnum::E(_) => 3_u8,
            }
        }
    };

    let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "match_a",
        None,
        Some(&[]),
        Some(sierra_casm_runner),
        None,
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "match_a", Some(&[]), None, None, None)
            .right()
            .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("match_a").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "match_b",
        None,
        Some(&[]),
        Some(sierra_casm_runner),
        None,
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native =
        run_native_or_vm_program(&program_for_args, "match_b", Some(&[]), None, None, None)
            .right()
            .unwrap();

    compare_outputs(
        &program_for_args.1,
        &sierra_casm_runner.find_function("match_b").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
