use crate::common::{any_felt252, load_cairo, run_native_program, run_vm_program};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JitValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref ARRAY_GET: (String, Program, SierraCasmRunner) = load_cairo! {
        use array::ArrayTrait;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn run_test(value: felt252, idx: felt252) -> felt252 {
            let mut numbers: Array<felt252> = ArrayTrait::new();

            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            numbers.append(value);
            *numbers.at(idx.try_into().unwrap())
        }
    };
}

#[test]
fn array_get_test() {
    let program = &ARRAY_GET;
    let result_vm = run_vm_program(
        program,
        "run_test",
        &[Arg::Value(Felt252::new(10)), Arg::Value(Felt252::new(5))],
        Some(GAS),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[JitValue::Felt252(10.into()), JitValue::Felt252(5.into())],
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
    fn array_get_test_proptest(value in any_felt252(), idx in 0u32..26) {
        let program = &ARRAY_GET;
        let result_vm = run_vm_program(program, "run_test", &[
            Arg::Value(value.clone()),
            Arg::Value(Felt252::new(idx))
        ], Some(GAS)).unwrap();
        let result_native = run_native_program(program, "run_test", &[JitValue::Felt252(value), JitValue::Felt252(idx.into())]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }
}
