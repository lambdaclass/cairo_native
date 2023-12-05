use crate::common::{any_felt252, load_cairo, run_native_or_vm_program};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;
use std::borrow::Borrow;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref DICT_GET_INSERT: (String, Program, SierraCasmRunner) = load_cairo! {
        use traits::Default;
        use dict::Felt252DictTrait;

        fn run_test(key: felt252, val: felt252) -> felt252 {
            let mut dict: Felt252Dict<felt252> = Default::default();
            dict.insert(key, val);
            dict.get(key)
        }
    };
}

proptest! {
    #[test]
    #[ignore = "gas mismatch in dicts"]
    fn dict_get_insert_proptest(a in any_felt252(), b in any_felt252()) {
        let program = &DICT_GET_INSERT;

        let (program_for_args, sierra_casm_runner) =
        ((program.0.clone(), program.1.clone()), program.2.borrow());

    let result_vm = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        None,
        Some(&[Arg::Value(a.clone()), Arg::Value(b.clone())]),
        Some(sierra_casm_runner),
        Some(GAS),
    )
    .left()
    .unwrap()
    .unwrap();

    let result_native = run_native_or_vm_program(
        &program_for_args,
        "run_test",
        Some(&[JITValue::Felt252(a), JITValue::Felt252(b)]),
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
}
