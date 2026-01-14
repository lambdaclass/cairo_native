use crate::common::{
    any_felt, compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS,
};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::{starknet::DummySyscallHandler, Value};
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    static ref INTO_ENTRIES_ARRAY_VALUES: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::nullable::NullableTrait;
        use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};


        fn into_entries_array_values() -> Array<(felt252, Nullable<Array<u8>>, Nullable<Array<u8>>)> {
            let arr0 = array![1, 2, 3];
            let arr1 = array![4, 5, 6];
            let arr2 = array![7, 8, 9];

            let mut dict: Felt252Dict<Nullable<Array<u8>>> = Default::default();
            dict.insert(0, NullableTrait::new(arr0));
            dict.insert(1, NullableTrait::new(arr1));
            dict.insert(2, NullableTrait::new(arr2));

            dict.squash().into_entries()
        }
    };
}

#[test]
fn test_squashed() {
    let program = &INTO_ENTRIES_ARRAY_VALUES;
    let result_vm = run_vm_program(
        program,
        "into_entries_array_values",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "into_entries_array_values",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("into_entries_array_values")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
