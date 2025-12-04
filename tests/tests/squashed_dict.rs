use crate::common::{compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::{starknet::DummySyscallHandler, Value};
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    static ref INTO_ENTRIES_U8_VALUES: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};

        fn into_entries_u8_values() -> Array<(felt252, u8, u8)> {
            let mut dict: Felt252Dict<u8> = Default::default();
            dict.insert(0, 0);
            dict.insert(1, 1);
            dict.insert(2, 2);
            dict.squash().into_entries()
        }
    };
    static ref INTO_ENTRIES_U32_VALUES: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};

        fn into_entries_u32_values() -> Array<(felt252, u32, u32)> {
            let mut dict: Felt252Dict<u32> = Default::default();
            dict.insert(0, 0);
            dict.insert(1, 1);
            dict.insert(2, 2);
            dict.squash().into_entries()
        }
    };
    static ref INTO_ENTRIES_U128_VALUES: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};

        fn into_entries_u128_values() -> Array<(felt252, u128, u128)> {
            let mut dict: Felt252Dict<u128> = Default::default();
            dict.insert(0, 0);
            dict.insert(1, 1);
            dict.insert(2, 2);
            dict.squash().into_entries()
        }
    };
    static ref INTO_ENTRIES_FELT252_VALUES: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictTrait};

        fn into_entries_felt252_values() -> Array<(felt252, felt252, felt252)> {
            let mut dict: Felt252Dict<felt252> = Default::default();
            dict.insert(0, 0);
            dict.insert(1, 1);
            dict.insert(2, 2);
            dict.squash().into_entries()
        }
    };
}

#[test]
fn test_into_entries_u8_values() {
    let program = &INTO_ENTRIES_U8_VALUES;
    let endpoint = "into_entries_u8_values";
    let a = true;
    let result_vm = run_vm_program(program, endpoint, vec![], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(
        program,
        endpoint,
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("into_entries_u8_values")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_into_entries_u32_values() {
    let program = &INTO_ENTRIES_U32_VALUES;
    let endpoint = "into_entries_u32_values";
    let a = true;
    let result_vm = run_vm_program(program, endpoint, vec![], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(
        program,
        endpoint,
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("into_entries_u32_values")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_into_entries_u128_values() {
    let program = &INTO_ENTRIES_U128_VALUES;
    let endpoint = "into_entries_u128_values";
    let a = true;
    let result_vm = run_vm_program(program, endpoint, vec![], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(
        program,
        endpoint,
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("into_entries_u128_values")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_into_entries_felt252_values() {
    let program = &INTO_ENTRIES_FELT252_VALUES;
    let endpoint = "into_entries_felt252_values";
    let a = true;
    let result_vm = run_vm_program(program, endpoint, vec![], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(
        program,
        endpoint,
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("into_entries_felt252_values")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
