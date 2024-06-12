//use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
//use crate::common::{compare_outputs, DEFAULT_GAS};
use crate::common::{compare_outputs, DEFAULT_GAS};
//use cairo_felt::Felt252 as DeprecatedFelt;
use cairo_felt::Felt252 as DeprecatedFelt;
//use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_runner::{Arg, SierraCasmRunner};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::starknet::DummySyscallHandler;
use cairo_native::starknet::DummySyscallHandler;
//use cairo_native::values::JitValue;
use cairo_native::values::JitValue;
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use proptest::prelude::*;
use proptest::prelude::*;
//

//lazy_static! {
lazy_static! {
//    static ref ARRAY_GET: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref ARRAY_GET: (String, Program, SierraCasmRunner) = load_cairo! {
//        use array::ArrayTrait;
        use array::ArrayTrait;
//        use traits::TryInto;
        use traits::TryInto;
//        use core::option::OptionTrait;
        use core::option::OptionTrait;
//

//        fn run_test(value: felt252, idx: felt252) -> felt252 {
        fn run_test(value: felt252, idx: felt252) -> felt252 {
//            let mut numbers: Array<felt252> = ArrayTrait::new();
            let mut numbers: Array<felt252> = ArrayTrait::new();
//

//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            numbers.append(value);
            numbers.append(value);
//            *numbers.at(idx.try_into().unwrap())
            *numbers.at(idx.try_into().unwrap())
//        }
        }
//    };
    };
//}
}
//

//#[test]
#[test]
//fn array_get_test() {
fn array_get_test() {
//    let program = &ARRAY_GET;
    let program = &ARRAY_GET;
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[
        &[
//            Arg::Value(DeprecatedFelt::from(10)),
            Arg::Value(DeprecatedFelt::from(10)),
//            Arg::Value(DeprecatedFelt::from(5)),
            Arg::Value(DeprecatedFelt::from(5)),
//        ],
        ],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(10.into()), JitValue::Felt252(5.into())],
        &[JitValue::Felt252(10.into()), JitValue::Felt252(5.into())],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("run_test").unwrap().id,
        &program.2.find_function("run_test").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
//

//proptest! {
proptest! {
//    #[test]
    #[test]
//    fn array_get_test_proptest(value in any_felt(), idx in 0u32..26) {
    fn array_get_test_proptest(value in any_felt(), idx in 0u32..26) {
//        let program = &ARRAY_GET;
        let program = &ARRAY_GET;
//        let result_vm = run_vm_program(program, "run_test", &[
        let result_vm = run_vm_program(program, "run_test", &[
//            Arg::Value(DeprecatedFelt::from_bytes_be(&value.to_bytes_be())),
            Arg::Value(DeprecatedFelt::from_bytes_be(&value.to_bytes_be())),
//            Arg::Value(DeprecatedFelt::from(idx))
            Arg::Value(DeprecatedFelt::from(idx))
//        ], Some(DEFAULT_GAS as usize)).unwrap();
        ], Some(DEFAULT_GAS as usize)).unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(value), JitValue::Felt252(idx.into())],
            &[JitValue::Felt252(value), JitValue::Felt252(idx.into())],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &program.1,
            &program.1,
//            &program.2.find_function("run_test").unwrap().id,
            &program.2.find_function("run_test").unwrap().id,
//            &result_vm,
            &result_vm,
//            &result_native,
            &result_native,
//        )
        )
//        .unwrap();
        .unwrap();
//    }
    }
//}
}
