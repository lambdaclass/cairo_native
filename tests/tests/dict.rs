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
    fn dict_get_insert_proptest(a in any_felt(), b in any_felt()) {
        let program = &DICT_GET_INSERT;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(Felt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(Felt::from_bytes_be(&b.clone().to_bytes_be()))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a), Value::Felt252(b)],
            Some(DEFAULT_GAS as u128),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
