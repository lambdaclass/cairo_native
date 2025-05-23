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
    static ref SNAPSHOT_LOOP: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::dict::Felt252Dict;

        fn run_test() {
            let mut dict: Felt252Dict<u64> = Default::default();

            for number in 0..50_u64 {
                let snapshot = @dict;

                let key = number.try_into().unwrap();
                dict.insert(key, number);

                drop(snapshot)
            }
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
            vec![Arg::Value(Felt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(Felt::from_bytes_be(&b.clone().to_bytes_be()))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a), Value::Felt252(b)],
            Some(DEFAULT_GAS),
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

#[test]
fn dict_snapshot_loop() {
    let program = &SNAPSHOT_LOOP;
    run_native_program(
        program,
        "run_test",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );
}
