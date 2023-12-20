use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
use cairo_felt::Felt252 as DeprecatedFelt;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;

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
    fn dict_get_insert_proptest(a in any_felt(), b in any_felt()) {
        let program = &DICT_GET_INSERT;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(DeprecatedFelt::from_bytes_be(&b.clone().to_bytes_be()))],
            Some(GAS),
        )
        .unwrap();

        let result_native = run_native_program(program, "run_test", &[JITValue::Felt252(a), JITValue::Felt252(b)]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
