use crate::common::{
    any_felt, compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS,
};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::{include_program, starknet::DummySyscallHandler, Value};
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    static ref DICT_GET_INSERT: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/dict_get_insert.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "dict_get_insert".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref SNAPSHOT_LOOP: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/snapshot_loop.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "snapshot_loop".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
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
