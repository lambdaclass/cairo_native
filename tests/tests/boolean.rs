use crate::common::{compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::{include_program, starknet::DummySyscallHandler, Value};
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    static ref FELT252_TO_BOOL: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/felt252_to_bool.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "felt252_to_bool".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref BOOL_NOT: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/bool_not.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "bool_not".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref BOOL_AND: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/bool_and.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "bool_and".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref BOOL_OR: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/bool_or.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "bool_or".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref BOOL_XOR: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/bool_xor.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "bool_xor".to_string();
        let runner = SierraCasmRunner::new(
            program.clone(),
            Some(Default::default()),
            Default::default(),
            None,
        )
        .unwrap();
        (module_name, program, runner)
    };
    static ref BOOL_TO_FELT252: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/bool_to_felt252.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "bool_to_felt252".to_string();
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

#[test]
fn felt252_to_bool_bug() {
    let program = &FELT252_TO_BOOL;
    let a = true;
    let result_vm = run_vm_program(
        program,
        "run_test",
        vec![Arg::Value(Felt::from(a))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[Value::Felt252(a.into())],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();

    let a = false;
    let result_vm = run_vm_program(
        program,
        "run_test",
        vec![Arg::Value(Felt::from(a))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[Value::Felt252(a.into())],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
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
    fn bool_to_felt252_proptest(a: bool) {
        let program = &BOOL_TO_FELT252;
        let result_vm = run_vm_program(program, "run_test", vec![
            Arg::Value(Felt::from(a)),
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_not_proptest(a: bool) {
        let program = &BOOL_NOT;
        let result_vm = run_vm_program(program, "run_test", vec![
            Arg::Value(Felt::from(a)),
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_and_proptest(a: bool, b: bool) {
        let program = &BOOL_AND;
        let result_vm = run_vm_program(program, "run_test", vec![
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_or_proptest(a: bool, b: bool) {
        let program = &BOOL_OR;
        let result_vm = run_vm_program(program, "run_test", vec![
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }

    #[test]
    fn bool_xor_proptest(a: bool, b: bool) {
        let program = &BOOL_XOR;
        let result_vm = run_vm_program(program, "run_test", vec![
            Arg::Value(Felt::from(a)),
            Arg::Value(Felt::from(b))
        ], Some(DEFAULT_GAS as usize)).unwrap();
        let result_native = run_native_program(
            program,
            "run_test",
            &[Value::Felt252(a.into()), Value::Felt252(b.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )
        .unwrap();
    }
}
