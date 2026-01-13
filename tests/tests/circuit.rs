use crate::common::{compare_outputs, DEFAULT_GAS};
use crate::common::{get_compiled_program, load_cairo, run_native_program, run_vm_program};
use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use cairo_native::starknet::DummySyscallHandler;
use cairo_native::{include_program, Value};
use lazy_static::lazy_static;

lazy_static! {
    // Taken from: https://github.com/starkware-libs/sequencer/blob/7ee6f4c8a81def87402c626c9d72a33c74bc3243/crates/blockifier/feature_contracts/cairo1/test_contract.cairo#L656
    static ref TEST: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/circuit.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "circuit".to_string();
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

lazy_static! {
    static ref GARAGA_CIRCUITS: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/garaga_circuits.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "garaga_circuits".to_string();
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

lazy_static! {
    // Taken from: https://github.com/kkrt-labs/kakarot/blob/563af42d5fe9888f8f49cf22003d2085612bf42c/cairo/kakarot-ssj/crates/evm/src/precompiles/ec_operations/ec_add.cairo#L143
    static ref KAKAROT_CIRCUIT: (String, Program, SierraCasmRunner) = {
        let versioned_program =
            include_program!("test_data_artifacts/programs/kakarot_circuit.sierra.json");
        let program = versioned_program.into_v1().unwrap().program;
        let module_name = "kakarot_circuit".to_string();
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

// lazy_static! {
//     static ref BIG_CIRCUIT: (String, Program, SierraCasmRunner) = {
//         let versioned_program =
//             include_program!("test_data_artifacts/programs/big_circuit.sierra.json");
//         let program = versioned_program.into_v1().unwrap().program;
//         let module_name = "big_circuit".to_string();
//         let runner = SierraCasmRunner::new(
//             program.clone(),
//             Some(Default::default()),
//             Default::default(),
//             None,
//         )
//         .unwrap();
//         (module_name, program, runner)
//     };
// }

#[test]
fn test_circuit_guarantee_first_limb() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_guarantee_first_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_first_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_first_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_guarantee_last_limb() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_guarantee_last_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_last_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_last_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_guarantee_middle_limb() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_guarantee_middle_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_middle_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_middle_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_add() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_add",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_add",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_add").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_sub() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_sub",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_sub",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_sub").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_mul() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_mul",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_mul",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_mul").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_inv() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_inv",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_inv",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_inv").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_full() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_full",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_full",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_full").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_fail() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_circuit_fail",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_fail",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_fail").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_into_u96_guarantee() {
    let program = &get_compiled_program("circuit");

    let result_vm = run_vm_program(
        program,
        "test_into_u96_guarantee",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_into_u96_guarantee",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_into_u96_guarantee")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_y_inv_x_neg_over_y_bn254() {
    let program = &get_compiled_program("garaga_circuits");

    let result_vm = run_vm_program(
        program,
        "compute_yInvXnegOverY_BN254",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "compute_yInvXnegOverY_BN254",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("compute_yInvXnegOverY_BN254")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_batch_3_mod_bn254() {
    let program = &get_compiled_program("garaga_circuits");

    let result_vm = run_vm_program(
        program,
        "batch_3_mod_bn254",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "batch_3_mod_bn254",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("batch_3_mod_bn254").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_add_ec_points_g2() {
    let program = &get_compiled_program("garaga_circuits");

    let result_vm = run_vm_program(
        program,
        "run_ADD_EC_POINTS_G2_circuit",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "run_ADD_EC_POINTS_G2_circuit",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("run_ADD_EC_POINTS_G2_circuit")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

// NOTE: Since Cairo 2.14.0-dev.1, the BIG_CIRCUIT program takes forever to
// compile to Sierra. Enable this test once fixed.
// #[test]
// #[ignore]
// fn test_circuit_clear_cofactor_bls12_381() {
//     let program = &BIG_CIRCUIT;

//     let result_vm = run_vm_program(
//         program,
//         "run_CLEAR_COFACTOR_BLS12_381_circuit",
//         vec![],
//         Some(DEFAULT_GAS as usize),
//     )
//     .unwrap();

//     let result_native = run_native_program(
//         program,
//         "run_CLEAR_COFACTOR_BLS12_381_circuit",
//         &[],
//         Some(DEFAULT_GAS),
//         Option::<DummySyscallHandler>::None,
//     );

//     compare_outputs(
//         &program.1,
//         &program
//             .2
//             .find_function("run_CLEAR_COFACTOR_BLS12_381_circuit")
//             .unwrap()
//             .id,
//         &result_vm,
//         &result_native,
//     )
//     .unwrap();
// }

#[test]
fn test_circuit_add_ec_point_unchecked() {
    let program = &get_compiled_program("kakarot_circuit");

    let result_vm = run_vm_program(
        program,
        "add_ec_point_unchecked",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "add_ec_point_unchecked",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("add_ec_point_unchecked")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
