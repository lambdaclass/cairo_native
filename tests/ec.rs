use crate::common::{any_felt252, feltn, load_cairo, run_native_program, run_vm_program};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use common::compare_outputs;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use proptest::prelude::*;
use serde_json::json;
use std::str::FromStr;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref EC_POINT_TRY_NEW: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::{ec::{ec_point_try_new, EcPoint}};

        fn run_test(x: felt252, y: felt252) -> Option<EcPoint> {
            ec_point_try_new(x, y)
        }
    };
    static ref EC_POINT_FROM_X: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::{ec::{ec_point_from_x, EcPoint}};

        fn run_test(x: felt252) -> Option<EcPoint> {
            ec_point_from_x(x)
        }
    };
    static ref EC_POINT_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::ec::{ec_point_zero, EcPoint};

        fn run_test() -> EcPoint {
            ec_point_zero()
        }
    };
}

#[test]
fn ec_point_zero() {
    let program = &EC_POINT_ZERO;
    let result_vm = run_vm_program(program, "run_test", &[], Some(GAS)).unwrap();
    let result_native = run_native_program(program, "run_test", json!([]));

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[ignore = "TODO: possible bug in ec_point_from_x_nz"]
#[test]
fn ec_point_from_x() {
    let x = Felt252::new(
        BigUint::from_str(
            "10503791839462130483045092717244804953879649418761481950933471772092536173",
        )
        .unwrap(),
    );
    let program = &EC_POINT_FROM_X;
    let result_vm =
        run_vm_program(program, "run_test", &[Arg::Value(x.clone())], Some(GAS)).unwrap();
    let result_native =
        run_native_program(program, "run_test", json!([null, feltn(x.to_bigint())]));

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
    fn ec_point_try_new_proptest(a in any_felt252(), b in any_felt252()) {
        let program = &EC_POINT_TRY_NEW;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.clone()), Arg::Value(b.clone())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a.to_bigint()), feltn(b.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[ignore = "TODO: possible bug in ec_point_from_x_nz"]
    #[test]
    fn ec_point_from_x_proptest(a in any_felt252()) {
        let program = &EC_POINT_FROM_X;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(a.clone())],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", json!([null, feltn(a.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
