use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
use cairo_felt::Felt252 as DeprecatedFelt;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use num_bigint::BigUint;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;
use std::str::FromStr;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref EC_POINT_TRY_NEW: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::{ec::{ec_point_try_new_nz, EcPoint}};
        use core::zeroable::NonZero;

        fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
            ec_point_try_new_nz(x, y)
        }
    };
    static ref EC_POINT_FROM_X: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::{ec::{ec_point_from_x_nz, EcPoint}};
        use core::zeroable::NonZero;

        fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
            ec_point_from_x_nz(x)
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
    let result_native = run_native_program(program, "run_test", &[]);

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn ec_point_from_x_big() {
    let x = DeprecatedFelt::new(
        BigUint::from_str(
            "10503791839462130483045092717244804953879649418761481950933471772092536173",
        )
        .unwrap(),
    );
    let program = &EC_POINT_FROM_X;
    let result_vm =
        run_vm_program(program, "run_test", &[Arg::Value(x.clone())], Some(GAS)).unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[JITValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
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
fn ec_point_from_x_small() {
    let x = DeprecatedFelt::new(BigUint::from_str("1234").unwrap());
    let program = &EC_POINT_FROM_X;
    let result_vm =
        run_vm_program(program, "run_test", &[Arg::Value(x.clone())], Some(GAS)).unwrap();
    let result_native = run_native_program(
        program,
        "run_test",
        &[JITValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
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
    fn ec_point_try_new_proptest(a in any_felt(), b in any_felt()) {
        let program = &EC_POINT_TRY_NEW;
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

    #[test]
    fn ec_point_from_x_proptest(a in any_felt()) {
        let program = &EC_POINT_FROM_X;
        let result_vm = run_vm_program(
            program,
            "run_test",
            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be()))],
            Some(GAS),
        )
        .unwrap();
        let result_native = run_native_program(program, "run_test", &[JITValue::Felt252(a)]);

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
