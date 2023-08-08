use crate::common::{any_felt252, feltn, load_cairo, run_native_program, run_vm_program};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;
use serde_json::json;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref EC_POINT_TRY_NEW: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::{ec::{ec_point_try_new, EcPoint}};

        fn run_test(x: felt252, y: felt252) -> Option<EcPoint> {
            ec_point_try_new(x, y)
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
        let result_native = run_native_program(program, "run_test", json!([feltn(a.to_bigint()), feltn(b.to_bigint())]));

        compare_outputs(
            &program.1,
            &program.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
