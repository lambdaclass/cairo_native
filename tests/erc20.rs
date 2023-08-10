use crate::common::{any_felt252, feltn, load_cairo, run_native_program, run_vm_program};
use cairo_felt::Felt252;
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;
use serde_json::json;

mod common;

const GAS: usize = usize::MAX;

#[test]
fn erc20_compile_test() {


    let program = &common::load_starknet_path("programs/erc20.cairo");
    /*
    let result_vm = run_vm_program(
        program,
        "run_test",
        &[Arg::Value(Felt252::new(10)), Arg::Value(Felt252::new(5))],
        Some(GAS),
    )
    .unwrap();
    let result_native = run_native_program(program, "run_test", json!([null, feltn(10), feltn(5)]));

    compare_outputs(
        &program.1,
        &program.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
*/
}
