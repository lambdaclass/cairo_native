use crate::common::GAS;
use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use common::{compare_outputs, run_native_program, run_vm_program};
use std::{fs::File, io::BufReader};
use test_case::test_case;

mod common;
// alexandria_math
#[test_case("fib")]
#[test_case("karatsuba" => ignore["SIGKILL"])]
#[test_case("armstrong_number")]
#[test_case("aliquot_sum" => ignore["System out of memory"])]
#[test_case("collatz_sequence" => ignore["Result mismatch"])]
#[test_case("extended_euclidean_algorithm")]
fn test_cases(function_name: &str) {
    compare_inputless_function(function_name)
}

#[track_caller]
fn compare_inputless_function(function_name: &str) {
    // Load file compiled using `scarb build``
    let file = File::open("tests/alexandria/target/dev/alexandria.sierra.json").unwrap();
    let reader = BufReader::new(file);
    let program: Program = serde_json::from_reader(reader).unwrap();
    let module_name = "alexandria";
    let runner = SierraCasmRunner::new(
        program.clone(),
        Some(Default::default()),
        Default::default(),
    )
    .unwrap();

    let program: (String, Program, SierraCasmRunner) = (module_name.to_string(), program, runner);
    let program = &program;

    let result_vm = run_vm_program(program, function_name, &[], Some(GAS as usize)).unwrap();

    let result_native = run_native_program(program, function_name, &[]);

    compare_outputs(
        &program.1,
        &program.2.find_function(function_name).unwrap().id,
        &result_vm,
        &result_native,
    )
    .expect("compare error");
}
