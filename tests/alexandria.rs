use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use common::{compare_outputs, run_native_program, run_vm_program, DEFAULT_GAS};
use std::{fs::File, io::BufReader};
use test_case::test_case;

mod common;

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

    let result_vm =
        run_vm_program(program, function_name, &[], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(program, function_name, &[], Some(DEFAULT_GAS as u128));

    compare_outputs(
        &program.1,
        &program.2.find_function(function_name).unwrap().id,
        &result_vm,
        &result_native,
    )
    .expect("compare error");
}

// alexandria_math
#[test_case("fib")]
#[test_case("karatsuba" => ignore["System out of memory"])]
#[test_case("armstrong_number")]
#[test_case("aliquot_sum" => ignore["System out of memory"])]
#[test_case("collatz_sequence" => ignore["Result mismatch"])]
#[test_case("extended_euclidean_algorithm")]
// alexandria_data_structures
#[test_case("vec" => ignore["Gas mismatch"])]
#[test_case("stack" => ignore["Gas mismatch"])]
#[test_case("queue")]
#[test_case("bit_array" => ignore["Gas mismatch"])]
// alexandria_encoding
#[test_case("base64_encode" => ignore["Gas mismatch"])]
#[test_case("reverse_bits" => ignore["Invalid MlIR"])]
#[test_case("reverse_bytes"=> ignore["Invalid MlIR"])]
fn test_cases(function_name: &str) {
    compare_inputless_function(function_name)
}
