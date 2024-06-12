//use crate::common::{compare_outputs, run_native_program, run_vm_program, DEFAULT_GAS};
use crate::common::{compare_outputs, run_native_program, run_vm_program, DEFAULT_GAS};
//use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_runner::SierraCasmRunner;
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::starknet::DummySyscallHandler;
use cairo_native::starknet::DummySyscallHandler;
//use std::{fs::File, io::BufReader};
use std::{fs::File, io::BufReader};
//use test_case::test_case;
use test_case::test_case;
//

//#[track_caller]
#[track_caller]
//fn compare_inputless_function(function_name: &str) {
fn compare_inputless_function(function_name: &str) {
//    // Load file compiled using `scarb build``
    // Load file compiled using `scarb build``
//    let file = File::open("tests/alexandria/target/dev/alexandria.sierra.json").unwrap();
    let file = File::open("tests/alexandria/target/dev/alexandria.sierra.json").unwrap();
//    let reader = BufReader::new(file);
    let reader = BufReader::new(file);
//    let program: Program = serde_json::from_reader(reader).unwrap();
    let program: Program = serde_json::from_reader(reader).unwrap();
//    let module_name = "alexandria";
    let module_name = "alexandria";
//    let runner = SierraCasmRunner::new(
    let runner = SierraCasmRunner::new(
//        program.clone(),
        program.clone(),
//        Some(Default::default()),
        Some(Default::default()),
//        Default::default(),
        Default::default(),
//        None,
        None,
//    )
    )
//    .unwrap();
    .unwrap();
//

//    let program: (String, Program, SierraCasmRunner) = (module_name.to_string(), program, runner);
    let program: (String, Program, SierraCasmRunner) = (module_name.to_string(), program, runner);
//    let program = &program;
    let program = &program;
//

//    let result_vm =
    let result_vm =
//        run_vm_program(program, function_name, &[], Some(DEFAULT_GAS as usize)).unwrap();
        run_vm_program(program, function_name, &[], Some(DEFAULT_GAS as usize)).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        function_name,
        function_name,
//        &[],
        &[],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function(function_name).unwrap().id,
        &program.2.find_function(function_name).unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .expect("compare error");
    .expect("compare error");
//}
}
//

//// alexandria_math
// alexandria_math
//#[test_case("fib")]
#[test_case("fib")]
//#[test_case("karatsuba")]
#[test_case("karatsuba")]
//#[test_case("armstrong_number")]
#[test_case("armstrong_number")]
//#[test_case("collatz_sequence")]
#[test_case("collatz_sequence")]
//#[test_case("aliquot_sum")]
#[test_case("aliquot_sum")]
//#[test_case("extended_euclidean_algorithm")]
#[test_case("extended_euclidean_algorithm")]
//// alexandria_data_structures
// alexandria_data_structures
//#[test_case("vec")]
#[test_case("vec")]
//#[test_case("stack")]
#[test_case("stack")]
//#[test_case("queue")]
#[test_case("queue")]
//#[test_case("bit_array")]
#[test_case("bit_array")]
//// alexandria_encoding
// alexandria_encoding
//#[test_case("base64_encode")]
#[test_case("base64_encode")]
//#[test_case("reverse_bits")]
#[test_case("reverse_bits")]
//#[test_case("reverse_bytes")]
#[test_case("reverse_bytes")]
//fn test_cases(function_name: &str) {
fn test_cases(function_name: &str) {
//    compare_inputless_function(function_name)
    compare_inputless_function(function_name)
//}
}
