#![allow(clippy::items_after_test_module)]

use std::fs;
use std::fs::File;
use std::io::{Read, Seek};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cairo_felt::Felt252;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_runner::{RunResult, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::ProgramParser;
use cfg_match::cfg_match;
use color_eyre::eyre::WrapErr;
use color_eyre::Result;
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Num;
use sierra2mlir::types::DEFAULT_PRIME;
use sierra2mlir::{compile, execute};
use test_case::test_case;

// Tests behaviour of the generated MLIR against the behaviour of starkware's own sierra runner
// Such tests must be an argumentless main function consisting of calls to the function in question

#[test_case("array/append", None)]
#[test_case("array/append", Some(100000))]
#[test_case("array/index_invalid", None)]
#[test_case("array/index_invalid", Some(10000))]
#[test_case("array/pop_front_invalid", None)]
#[test_case("array/pop_front_invalid", Some(10000))]
#[test_case("array/pop_front_valid", None)]
#[test_case("array/pop_front_valid", Some(100000))]
#[test_case("bitwise/and", None)]
#[test_case("bitwise/and", Some(100000))]
#[test_case("bitwise/or", None)]
#[test_case("bitwise/or", Some(100000))]
#[test_case("bitwise/xor", None)]
#[test_case("bitwise/xor", Some(100000))]
#[test_case("bool/and", None)]
#[test_case("bool/and", Some(100000))]
#[test_case("bool/not", None)]
#[test_case("bool/not", Some(100000))]
#[test_case("bool/or", None)]
#[test_case("bool/or", Some(100000))]
#[test_case("bool/to_felt252", None)]
#[test_case("bool/to_felt252", Some(100000))]
#[test_case("bool/xor", None)]
#[test_case("bool/xor", Some(100000))]
#[test_case("enums/enum_init", None)]
#[test_case("enums/enum_init", Some(100000))]
#[test_case("enums/enum_match", None)]
#[test_case("enums/enum_match", Some(100000))]
#[test_case("enums/single_value", None)]
#[test_case("enums/single_value", Some(100000))]
#[test_case("felt_ops/add", None)]
#[test_case("felt_ops/add", Some(100000))]
#[test_case("felt_ops/div", None)]
#[test_case("felt_ops/div", Some(100000))]
#[test_case("felt_ops/felt_is_zero", None)]
#[test_case("felt_ops/felt_is_zero", Some(100000))]
#[test_case("felt_ops/mul", None)]
#[test_case("felt_ops/mul", Some(100000))]
#[test_case("felt_ops/negation", None)]
#[test_case("felt_ops/negation", Some(100000))]
#[test_case("felt_ops/sub", None)]
#[test_case("felt_ops/sub", Some(100000))]
#[test_case("fib_counter", Some(1000000))]
#[test_case("fib_local", Some(1000000))]
#[test_case("nullable/test_nullable", None)]
#[test_case("nullable/test_nullable", Some(100000))]
#[test_case("pedersen", None)]
#[test_case("pedersen", Some(50000))]
#[test_case("poseidon", None)]
#[test_case("poseidon", Some(50000))]
#[test_case("returns/enums", None)]
#[test_case("returns/enums", Some(100000))]
#[test_case("returns/simple", None)]
#[test_case("returns/simple", Some(100000))]
#[test_case("returns/tuple", None)]
#[test_case("returns/tuple", Some(100000))]
#[test_case("structs/basic", None)]
#[test_case("structs/basic", Some(100000))]
#[test_case("structs/bigger", None)]
#[test_case("structs/bigger", Some(100000))]
#[test_case("structs/enum_member", None)]
#[test_case("structs/enum_member", Some(100000))]
#[test_case("structs/nested", None)]
#[test_case("structs/nested", Some(100000))]
#[test_case("uint/compare", None)]
#[test_case("uint/compare", Some(100000))]
#[test_case("uint/consts", None)]
#[test_case("uint/consts", Some(100000))]
#[test_case("uint/downcasts", None)]
#[test_case("uint/downcasts", Some(100000))]
#[test_case("uint/safe_divmod", None)]
#[test_case("uint/safe_divmod", Some(200000))]
#[test_case("uint/uint_addition", None)]
#[test_case("uint/uint_addition", Some(100000))]
#[test_case("uint/uint_subtraction", None)]
#[test_case("uint/uint_subtraction", Some(100000))]
#[test_case("uint/uint_try_from_felt", None)]
#[test_case("uint/uint_try_from_felt", Some(100000))]
#[test_case("uint/upcasts", None)]
#[test_case("uint/upcasts", Some(100000))]
//#[test_case("uint/wide_mul", None)]           // needs #146
//#[test_case("uint/wide_mul", Some(100000))]   // needs #146
#[test_case("gas/available_gas", Some(200))]
#[test_case("unwrap_non_zero", None)]
#[test_case("unwrap_non_zero", Some(100000))]
fn comparison_test(test_name: &str, available_gas: Option<usize>) -> Result<(), String> {
    let program = compile_sierra_program(test_name);
    compile_to_mlir_with_consistency_check(test_name, &program, available_gas);
    let llvm_result = run_mlir(test_name, &program, available_gas)?;

    let casm_result = run_sierra_via_casm(program, available_gas);

    match casm_result {
        Ok(result) => match result.value {
            // Casm runner succeeded
            cairo_lang_runner::RunResultValue::Success(casm_values) => {
                println!("llvm result: {:?}\n", llvm_result);
                println!("Casm result: {:?}\n", casm_values);
                // Since the casm runner succeeded, we expect that llvm program didn't panic
                let (llvm_result, llvm_remaining_gas) = llvm_result.unwrap();
                assert_eq!(
                    casm_values.len(),
                    llvm_result.len(),
                    "Casm values and llvm values are of different lengths"
                );
                // gas_counter from cairo-runner won't be available if the program doesn't use the gas builtin
                // even if it spends constant time gas.
                // it needs to have the gas builtin present in the program, which appears with libfuncs such as withdraw_gas

                if result.gas_counter.is_some() {
                    assert!(
                        available_gas.is_some(),
                        "if cairo-runner returned a gas counter, mlir should too"
                    )
                }
                if available_gas.is_some() && result.gas_counter.is_some() {
                    let casm_gas =
                        result.gas_counter.expect("casm gas counter should exist").to_biguint();
                    let llvm_gas = llvm_remaining_gas.expect("mlir gas counter should exist");
                    assert_eq!(casm_gas, llvm_gas, "remaning gas mismatch");
                }
                let prime = DEFAULT_PRIME.parse::<BigUint>().unwrap();
                for i in 0..casm_values.len() {
                    assert!(
                        llvm_result[i] < prime,
                        "Test no. {} of {} failed. {} >= PRIME. Expected {} (-{})",
                        i + 1,
                        test_name,
                        llvm_result[i],
                        casm_values[i],
                        prime - casm_values[i].to_biguint()
                    );
                    assert_eq!(
                        casm_values[i].to_biguint(),
                        llvm_result[i],
                        "Test no. {} of {} failed. {}(casm) != {}(llvm) (-{} != -{})",
                        i + 1,
                        test_name,
                        casm_values[i],
                        llvm_result[i],
                        prime.clone() - casm_values[i].to_biguint(),
                        prime - llvm_result[i].clone()
                    )
                }
            }
            cairo_lang_runner::RunResultValue::Panic(panic_data) => {
                // The casm runner panicked, so we expect that lli returned a (controlled) failure
                let casm_panic_message = get_string_from_felts(panic_data);
                if llvm_result.is_ok() {
                    panic!("Casm runner panicked with error message: \"{}\", but llvm run returned: {:?}", casm_panic_message, llvm_result.unwrap());
                }
                let llvm_panic_message = llvm_result.unwrap_err();

                assert_eq!(
                    llvm_panic_message, casm_panic_message,
                    "LLvm panic message (lhs) should equal casm panic message (rhs)"
                );
            }
        },
        Err(e) => {
            todo!("Comparison tests where the cairo runner fails: {e}");
        }
    }
    Ok(())
}

fn compile_sierra_program(test_name: &str) -> Program {
    let test_path = Path::new(".").join("tests").join("comparison").join(test_name);
    let sierra_path = test_path.with_extension("sierra");
    let cairo_path = test_path.with_extension("cairo");

    if sierra_path.exists() {
        let sierra_code =
            fs::read_to_string(format!("./tests/comparison/{test_name}.sierra")).unwrap();
        ProgramParser::new().parse(&sierra_code).unwrap()
    } else if cairo_path.exists() {
        let program_ptr = cairo_lang_compiler::compile_cairo_project_at_path(
            &cairo_path,
            CompilerConfig { replace_ids: true, ..Default::default() },
        )
        .expect("Cairo compilation failed");
        let program = Arc::try_unwrap(program_ptr).unwrap();
        fs::write(sierra_path, program.to_string()).unwrap();
        program
    } else {
        panic!("Cannot find {test_name}.sierra or {test_name}.cairo")
    }
}

fn compile_to_mlir_with_consistency_check(
    test_name: &str,
    program: &Program,
    available_gas: Option<usize>,
) {
    let out_dir = get_outdir();
    let test_file_name = flatten_test_name(test_name);
    let compiled_code = compile(program, false, false, true, 1, available_gas).unwrap();
    let optimised_compiled_code = compile(program, false, false, true, 1, available_gas).unwrap();
    let mlir_file = out_dir.join(format!("{test_file_name}.mlir")).display().to_string();
    let optimised_mlir_file =
        out_dir.join(format!("{test_file_name}-opt.mlir")).display().to_string();
    std::fs::write(mlir_file.as_str(), &compiled_code).unwrap();
    std::fs::write(optimised_mlir_file.as_str(), &optimised_compiled_code).unwrap();
    for _ in 0..5 {
        let repeat_compiled_code = compile(program, false, false, true, 1, available_gas).unwrap();
        if compiled_code != repeat_compiled_code {
            let mlir_repeat_file =
                out_dir.join(format!("{test_file_name}-repeat.mlir")).display().to_string();
            std::fs::write(mlir_repeat_file.as_str(), &repeat_compiled_code).unwrap();
        }
        assert_eq!(
            compiled_code, repeat_compiled_code,
            "Repeat compilation produced differing code"
        );
    }
}

// Invokes starkware's runner that compiles sierra to casm and runs it
// This provides us with the intended results to compare against
fn run_sierra_via_casm(program: Program, available_gas: Option<usize>) -> Result<RunResult> {
    let runner = SierraCasmRunner::new(
        program,
        available_gas.map(|_| Default::default()),
        Default::default(),
    )?;

    let func = runner.find_function("::main")?;
    runner
        .run_function(func, &[], available_gas, Default::default())
        .with_context(|| "Failed to run the function.")
}

// Runs the test file via reading the mlir file, compiling it to llir, then invoking lli to run it
#[allow(clippy::type_complexity)]
fn run_mlir(
    test_name: &str,
    program: &Program,
    available_gas: Option<usize>,
) -> Result<Result<(Vec<BigUint>, Option<BigUint>), String>, String> {
    // Allows folders of comparison tests without write producing a file not found
    let test_file_name = flatten_test_name(test_name);

    let out_dir = get_outdir();

    let mut output = String::new();
    let output_path = out_dir.join(format!("{test_file_name}-{}.out", available_gas.is_some()));

    {
        let mut output_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_path)
            .unwrap();
        let fd = output_file.as_raw_fd();
        let engine = execute(program, true, fd, available_gas).unwrap();
        unsafe {
            engine.invoke_packed("main", &mut []).unwrap();
        }
        output_file.seek(std::io::SeekFrom::Start(0)).unwrap();
        output_file.read_to_string(&mut output).unwrap();
    }

    parse_llvm_result(&output, available_gas.is_some())
        .ok_or("Unable to parse llvm result".to_string())
}

// Parses the human-readable output from running the llir code into a raw list of outputs
// Option is for whether it was parsable
// Result is for whether the run succeeded or failed
#[allow(clippy::type_complexity)]
fn parse_llvm_result(
    res: &str,
    has_gas: bool,
) -> Option<Result<(Vec<BigUint>, Option<BigUint>), String>> {
    println!("Parsing llvm result: '{}', length: {}", res, res.chars().count());
    let lines = res.split('\n').collect_vec();
    if !lines.is_empty() && lines[0] == "Success" {
        let mut base = 1;
        let gas = if has_gas {
            base += 1;
            Some(BigUint::from_str_radix(&lines[1]["Remaining gas: ".len()..], 16).unwrap())
        } else {
            None
        };
        Some(Ok((
            lines
                .iter()
                .skip(base)
                .filter(|s| !s.is_empty())
                .map(|x| BigUint::from_str_radix(x, 16).unwrap())
                .collect(),
            gas,
        )))
    } else if !lines.is_empty() && lines[0] == "Program panicked" {
        Some(Err(lines[1..].join("\n")))
    } else {
        None
    }
}

fn flatten_test_name(test_name: &str) -> String {
    test_name.replace('_', "__").replace('/', "_")
}

fn get_outdir() -> PathBuf {
    Path::new(".").join("tests").join("comparison").join("out")
}

fn get_string_from_felts(felts: Vec<Felt252>) -> String {
    let char_data = felts.iter().flat_map(|felt| felt.to_be_bytes()).collect_vec();
    println!("Parsing char_data {:?}", char_data);
    let zero_count_opt = char_data.iter().position(|c| *c != 0);
    println!("Zero count {:?}", zero_count_opt);

    if let Some(zero_count) = zero_count_opt {
        String::from_utf8_lossy(&char_data[zero_count..char_data.len()]).to_string()
    } else {
        "".to_string()
    }
}

pub const fn library_preload_env_var() -> &'static str {
    cfg_match! {
        target_os = "linux" => "LD_PRELOAD",
        target_os = "macos" => "DYLD_INSERT_LIBRARIES",
        _ => compile_error!("Unsupported OS."),
    }
}
