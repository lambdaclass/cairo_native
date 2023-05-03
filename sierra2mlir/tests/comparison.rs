#![allow(clippy::items_after_test_module)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::{env, fs};

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
use sierra2mlir::compile;
use sierra2mlir::types::DEFAULT_PRIME;
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
//#[test_case("uint/wide_mul", None)]
//#[test_case("uint/wide_mul", Some(100000))]
#[test_case("gas/available_gas", Some(200))]
fn comparison_test(test_name: &str, available_gas: Option<usize>) -> Result<(), String> {
    let program = compile_sierra_program(test_name);
    compile_to_mlir_with_consistency_check(test_name, &program, available_gas);
    let llvm_result = run_mlir(test_name)?;

    let casm_result = run_sierra_via_casm(program, available_gas);

    match casm_result {
        Ok(result) => match result.value {
            // Casm runner succeeded
            cairo_lang_runner::RunResultValue::Success(casm_values) => {
                println!("llvm result: {:?}\n", llvm_result);
                println!("Casm result: {:?}\n", casm_values);
                // Since the casm runner succeeded, we expect that llvm program didn't panic
                let llvm_result = llvm_result.unwrap();
                assert_eq!(
                    casm_values.len(),
                    llvm_result.len(),
                    "Casm values and llvm values are of different lengths"
                );
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
            todo!("Comparison tests where the cairo runner fails:\n{e}");
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
    let runner = SierraCasmRunner::new(program, Some(Default::default()), Default::default())
        .with_context(|| "Failed setting up runner.")?;

    let func = runner.find_function("::main")?;
    runner
        .run_function(func, &[], available_gas, Default::default())
        .with_context(|| "Failed to run the function.")
}

// Runs the test file via reading the mlir file, compiling it to llir, then invoking lli to run it
fn run_mlir(test_name: &str) -> Result<Result<Vec<BigUint>, String>, String> {
    let out_dir = get_outdir();

    // Allows folders of comparison tests without write producing a file not found
    let test_file_name = flatten_test_name(test_name);

    let mlir_file = out_dir.join(format!("{test_file_name}.mlir")).display().to_string();
    let optimised_mlir_file =
        out_dir.join(format!("{test_file_name}-opt.mlir")).display().to_string();
    let output_file = out_dir.join(format!("{test_file_name}.ll")).display().to_string();
    let optimised_output_file =
        out_dir.join(format!("{test_file_name}-opt.ll")).display().to_string();

    let result = run_mlir_file_via_llvm(&mlir_file, &output_file)?;
    let optimised_result = run_mlir_file_via_llvm(&optimised_mlir_file, &optimised_output_file)?;

    assert_eq!(
        result, optimised_result,
        "Compiling with the optimised flag produced different behaviour"
    );

    Ok(result)
}

// Outer result is for whether lli succeeded and produced a parsable result
// Inner result is for whether the program panicked
fn run_mlir_file_via_llvm(
    mlir_file: &str,
    output_file: &str,
) -> Result<Result<Vec<BigUint>, String>, String> {
    let mlir_prefix = find_mlir_prefix();
    let lli_path = mlir_prefix.join("bin").join("lli");
    let mlir_translate_path = mlir_prefix.join("bin").join("mlir-translate");

    let mlir_output = Command::new(mlir_translate_path)
        .arg("--mlir-to-llvmir")
        .arg("-o")
        .arg(output_file)
        .arg(mlir_file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();

    if !mlir_output.stdout.is_empty() || !mlir_output.stderr.is_empty() {
        println!(
            "Mlir_output ({}):\n    stdout: {}\n    stderr: {}",
            mlir_file,
            String::from_utf8(mlir_output.stdout).unwrap(),
            String::from_utf8(mlir_output.stderr).unwrap()
        );
    }

    let ld_env = library_preload_env_var();
    let lli_cmd = Command::new(lli_path)
        .arg(output_file)
        .env(ld_env, env!("S2M_UTILS_PATH"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let lli_output = lli_cmd.wait_with_output().unwrap();
    dbg!(lli_output.status);

    if !lli_output.stderr.is_empty() {
        return Err(format!(
            "lli failed with output: {}",
            String::from_utf8(lli_output.stderr).unwrap()
        ));
    }

    let output = std::str::from_utf8(&lli_output.stdout).unwrap().trim();

    parse_llvm_result(output).ok_or("Unable to parse llvm result".to_string())
}

// Parses the human-readable output from running the llir code into a raw list of outputs
// Option is for whether it was parsable
// Result is for whether the run succeeded or failed
fn parse_llvm_result(res: &str) -> Option<Result<Vec<BigUint>, String>> {
    println!("Parsing llvm result: '{}', length: {}", res, res.chars().count());
    let lines = res.split('\n').collect_vec();
    if !lines.is_empty() && lines[0] == "Success" {
        Some(Ok(lines
            .iter()
            .skip(1)
            .filter(|s| !s.is_empty())
            .map(|x| BigUint::from_str_radix(x, 16).unwrap())
            .collect()))
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

fn find_mlir_prefix() -> PathBuf {
    match env::var_os("MLIR_SYS_160_PREFIX") {
        Some(x) => Path::new(x.to_str().unwrap()).to_owned(),
        None => {
            let cmd_output = Command::new("../scripts/find-llvm.sh")
                .stdout(Stdio::piped())
                .spawn()
                .unwrap()
                .wait_with_output()
                .unwrap();

            PathBuf::from(String::from_utf8(cmd_output.stdout).unwrap().trim())
        }
    }
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
