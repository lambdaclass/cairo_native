use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cairo_lang_runner::{RunResult, SierraCasmRunner};
use cairo_lang_sierra::ProgramParser;
use color_eyre::eyre::WrapErr;
use color_eyre::Result;
use num_bigint::BigUint;
use num_traits::Num;
use sierra2mlir::compile;
use sierra2mlir::types::DEFAULT_PRIME;
use test_case::test_case;

// Tests behaviour of the generated MLIR against the behaviour of starkware's own sierra runner
// Such tests must be an argumentless main function consisting of calls to the function in question

#[test_case("array/example_array")]
#[test_case("fib_counter")]
#[test_case("fib_local")]
#[test_case("bitwise/and")]
#[test_case("bitwise/or")]
#[test_case("bitwise/xor")]
#[test_case("bool/and")]
#[test_case("bool/not")]
#[test_case("bool/or")]
#[test_case("bool/to_felt252")]
#[test_case("bool/xor")]
#[test_case("felt_ops/add")]
#[test_case("felt_ops/sub")]
#[test_case("felt_ops/mul")]
#[test_case("felt_ops/negation")]
#[test_case("felt_ops/felt_is_zero")]
#[test_case("enums/enum_init")]
#[test_case("enums/enum_match")]
#[test_case("enums/single_value")]
#[test_case("returns/simple")]
#[test_case("returns/tuple")]
#[test_case("returns/enums")]
#[test_case("structs/basic")]
#[test_case("structs/bigger")]
#[test_case("structs/nested")]
#[test_case("structs/enum_member")]
#[test_case("uint/consts")]
#[test_case("uint/compare")]
#[test_case("uint/upcasts")]
#[test_case("uint/safe_divmod")]
#[test_case("uint/wide_mul")]
#[test_case("uint/uint_addition")]
#[test_case("uint/uint_substraction")]
// #[test_case("felt_ops/div")] - div blocked on bug on div
fn comparison_test(test_name: &str) -> Result<(), String> {
    let sierra_code =
        fs::read_to_string(&format!("./tests/comparison/{test_name}.sierra")).unwrap();
    compile_to_mlir_with_consistency_check(test_name, &sierra_code);
    let llvm_result = run_mlir(test_name)?;

    let casm_result = run_sierra_via_casm(&sierra_code);

    match casm_result {
        Ok(result) => match result.value {
            cairo_lang_runner::RunResultValue::Success(casm_values) => {
                println!("Casm result: {:?}\n", casm_values);
                println!("llvm result: {:?}\n", llvm_result);
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
            cairo_lang_runner::RunResultValue::Panic(_) => {
                todo!("Comparison tests where the cairo runner panics");
            }
        },
        Err(_) => {
            todo!("Comparison tests where the cairo runner fails");
        }
    }
    Ok(())
}

fn compile_to_mlir_with_consistency_check(test_name: &str, sierra_code: &str) {
    let program = ProgramParser::new().parse(sierra_code).unwrap();
    let out_dir = get_outdir();
    let test_file_name = flatten_test_name(test_name);
    let compiled_code = compile(&program, false, false, true, 1).unwrap();
    let optimised_compiled_code = compile(&program, false, false, true, 1).unwrap();
    let mlir_file = out_dir.join(format!("{test_file_name}.mlir")).display().to_string();
    let optimised_mlir_file =
        out_dir.join(format!("{test_file_name}-opt.mlir")).display().to_string();
    std::fs::write(mlir_file.as_str(), &compiled_code).unwrap();
    std::fs::write(optimised_mlir_file.as_str(), &optimised_compiled_code).unwrap();
    for _ in 0..5 {
        let repeat_compiled_code = compile(&program, false, false, true, 1).unwrap();
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
fn run_sierra_via_casm(sierra_code: &str) -> Result<RunResult> {
    let sierra_program = ProgramParser::new().parse(sierra_code).unwrap();

    let runner =
        SierraCasmRunner::new(sierra_program, None).with_context(|| "Failed setting up runner.")?;

    runner.run_function("::main", &[], None).with_context(|| "Failed to run the function.")
}

// Runs the test file via reading the mlir file, compiling it to llir, then invoking lli to run it
fn run_mlir(test_name: &str) -> Result<Vec<BigUint>, String> {
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

fn run_mlir_file_via_llvm(mlir_file: &str, output_file: &str) -> Result<Vec<BigUint>, String> {
    let mlir_prefix = std::env::var("MLIR_SYS_160_PREFIX").unwrap();
    let lli_path = Path::new(mlir_prefix.as_str()).join("bin").join("lli");
    let mlir_translate_path = Path::new(mlir_prefix.as_str()).join("bin").join("mlir-translate");

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

    let lli_cmd = Command::new(lli_path)
        .arg(output_file)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let lli_output = lli_cmd.wait_with_output().unwrap();

    if !lli_output.stderr.is_empty() {
        return Err(format!(
            "lli failed with output: {}",
            String::from_utf8(lli_output.stderr).unwrap()
        ));
    }

    let output = std::str::from_utf8(&lli_output.stdout).unwrap().trim();

    Ok(parse_llvm_result(output))
}

// Parses the human-readable output from running the llir code into a raw list of outputs
fn parse_llvm_result(res: &str) -> Vec<BigUint> {
    println!("Parsing llvm result: '{}', length: {}", res, res.chars().count());
    return res
        .split('\n')
        .filter(|s| !s.is_empty())
        .map(|x| BigUint::from_str_radix(x, 16).unwrap())
        .collect();
}

fn flatten_test_name(test_name: &str) -> String {
    test_name.replace('_', "__").replace('/', "_")
}

fn get_outdir() -> PathBuf {
    Path::new(".").join("tests").join("comparison").join("out")
}
