use std::fs;
use std::path::Path;
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

#[test_case("simple_return")]
#[test_case("tuple_return")]
#[test_case("enum_return")]
#[test_case("fib_counter")]
#[test_case("felt_ops/add")]
#[test_case("felt_ops/sub")]
#[test_case("felt_ops/mul")]
#[test_case("felt_ops/negation")]
// #[test_case("felt_ops/div")] - div blocked on panic and array
fn comparison_test(test_name: &str) -> Result<(), String> {
    let sierra_code =
        fs::read_to_string(&format!("./tests/comparison/{test_name}.sierra")).unwrap();
    let llvm_result = run_sierra_via_llvm(test_name, &sierra_code)?;

    let casm_result = run_sierra_via_casm(&sierra_code);

    match casm_result {
        Ok(result) => match result.value {
            cairo_lang_runner::RunResultValue::Success(casm_values) => {
                println!("Casm result: {:?}\n", casm_values);
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

// Invokes starkware's runner that compiles sierra to casm and runs it
// This provides us with the intended results to compare against
fn run_sierra_via_casm(sierra_code: &str) -> Result<RunResult> {
    let sierra_program = ProgramParser::new().parse(sierra_code).unwrap();

    let runner = SierraCasmRunner::new(sierra_program, false)
        .with_context(|| "Failed setting up runner.")?;

    runner.run_function("::main", &[], None).with_context(|| "Failed to run the function.")
}

// Runs the test file via compiling to mlir, then llir, then invoking lli to run it
fn run_sierra_via_llvm(test_name: &str, sierra_code: &str) -> Result<Vec<BigUint>, String> {
    let program = ProgramParser::new().parse(sierra_code).unwrap();

    let tmp_dir = tempdir::TempDir::new("test_comparison").unwrap().into_path();

    // Allows folders of comparison tests without write producing a file not found
    let test_file_name = flatten_test_name(test_name);

    let mlir_file = tmp_dir.join(format!("{test_file_name}.mlir")).display().to_string();
    let output_file = tmp_dir.join(format!("{test_file_name}.ll")).display().to_string();

    let compiled_code = compile(&program, false, false, true, 1).unwrap();
    std::fs::write(mlir_file.as_str(), compiled_code).unwrap();

    let mlir_prefix = std::env::var("MLIR_SYS_160_PREFIX").unwrap();
    let mlir_translate_path = Path::new(mlir_prefix.as_str()).join("bin").join("mlir-translate");
    let lli_path = Path::new(mlir_prefix.as_str()).join("bin").join("lli");

    let mlir_output = Command::new(mlir_translate_path)
        .arg("--mlir-to-llvmir")
        .arg("-o")
        .arg(output_file.as_str())
        .arg(mlir_file.as_str())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();

    if !mlir_output.stdout.is_empty() || !mlir_output.stderr.is_empty() {
        println!(
            "Mlir_output:\n    stdout: {}\n    stderr: {}",
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
