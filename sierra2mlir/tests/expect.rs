use std::path::Path;
use std::process::{Command, Stdio};

use cairo_lang_sierra::ProgramParser;
use color_eyre::Result;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num};
use sierra2mlir::compile;

// Tests behaviour of the generated MLIR against the expected values.

#[test]
fn array_append() -> Result<(), String> {
    let sierra_code = include_str!("comparison/array/example_array.sierra");
    let llvm_result = run_sierra_via_llvm("example_array", sierra_code)?;

    assert_eq!(
        llvm_result,
        vec![
            BigUint::from_i64(1).unwrap(),
            BigUint::from_i64(2).unwrap(),
            BigUint::from_i64(3).unwrap(),
            BigUint::from_i64(4).unwrap(),
            BigUint::from_i64(5).unwrap(),
            BigUint::from_i64(5).unwrap(),
        ]
    );

    Ok(())
}

// Runs the test file via compiling to mlir, then llir, then invoking lli to run it
fn run_sierra_via_llvm(test_name: &str, sierra_code: &str) -> Result<Vec<BigUint>, String> {
    let program = ProgramParser::new().parse(sierra_code).unwrap();

    let out_dir = Path::new(".").join("tests").join("comparison").join("out");

    // Allows folders of comparison tests without write producing a file not found
    let test_file_name = flatten_test_name(test_name);

    let mlir_file = out_dir.join(format!("{test_file_name}.mlir")).display().to_string();
    let output_file = out_dir.join(format!("{test_file_name}.ll")).display().to_string();

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
