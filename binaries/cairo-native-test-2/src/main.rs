use anyhow::bail;
use cairo_lang_test_plugin::TestCompilation;
use cairo_native_bin_utils::{
    test::{display_tests_summary, filter_test_cases, run_tests},
    RunArgs,
};
use clap::Parser;
use std::{fs::File, path::PathBuf};

/// Executes all the Sierra tests specified by the given test metadata.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// Path to the input Sierra tests
    tests_path: PathBuf,
    /// The filter for the tests, running only tests containing the filter string.
    #[arg(short, long, default_value_t = String::default())]
    filter: String,
    /// Whether to run ignored tests as well.
    #[arg(long, default_value_t = false)]
    include_ignored: bool,
    /// Whether to run only the ignored tests.
    #[arg(long, default_value_t = false)]
    ignored: bool,
    /// Optimization level (Valid: 0, 1, 2, 3).
    /// Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
    /// Compares test result with Cairo VM.
    #[arg(long, default_value_t = false)]
    compare_with_cairo_vm: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tests_file = File::open(args.tests_path)?;
    let test_compilation: TestCompilation = serde_json::from_reader(&tests_file)?;

    let (test_compilation, filtered_out) = filter_test_cases(
        test_compilation,
        args.include_ignored,
        args.ignored,
        args.filter.clone(),
    );

    let summary = run_tests(
        test_compilation.metadata.named_tests,
        test_compilation.sierra_program.program,
        test_compilation.metadata.function_set_costs,
        test_compilation.metadata.contracts_info,
        RunArgs {
            opt_level: args.opt_level,
            compare_with_vm: args.compare_with_cairo_vm,
        },
    )?;

    display_tests_summary(&summary, filtered_out);

    if !summary.failed.is_empty() || !summary.mismatch.is_empty() {
        bail!("test failed")
    }

    Ok(())
}
