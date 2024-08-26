mod utils;

use std::{env, fs};

use anyhow::Context;
use cairo_lang_test_plugin::TestCompilation;
use clap::Parser;
use scarb_metadata::{Metadata, MetadataCommand, ScarbCommand};
use scarb_ui::args::PackagesFilter;
use utils::test::{display_tests_summary, filter_test_cases, find_testable_targets, run_tests};
use utils::{RunArgs, RunMode};

/// Compiles all packages from a Scarb project matching `packages_filter` and
/// runs all functions marked with `#[test]`. Exits with 1 if the compilation
/// or run fails, otherwise 0.
#[derive(Parser, Clone, Debug)]
#[command(author, version, verbatim_doc_comment)]
struct Args {
    #[command(flatten)]
    packages_filter: PackagesFilter,
    /// Run only tests whose name contain FILTER.
    #[arg(short, long, default_value = "")]
    filter: String,
    /// Run ignored and not ignored tests.
    #[arg(long, default_value_t = false)]
    include_ignored: bool,
    /// Run only ignored tests.
    #[arg(long, default_value_t = false)]
    ignored: bool,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
}

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();

    let metadata = MetadataCommand::new().inherit_stderr().exec()?;

    // Filter packages.
    let matched = args.packages_filter.match_many(&metadata)?;
    let filter = PackagesFilter::generate_for::<Metadata>(matched.iter());

    // Build only the filtered packages.
    ScarbCommand::new()
        .arg("build")
        .arg("--test")
        .env("SCARB_PACKAGES_FILTER", filter.to_env())
        .run()?;

    // Get `target` directory.
    let profile = env::var("SCARB_PROFILE").unwrap_or("dev".into());
    let default_target_dir = metadata.runtime_manifest.join("target");
    let target_dir = metadata
        .target_dir
        .clone()
        .unwrap_or(default_target_dir)
        .join(profile);

    // Iterate over the filtered packages.
    for package in matched {
        println!("testing {} ...", package.name);

        // Iterate over the filtered targets.
        for target in find_testable_targets(&package) {
            let file_path = target_dir.join(format!("{}.test.json", target.name.clone()));
            let compiled = serde_json::from_str::<TestCompilation>(
                &fs::read_to_string(file_path.clone())
                    .with_context(|| format!("failed to read file: {file_path}"))?,
            )
            .with_context(|| format!("failed to deserialize compiled tests file: {file_path}"))?;

            let (compiled, filtered_out) = filter_test_cases(
                compiled,
                args.include_ignored,
                args.ignored,
                args.filter.clone(),
            );

            let summary = run_tests(
                compiled.metadata.named_tests,
                compiled.sierra_program.program,
                compiled.metadata.function_set_costs,
                RunArgs {
                    run_mode: args.run_mode.clone(),
                    opt_level: args.opt_level,
                },
            )?;

            display_tests_summary(&summary, filtered_out);
        }
    }

    Ok(())
}
