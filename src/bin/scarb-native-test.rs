use anyhow::Context;
use cairo_lang_sierra::program::VersionedProgram;
use cairo_lang_test_plugin::{TestCompilation, TestCompilationMetadata};
use clap::{Parser, ValueEnum};
use scarb_metadata::{Metadata, MetadataCommand, ScarbCommand};
use scarb_ui::args::PackagesFilter;
use std::{collections::HashSet, env, fs, path::Path};
use utils::{
    test::{display_tests_summary, filter_test_cases, find_testable_targets, run_tests},
    RunArgs, RunMode,
};

mod utils;

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
    /// Choose test kind to run.
    #[arg(short, long)]
    test_kind: Option<TestKind>,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum TestKind {
    Unit,
    Integration,
    #[default]
    All,
}

impl TestKind {
    pub fn matches(&self, kind: &str) -> bool {
        match self {
            TestKind::Unit => kind == "unit",
            TestKind::Integration => kind == "integration",
            TestKind::All => true,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();

    let metadata = MetadataCommand::new().inherit_stderr().exec()?;

    // Filter packages.
    let matched = args.packages_filter.match_many(&metadata)?;
    let filter = PackagesFilter::generate_for::<Metadata>(matched.iter());
    let test_kind = args.test_kind.unwrap_or_default();
    let target_names = matched
        .iter()
        .flat_map(|package| {
            find_testable_targets(package)
                .iter()
                .filter(|target| {
                    test_kind.matches(
                        target
                            .params
                            .get("test-type")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default(),
                    )
                })
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Build only the filtered packages.
    ScarbCommand::new()
        .arg("build")
        .arg("--test")
        .env("SCARB_TARGET_NAMES", target_names.clone().join(","))
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

    let mut deduplicator = TargetGroupDeduplicator::default();
    for package in matched {
        println!("testing {} ...", package.name);

        // Iterate over the filtered targets.
        for target in find_testable_targets(&package) {
            if !target_names.contains(&target.name) {
                continue;
            }
            let name = target
                .params
                .get("group-id")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or(target.name.clone());
            let already_seen = deduplicator.visit(package.name.clone(), name.clone());
            if already_seen {
                continue;
            }
            let compiled = deserialize_test_compilation(target_dir.as_std_path(), name.clone())?;

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

#[derive(Default)]
struct TargetGroupDeduplicator {
    seen: HashSet<(String, String)>,
}

impl TargetGroupDeduplicator {
    /// Returns true if already visited.
    pub fn visit(&mut self, package_name: String, group_name: String) -> bool {
        !self.seen.insert((package_name, group_name))
    }
}

fn deserialize_test_compilation(
    target_dir: &Path,
    name: String,
) -> anyhow::Result<TestCompilation> {
    let file_path = target_dir.join(format!("{}.test.json", name));
    let test_comp_metadata = serde_json::from_str::<TestCompilationMetadata>(
        &fs::read_to_string(file_path.clone())
            .with_context(|| format!("failed to read file: {file_path:?}"))?,
    )
    .with_context(|| {
        format!("failed to deserialize compiled tests metadata file: {file_path:?}")
    })?;

    let file_path = target_dir.join(format!("{}.test.sierra.json", name));
    let sierra_program = serde_json::from_str::<VersionedProgram>(
        &fs::read_to_string(file_path.clone())
            .with_context(|| format!("failed to read file: {file_path:?}"))?,
    )
    .with_context(|| format!("failed to deserialize compiled tests sierra file: {file_path:?}"))?;

    Ok(TestCompilation {
        sierra_program: sierra_program.into_v1()?,
        metadata: test_comp_metadata,
    })
}
