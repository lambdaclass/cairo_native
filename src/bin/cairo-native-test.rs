use anyhow::bail;
use cairo_lang_compiler::{
    db::RootDatabase,
    diagnostics::DiagnosticsReporter,
    project::{check_compiler_path, setup_project},
};
use cairo_lang_filesystem::cfg::{Cfg, CfgSet};
use cairo_lang_starknet::starknet_plugin_suite;
use cairo_lang_test_plugin::{
    compile_test_prepared_db, test_plugin_suite, TestCompilation, TestsCompilationConfig,
};
use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use utils::{
    test::{display_tests_summary, filter_test_cases, run_tests},
    RunArgs, RunMode,
};

mod utils;

/// Compiles a Cairo project and runs all the functions marked as `#[test]`.
/// Exits with 1 if the compilation or run fails, otherwise 0.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The Cairo project path to compile and run its tests.
    path: PathBuf,
    /// Whether path is a single file.
    #[arg(short, long)]
    single_file: bool,
    /// Allows the compilation to succeed with warnings.
    #[arg(long)]
    allow_warnings: bool,
    /// The filter for the tests, running only tests containing the filter string.
    #[arg(short, long, default_value_t = String::default())]
    filter: String,
    /// Skips compilation for tests/functions containing any of the given filters.
    /// Unlike `--filter`, the matching tests are not even compiled by native.
    ///
    /// DISCLAIMER: This is a hacky and temporary flag, used to run corelib tests
    /// when not all libfuncs are implemented.
    #[arg(long)]
    skip_compilation: Vec<String>,
    /// Should we run ignored tests as well.
    #[arg(long, default_value_t = false)]
    include_ignored: bool,
    /// Should we run only the ignored tests.
    #[arg(long, default_value_t = false)]
    ignored: bool,
    /// Should we add the starknet plugin to run the tests.
    #[arg(long, default_value_t = false)]
    starknet: bool,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
}

fn main() -> anyhow::Result<()> {
    // Parse command-line arguments.
    let args = Args::parse();

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    check_compiler_path(args.single_file, &args.path)?;

    let db = &mut {
        let mut b = RootDatabase::builder();
        b.detect_corelib();
        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
        b.with_default_plugin_suite(test_plugin_suite());
        if args.starknet {
            b.with_default_plugin_suite(starknet_plugin_suite());
        }

        b.build()?
    };

    let main_crate_ids = setup_project(db, &args.path)?;

    let db = db.snapshot();
    let test_crate_ids = main_crate_ids.clone();
    let test_config = TestsCompilationConfig {
        starknet: args.starknet,
        add_statements_functions: false,
        add_statements_code_locations: false,
        contract_declarations: None,
        contract_crate_ids: None,
        executable_crate_ids: None,
    };

    let mut diag_reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);
    if args.allow_warnings {
        diag_reporter = diag_reporter.allow_warnings();
    }

    let build_test_compilation =
        compile_test_prepared_db(&db, test_config, test_crate_ids.clone(), diag_reporter)?;

    let (compiled, filtered_out) = filter_test_cases(
        build_test_compilation,
        args.include_ignored,
        args.ignored,
        args.filter.clone(),
    );

    let compiled = filter_test_case_compilation(compiled, &args.skip_compilation);

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
    if !summary.failed.is_empty() {
        bail!(
            "test result: {}. {} passed; {} failed; {} ignored",
            "FAILED".bright_red(),
            summary.passed.len(),
            summary.failed.len(),
            summary.ignored.len()
        );
    }

    Ok(())
}

/// Removes matching tests from `TestCompilation`. This not only includes the
/// test cases, but also the function definition in the inner sierra program.
/// This means that the matching tests are not even compiled.
///
/// DISCLAIMER: This is a hacky and temporary function, used to run corelib tests
/// when not all libfuncs are implemented.
fn filter_test_case_compilation(
    mut compiled: TestCompilation,
    compilation_filter: &[String],
) -> TestCompilation {
    let should_skip_test = |name: &str| -> bool {
        compilation_filter
            .iter()
            .any(|filter| name.contains(filter))
    };

    // Remove matching function definitions.
    compiled.sierra_program.program.funcs.retain(|f| {
        let name =
            f.id.debug_name
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or_default();

        let skipped = should_skip_test(name);

        if skipped {
            println!("skipping compilation of: {}", name);
        }

        !skipped
    });

    // Ignore matching test cases.
    compiled
        .metadata
        .named_tests
        .iter_mut()
        .for_each(|(test, case)| {
            if should_skip_test(test) {
                case.ignored = true
            }
        });

    compiled
}
