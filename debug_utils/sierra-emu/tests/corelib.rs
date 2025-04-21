use std::path::Path;

use cairo_lang_compiler::{db::RootDatabase, diagnostics::DiagnosticsReporter, project::{check_compiler_path, setup_project}};
use cairo_lang_filesystem::cfg::{Cfg, CfgSet};
use cairo_lang_starknet::starknet_plugin_suite;
use cairo_lang_test_plugin::{compile_test_prepared_db, test_plugin_suite, TestsCompilationConfig};

#[test]
fn test_corelib() {
    let compiler_path = Path::new("corelib");

    check_compiler_path(false, &compiler_path).unwrap();

    let db = &mut {
        let mut b = RootDatabase::builder();
        b.detect_corelib();
        b.with_cfg(CfgSet::from_iter([Cfg::name("test")]));
        b.with_default_plugin_suite(test_plugin_suite());
        b.with_default_plugin_suite(starknet_plugin_suite());
        b.build().unwrap()
    };

    let main_crate_ids = setup_project(db, &compiler_path).unwrap();

    let db = db.snapshot();
    let test_crate_ids = main_crate_ids.clone();
    let test_config = TestsCompilationConfig {
        starknet: true,
        add_statements_functions: false,
        add_statements_code_locations: false,
        contract_declarations: None,
        contract_crate_ids: None,
        executable_crate_ids: None,
    };

    let mut diag_reporter = DiagnosticsReporter::stderr().with_crates(&main_crate_ids);

    let build_test_compilation =
        compile_test_prepared_db(&db, test_config, test_crate_ids.clone(), diag_reporter).unwrap();

    let (compiled, filtered_out) = filter_test_cases(
        build_test_compilation,
        args.include_ignored,
        args.ignored,
        args.filter.clone(),
    );

    let compiled = filter_test_case_compilation(compiled, &args.skip_compilation);

    let summary =

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
}
