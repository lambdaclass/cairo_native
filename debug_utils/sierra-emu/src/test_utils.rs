#![cfg(test)]

use std::{fs, path::Path, sync::Arc};

use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    project::setup_project, CompilerConfig,
};
use cairo_lang_filesystem::db::init_dev_corelib;
use cairo_lang_sierra::program::Program;

use crate::{starknet::StubSyscallHandler, Value, VirtualMachine};

#[macro_export]
macro_rules! load_cairo {
    ( $( $program:tt )+ ) => {
        $crate::test_utils::load_cairo_from_str(stringify!($($program)+))
    };
}

pub(crate) fn load_cairo_from_str(cairo_str: &str) -> (String, Program) {
    let mut file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    let mut db = RootDatabase::default();

    fs::write(&mut file, cairo_str).unwrap();

    init_dev_corelib(
        &mut db,
        Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );

    let main_crate_ids = setup_project(&mut db, file.path()).unwrap();

    let sierra_with_dbg = compile_prepared_db(
        &db,
        main_crate_ids,
        CompilerConfig {
            diagnostics_reporter: DiagnosticsReporter::stderr(),
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let module_name = file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();
    (module_name.to_string(), sierra_with_dbg.program)
}

pub fn run_test_program(sierra_program: Program) -> Vec<Value> {
    let function = sierra_program
        .funcs
        .iter()
        .find(|f| {
            f.id.debug_name
                .as_ref()
                .map(|name| name.as_str().contains("main"))
                .unwrap_or_default()
        })
        .unwrap();

    let mut vm = VirtualMachine::new(Arc::new(sierra_program.clone()));

    let initial_gas = 1000000;

    let args: &[Value] = &[];
    vm.call_program(function, initial_gas, args.iter().cloned());

    let syscall_handler = &mut StubSyscallHandler::default();
    let trace = vm.run_with_trace(syscall_handler);

    trace
        .states
        .last()
        .unwrap()
        .items
        .values()
        .cloned()
        .collect()
}
