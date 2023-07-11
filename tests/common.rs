#![allow(unused_macros)]
#![allow(dead_code)]

use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_filesystem::db::init_dev_corelib;
use cairo_lang_runner::{
    Arg, RunResult, RunResultValue, RunnerError, SierraCasmRunner, StarknetState,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_starknet::contract::get_contracts_info;
use cairo_native::{
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::felt252::PRIME,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use num_bigint::{BigInt, Sign};
use std::{env::var, fs, ops::Neg, path::Path, sync::Arc};

macro_rules! load_cairo {
    ( $( $program:tt )+ ) => {
        $crate::common::load_cairo_str(stringify!($($program)+))
    };
}

#[allow(unused_imports)]
pub(crate) use load_cairo;

pub(crate) const GAS: usize = usize::MAX;

// Parse numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt(value: &str) -> [u32; 8] {
    let value = value.parse::<BigInt>().unwrap();
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Converts a casm variant to sierra.
pub const fn casm_variant_to_sierra(idx: i64, num_variants: i64) -> i64 {
    num_variants - 1 - (idx >> 1)
}

pub fn get_result_success(r: RunResultValue) -> Vec<String> {
    match r {
        RunResultValue::Success(x) => x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        RunResultValue::Panic(_) => panic!(),
    }
}

pub fn load_cairo_str(program_str: &str) -> (String, Program, SierraCasmRunner) {
    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    fs::write(&mut program_file, program_str).unwrap();

    let mut db = RootDatabase::default();
    init_dev_corelib(
        &mut db,
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );
    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
    let program = Arc::try_unwrap(
        compile_prepared_db(
            &mut db,
            main_crate_ids.clone(),
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .unwrap();

    let module_name = program_file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();

    let replacer = DebugReplacer { db: &db };
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();

    let runner =
        SierraCasmRunner::new(program.clone(), Some(Default::default()), contracts_info).unwrap();

    (module_name.to_string(), program, runner)
}

pub fn run_native_program(
    program: &(&str, &Program),
    entry_point: &str,
    args: serde_json::Value,
) -> serde_json::Value {
    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
    let program = &program.1;

    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
        .expect("Could not create the test program registry.");

    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_passes();

    let mut module = Module::new(Location::unknown(&context));

    let mut metadata = MetadataStorage::new();
    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        program,
        &registry,
        &mut metadata,
        None,
    )
    .expect("Could not compile test program to MLIR.");

    assert!(
        module.as_operation().verify(),
        "Test program generated invalid MLIR:\n{}",
        module.as_operation()
    );

    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());

    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
    pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    pass_manager
        .run(&mut module)
        .expect("Could not apply passes to the compiled test program.");

    let engine = ExecutionEngine::new(&module, 0, &[], false);

    #[cfg(feature = "with-runtime")]
    unsafe {
        engine.register_symbol(
            "cairo_native__libfunc__debug__print",
            cairo_native_runtime::cairo_native__libfunc__debug__print
                as *const fn(i32, *const [u8; 32], usize) -> i32 as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc_pedersen",
            cairo_native_runtime::cairo_native__libfunc_pedersen
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
        );
    }

    cairo_native::execute::<CoreType, CoreLibfunc, _, _>(
        &engine,
        &registry,
        &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .expect("Test program entry point not found.")
            .id,
        args,
        serde_json::value::Serializer,
    )
    .expect("Test program execution failed.")
}

pub fn run_vm_program(
    program: &(&str, &Program, &SierraCasmRunner),
    entry_point: &str,
    args: &[Arg],
    gas: Option<usize>,
) -> Result<RunResult, RunnerError> {
    let runner = program.2;
    runner.run_function(
        runner.find_function(entry_point).unwrap(),
        args,
        gas,
        StarknetState::default(),
    )
}
