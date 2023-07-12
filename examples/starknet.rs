#![feature(strict_provenance)]

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::metadata::{
    runtime_bindings::RuntimeBindingsMeta, syscall_handler::SyscallHandlerMeta, MetadataStorage,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use serde_json::json;
use std::{io, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

    let program = cairo_lang_compiler::compile_cairo_project_at_path(
        Path::new("programs/examples/starknet.cairo"),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let entry_point = match program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some("main"))
    {
        Some(x) => x,
        None => {
            // TODO: Use a proper error.
            eprintln!("Entry point `starknet::starknet::main` not found in program.");
            return Ok(());
        }
    };

    // Initialize MLIR.
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    // Compile the program.
    let mut module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Make the Starknet syscall handler available.
    // metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
        None,
    )?;

    // Lower to LLVM.
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

    pass_manager.run(&mut module)?;

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

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

    let params_input = json!([
        u64::MAX,
        metadata
            .get::<SyscallHandlerMeta>()
            .unwrap()
            .as_ptr()
            .addr()
    ]);

    cairo_native::execute(
        &engine,
        &registry,
        &entry_point.id,
        params_input,
        &mut serde_json::Serializer::pretty(io::stdout()),
    )
    .unwrap();

    Ok(())
}
