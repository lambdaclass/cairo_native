use crate::common::load_cairo;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::{ffi::{get_data_layout_rep, get_target_triple}, metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage}};
use melior::{
    dialect::DialectRegistry,
    ir::{attribute::StringAttribute, operation::OperationBuilder, Identifier, Location, Module, Region},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};
use std::error::Error;
use tempfile::NamedTempFile;

#[test]
pub fn compile_library() -> Result<(), Box<dyn Error>> {
    // Load the program.
    let context = Context::new();

    let program = load_cairo! {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            lhs + rhs
        }
    };

    // Initialize MLIR.
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);

    // Compile the program.
    let target_triple = get_target_triple();
    let data_layout = get_data_layout_rep().unwrap();
    let mut module = Module::from_operation(
        OperationBuilder::new("builtin.module", Location::unknown(&context))
            .add_attributes(&[
                (
                    Identifier::new(&context, "llvm.target_triple"),
                    StringAttribute::new(&context, &target_triple).into(),
                ),
                (
                    Identifier::new(&context, "llvm.data_layout"),
                    StringAttribute::new(&context, &data_layout).into(),
                ),
            ])
            .add_regions([Region::new()])
            .build()
            .unwrap(),
    )
    .unwrap();
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program.1)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    cairo_native::compile(
        &context,
        &module,
        &program.1,
        &registry,
        &mut metadata,
        None,
    )?;

    // lower to llvm dialect
    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.run(&mut module)?;

    let object = cairo_native::module_to_object(&module, Default::default())?;

    let file = NamedTempFile::new()?.into_temp_path();
    cairo_native::object_to_shared_lib(&object, &file)?;

    Ok(())
}
