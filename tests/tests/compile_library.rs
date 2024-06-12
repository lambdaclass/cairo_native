//use crate::common::load_cairo;
use crate::common::load_cairo;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreLibfunc, CoreType},
    extensions::core::{CoreLibfunc, CoreType},
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use cairo_native::metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage};
use cairo_native::metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage};
//use melior::{
use melior::{
//    dialect::DialectRegistry,
    dialect::DialectRegistry,
//    ir::{Location, Module},
    ir::{Location, Module},
//    pass::{self, PassManager},
    pass::{self, PassManager},
//    utility::{register_all_dialects, register_all_llvm_translations},
    utility::{register_all_dialects, register_all_llvm_translations},
//    Context,
    Context,
//};
};
//use std::error::Error;
use std::error::Error;
//use tempfile::NamedTempFile;
use tempfile::NamedTempFile;
//

//#[test]
#[test]
//pub fn compile_library() -> Result<(), Box<dyn Error>> {
pub fn compile_library() -> Result<(), Box<dyn Error>> {
//    // Load the program.
    // Load the program.
//    let context = Context::new();
    let context = Context::new();
//

//    let program = load_cairo! {
    let program = load_cairo! {
//        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//            lhs + rhs
            lhs + rhs
//        }
        }
//    };
    };
//

//    // Initialize MLIR.
    // Initialize MLIR.
//    context.append_dialect_registry(&{
    context.append_dialect_registry(&{
//        let registry = DialectRegistry::new();
        let registry = DialectRegistry::new();
//        register_all_dialects(&registry);
        register_all_dialects(&registry);
//        registry
        registry
//    });
    });
//    context.load_all_available_dialects();
    context.load_all_available_dialects();
//    register_all_llvm_translations(&context);
    register_all_llvm_translations(&context);
//

//    // Compile the program.
    // Compile the program.
//    let mut module = Module::new(Location::unknown(&context));
    let mut module = Module::new(Location::unknown(&context));
//    let mut metadata = MetadataStorage::new();
    let mut metadata = MetadataStorage::new();
//    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program.1)?;
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program.1)?;
//

//    // Make the runtime library available.
    // Make the runtime library available.
//    metadata.insert(RuntimeBindingsMeta::default()).unwrap();
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();
//

//    cairo_native::compile(
    cairo_native::compile(
//        &context,
        &context,
//        &module,
        &module,
//        &program.1,
        &program.1,
//        &registry,
        &registry,
//        &mut metadata,
        &mut metadata,
//        None,
        None,
//    )?;
    )?;
//

//    // lower to llvm dialect
    // lower to llvm dialect
//    let pass_manager = PassManager::new(&context);
    let pass_manager = PassManager::new(&context);
//    pass_manager.enable_verifier(true);
    pass_manager.enable_verifier(true);
//    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
//    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
//    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
//    pass_manager.run(&mut module)?;
    pass_manager.run(&mut module)?;
//

//    let object = cairo_native::module_to_object(&module, Default::default())?;
    let object = cairo_native::module_to_object(&module, Default::default())?;
//

//    let file = NamedTempFile::new()?.into_temp_path();
    let file = NamedTempFile::new()?.into_temp_path();
//    cairo_native::object_to_shared_lib(&object, &file)?;
    cairo_native::object_to_shared_lib(&object, &file)?;
//

//    Ok(())
    Ok(())
//}
}
