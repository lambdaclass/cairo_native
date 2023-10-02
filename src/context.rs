use crate::{
    error::compile::CompileError,
    metadata::{
        gas::{GasMetadata, MetadataComputationConfig},
        runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    module::NativeModule,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context,
};

/// Context of IRs, dialects and passes for Cairo programs compilation.
pub struct NativeContext {
    context: Context,
}

impl Default for NativeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeContext {
    pub fn new() -> Self {
        let context = initialize_mlir();
        Self { context }
    }

    /// Compiles a sierra program into MLIR and then lowers to LLVM.
    /// Returns the corresponding NativeModule struct.
    pub fn compile(&self, program: &Program) -> Result<NativeModule, CompileError> {
        let mut module = Module::new(Location::unknown(&self.context));

        let has_gas_builtin = program
            .type_declarations
            .iter()
            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin");

        let mut metadata = MetadataStorage::new();
        // Make the runtime library available.
        metadata.insert(RuntimeBindingsMeta::default());
        // We assume that GasMetadata will be always present when the program uses the gas builtin.
        if has_gas_builtin {
            let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());
            // Unwrapping here is not necessary since the insertion will only fail if there was
            // already some metadata of the same type.
            metadata.insert(gas_metadata);
        }

        // Create the Sierra program registry
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

        crate::compile(
            &self.context,
            &module,
            program,
            &registry,
            &mut metadata,
            None,
        )?;

        self.lower_to_llvm(&mut module)?;

        Ok(NativeModule::new(module, registry, metadata))
    }

    fn lower_to_llvm(&self, module: &mut Module) -> Result<(), CompileError> {
        let pass_manager = PassManager::new(&self.context);
        pass_manager.enable_verifier(true);
        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
        Ok(pass_manager.run(module)?)
    }
}

/// Initialize an MLIR context.
pub fn initialize_mlir() -> Context {
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_passes();
    context
}
