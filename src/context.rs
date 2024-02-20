use crate::{
    error::compile::CompileError,
    metadata::{
        gas::{GasMetadata, MetadataComputationConfig},
        runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    module::NativeModule,
    utils::run_pass_manager,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    Context,
};

/// Context of IRs, dialects and passes for Cairo programs compilation.
#[derive(Debug, Eq, PartialEq)]
pub struct NativeContext {
    context: Context,
}

unsafe impl Send for NativeContext {}
unsafe impl Sync for NativeContext {}

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

        run_pass_manager(&self.context, &mut module)?;

        Ok(NativeModule::new(module, registry, metadata))
    }

    /// Compiles a sierra program into MLIR and then lowers to LLVM. Using the given metadata.
    /// Returns the corresponding NativeModule struct.
    pub fn compile_with_metadata(&self, program: &Program, metadata_config: MetadataComputationConfig) -> Result<NativeModule, CompileError> {
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
            let gas_metadata = GasMetadata::new(program, metadata_config);
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

        run_pass_manager(&self.context, &mut module)?;

        Ok(NativeModule::new(module, registry, metadata))
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
    register_all_llvm_translations(&context);
    context
}
