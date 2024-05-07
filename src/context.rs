use crate::{
    debug_info::DebugLocations,
    error::Error,
    ffi::{get_data_layout_rep, get_target_triple},
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
use llvm_sys::target::{
    LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs,
    LLVM_InitializeAllTargets,
};
use melior::{
    dialect::DialectRegistry,
    ir::{
        attribute::StringAttribute,
        operation::{OperationBuilder, OperationPrintingFlags},
        Block, Identifier, Location, Module, Region,
    },
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    Context,
};
use std::sync::OnceLock;

/// Context of IRs, dialects and passes for Cairo programs compilation.
#[derive(Debug, Eq, PartialEq)]
pub struct NativeContext {
    context: Context,

    target_triple: String,
    data_layout: String,
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

        Self {
            context,
            target_triple: get_target_triple(),
            data_layout: get_data_layout_rep().unwrap(),
        }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn new_module(&self) -> Module {
        let module_op = OperationBuilder::new("builtin.module", Location::unknown(&self.context))
            .add_attributes(&[
                (
                    Identifier::new(&self.context, "llvm.target_triple"),
                    StringAttribute::new(&self.context, &self.target_triple).into(),
                ),
                (
                    Identifier::new(&self.context, "llvm.data_layout"),
                    StringAttribute::new(&self.context, &self.data_layout).into(),
                ),
            ])
            .add_regions([{
                let region = Region::new();
                region.append_block(Block::new(&[]));
                region
            }])
            .build()
            .unwrap();
        assert!(module_op.verify(), "module operation is not valid");

        Module::from_operation(module_op).unwrap()
    }

    /// Compiles a sierra program into MLIR and then lowers to LLVM.
    /// Returns the corresponding NativeModule struct.
    pub fn compile(
        &self,
        program: &Program,
        debug_locations: Option<DebugLocations>,
    ) -> Result<NativeModule, Error> {
        let mut module = self.new_module();

        let has_gas_builtin = program
            .type_declarations
            .iter()
            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin");

        let mut metadata = MetadataStorage::new();
        // Make the runtime library available.
        metadata.insert(RuntimeBindingsMeta::default());
        // We assume that GasMetadata will be always present when the program uses the gas builtin.
        let gas_metadata = if has_gas_builtin {
            GasMetadata::new(program, Some(MetadataComputationConfig::default()))
        } else {
            GasMetadata::new(program, None)
        }?;
        // Unwrapping here is not necessary since the insertion will only fail if there was
        // already some metadata of the same type.
        metadata.insert(gas_metadata);

        // Create the Sierra program registry
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

        crate::compile(
            &self.context,
            &module,
            program,
            &registry,
            &mut metadata,
            debug_locations.as_ref(),
        )?;

        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP_PREPASS") {
            if x == "1" || x == "true" {
                std::fs::write("dump-prepass.mlir", module.as_operation().to_string())
                    .expect("should work");
                std::fs::write(
                    "dump-prepass-debug.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, true),
                    )?,
                )
                .expect("should work");
            }
        }

        run_pass_manager(&self.context, &mut module)?;

        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
            if x == "1" || x == "true" {
                std::fs::write("dump.mlir", module.as_operation().to_string())
                    .expect("should work");
                std::fs::write(
                    "dump-debug.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, true),
                    )?,
                )
                .expect("should work");
            }
        }

        // The func to llvm pass has a bug where it sets the data layout string to ""
        // This works around it by setting it again.
        {
            let mut op = module.as_operation_mut();
            op.set_attribute(
                "llvm.data_layout",
                StringAttribute::new(&self.context, &self.data_layout).into(),
            );
        }

        Ok(NativeModule::new(module, registry, metadata))
    }

    /// Compiles a sierra program into MLIR and then lowers to LLVM. Using the given metadata.
    /// Returns the corresponding NativeModule struct.
    pub fn compile_with_metadata(
        &self,
        program: &Program,
        metadata_config: MetadataComputationConfig,
    ) -> Result<NativeModule, Error> {
        let mut module = self.new_module();

        let mut metadata = MetadataStorage::new();
        // Make the runtime library available.
        metadata.insert(RuntimeBindingsMeta::default());

        let gas_metadata = GasMetadata::new(program, Some(metadata_config))?;
        metadata.insert(gas_metadata);

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
    register_all_llvm_translations(&context);

    static INITIALIZED: OnceLock<()> = OnceLock::new();
    INITIALIZED.get_or_init(|| unsafe {
        register_all_passes();

        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmPrinters();

        tracing::debug!("Initialized LLVM targets.");
    });

    context
}
