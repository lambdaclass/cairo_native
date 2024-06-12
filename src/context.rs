//use std::sync::OnceLock;
use std::sync::OnceLock;
//

//use crate::{
use crate::{
//    debug_info::DebugLocations,
    debug_info::DebugLocations,
//    error::Error,
    error::Error,
//    ffi::{get_data_layout_rep, get_target_triple},
    ffi::{get_data_layout_rep, get_target_triple},
//    metadata::{
    metadata::{
//        gas::{GasMetadata, MetadataComputationConfig},
        gas::{GasMetadata, MetadataComputationConfig},
//        runtime_bindings::RuntimeBindingsMeta,
        runtime_bindings::RuntimeBindingsMeta,
//        MetadataStorage,
        MetadataStorage,
//    },
    },
//    module::NativeModule,
    module::NativeModule,
//    utils::run_pass_manager,
    utils::run_pass_manager,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreLibfunc, CoreType},
    extensions::core::{CoreLibfunc, CoreType},
//    program::Program,
    program::Program,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use llvm_sys::target::{
use llvm_sys::target::{
//    LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs,
    LLVM_InitializeAllAsmPrinters, LLVM_InitializeAllTargetInfos, LLVM_InitializeAllTargetMCs,
//    LLVM_InitializeAllTargets,
    LLVM_InitializeAllTargets,
//};
};
//use melior::{
use melior::{
//    dialect::DialectRegistry,
    dialect::DialectRegistry,
//    ir::{
    ir::{
//        attribute::StringAttribute,
        attribute::StringAttribute,
//        operation::{OperationBuilder, OperationPrintingFlags},
        operation::{OperationBuilder, OperationPrintingFlags},
//        Block, Identifier, Location, Module, Region,
        Block, Identifier, Location, Module, Region,
//    },
    },
//    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
//    Context,
    Context,
//};
};
//

///// Context of IRs, dialects and passes for Cairo programs compilation.
/// Context of IRs, dialects and passes for Cairo programs compilation.
//#[derive(Debug, Eq, PartialEq)]
#[derive(Debug, Eq, PartialEq)]
//pub struct NativeContext {
pub struct NativeContext {
//    context: Context,
    context: Context,
//}
}
//

//unsafe impl Send for NativeContext {}
unsafe impl Send for NativeContext {}
//unsafe impl Sync for NativeContext {}
unsafe impl Sync for NativeContext {}
//

//impl Default for NativeContext {
impl Default for NativeContext {
//    fn default() -> Self {
    fn default() -> Self {
//        Self::new()
        Self::new()
//    }
    }
//}
}
//

//impl NativeContext {
impl NativeContext {
//    pub fn new() -> Self {
    pub fn new() -> Self {
//        let context = initialize_mlir();
        let context = initialize_mlir();
//        Self { context }
        Self { context }
//    }
    }
//

//    pub fn context(&self) -> &Context {
    pub fn context(&self) -> &Context {
//        &self.context
        &self.context
//    }
    }
//

//    /// Compiles a sierra program into MLIR and then lowers to LLVM.
    /// Compiles a sierra program into MLIR and then lowers to LLVM.
//    /// Returns the corresponding NativeModule struct.
    /// Returns the corresponding NativeModule struct.
//    pub fn compile(
    pub fn compile(
//        &self,
        &self,
//        program: &Program,
        program: &Program,
//        debug_locations: Option<DebugLocations>,
        debug_locations: Option<DebugLocations>,
//    ) -> Result<NativeModule, Error> {
    ) -> Result<NativeModule, Error> {
//        static INITIALIZED: OnceLock<()> = OnceLock::new();
        static INITIALIZED: OnceLock<()> = OnceLock::new();
//        INITIALIZED.get_or_init(|| unsafe {
        INITIALIZED.get_or_init(|| unsafe {
//            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargets();
//            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargetInfos();
//            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllTargetMCs();
//            LLVM_InitializeAllAsmPrinters();
            LLVM_InitializeAllAsmPrinters();
//            tracing::debug!("initialized llvm targets");
            tracing::debug!("initialized llvm targets");
//        });
        });
//        let target_triple = get_target_triple();
        let target_triple = get_target_triple();
//

//        let module_region = Region::new();
        let module_region = Region::new();
//        module_region.append_block(Block::new(&[]));
        module_region.append_block(Block::new(&[]));
//

//        let data_layout_ret = &get_data_layout_rep()?;
        let data_layout_ret = &get_data_layout_rep()?;
//

//        let op = OperationBuilder::new(
        let op = OperationBuilder::new(
//            "builtin.module",
            "builtin.module",
//            Location::name(&self.context, "module", Location::unknown(&self.context)),
            Location::name(&self.context, "module", Location::unknown(&self.context)),
//        )
        )
//        .add_attributes(&[
        .add_attributes(&[
//            (
            (
//                Identifier::new(&self.context, "llvm.target_triple"),
                Identifier::new(&self.context, "llvm.target_triple"),
//                StringAttribute::new(&self.context, &target_triple).into(),
                StringAttribute::new(&self.context, &target_triple).into(),
//            ),
            ),
//            (
            (
//                Identifier::new(&self.context, "llvm.data_layout"),
                Identifier::new(&self.context, "llvm.data_layout"),
//                StringAttribute::new(&self.context, data_layout_ret).into(),
                StringAttribute::new(&self.context, data_layout_ret).into(),
//            ),
            ),
//        ])
        ])
//        .add_regions([module_region])
        .add_regions([module_region])
//        .build()?;
        .build()?;
//        assert!(op.verify(), "module operation is not valid");
        assert!(op.verify(), "module operation is not valid");
//

//        let mut module = Module::from_operation(op).expect("module failed to create");
        let mut module = Module::from_operation(op).expect("module failed to create");
//

//        let has_gas_builtin = program
        let has_gas_builtin = program
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin");
            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin");
//

//        let mut metadata = MetadataStorage::new();
        let mut metadata = MetadataStorage::new();
//        // Make the runtime library available.
        // Make the runtime library available.
//        metadata.insert(RuntimeBindingsMeta::default());
        metadata.insert(RuntimeBindingsMeta::default());
//        // We assume that GasMetadata will be always present when the program uses the gas builtin.
        // We assume that GasMetadata will be always present when the program uses the gas builtin.
//        let gas_metadata = if has_gas_builtin {
        let gas_metadata = if has_gas_builtin {
//            GasMetadata::new(program, Some(MetadataComputationConfig::default()))
            GasMetadata::new(program, Some(MetadataComputationConfig::default()))
//        } else {
        } else {
//            GasMetadata::new(program, None)
            GasMetadata::new(program, None)
//        }?;
        }?;
//        // Unwrapping here is not necessary since the insertion will only fail if there was
        // Unwrapping here is not necessary since the insertion will only fail if there was
//        // already some metadata of the same type.
        // already some metadata of the same type.
//        metadata.insert(gas_metadata);
        metadata.insert(gas_metadata);
//

//        // Create the Sierra program registry
        // Create the Sierra program registry
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;
//

//        crate::compile(
        crate::compile(
//            &self.context,
            &self.context,
//            &module,
            &module,
//            program,
            program,
//            &registry,
            &registry,
//            &mut metadata,
            &mut metadata,
//            debug_locations.as_ref(),
            debug_locations.as_ref(),
//        )?;
        )?;
//

//        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP_PREPASS") {
        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP_PREPASS") {
//            if x == "1" || x == "true" {
            if x == "1" || x == "true" {
//                std::fs::write("dump-prepass.mlir", module.as_operation().to_string())
                std::fs::write("dump-prepass.mlir", module.as_operation().to_string())
//                    .expect("should work");
                    .expect("should work");
//                std::fs::write(
                std::fs::write(
//                    "dump-prepass-debug.mlir",
                    "dump-prepass-debug.mlir",
//                    module.as_operation().to_string_with_flags(
                    module.as_operation().to_string_with_flags(
//                        OperationPrintingFlags::new().enable_debug_info(true, true),
                        OperationPrintingFlags::new().enable_debug_info(true, true),
//                    )?,
                    )?,
//                )
                )
//                .expect("should work");
                .expect("should work");
//            }
            }
//        }
        }
//

//        run_pass_manager(&self.context, &mut module)?;
        run_pass_manager(&self.context, &mut module)?;
//

//        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
//            if x == "1" || x == "true" {
            if x == "1" || x == "true" {
//                std::fs::write("dump.mlir", module.as_operation().to_string())
                std::fs::write("dump.mlir", module.as_operation().to_string())
//                    .expect("should work");
                    .expect("should work");
//                std::fs::write(
                std::fs::write(
//                    "dump-debug.mlir",
                    "dump-debug.mlir",
//                    module.as_operation().to_string_with_flags(
                    module.as_operation().to_string_with_flags(
//                        OperationPrintingFlags::new().enable_debug_info(true, true),
                        OperationPrintingFlags::new().enable_debug_info(true, true),
//                    )?,
                    )?,
//                )
                )
//                .expect("should work");
                .expect("should work");
//            }
            }
//        }
        }
//

//        // The func to llvm pass has a bug where it sets the data layout string to ""
        // The func to llvm pass has a bug where it sets the data layout string to ""
//        // This works around it by setting it again.
        // This works around it by setting it again.
//        {
        {
//            let mut op = module.as_operation_mut();
            let mut op = module.as_operation_mut();
//            op.set_attribute(
            op.set_attribute(
//                "llvm.data_layout",
                "llvm.data_layout",
//                StringAttribute::new(&self.context, data_layout_ret).into(),
                StringAttribute::new(&self.context, data_layout_ret).into(),
//            );
            );
//        }
        }
//

//        Ok(NativeModule::new(module, registry, metadata))
        Ok(NativeModule::new(module, registry, metadata))
//    }
    }
//

//    /// Compiles a sierra program into MLIR and then lowers to LLVM. Using the given metadata.
    /// Compiles a sierra program into MLIR and then lowers to LLVM. Using the given metadata.
//    /// Returns the corresponding NativeModule struct.
    /// Returns the corresponding NativeModule struct.
//    pub fn compile_with_metadata(
    pub fn compile_with_metadata(
//        &self,
        &self,
//        program: &Program,
        program: &Program,
//        metadata_config: MetadataComputationConfig,
        metadata_config: MetadataComputationConfig,
//    ) -> Result<NativeModule, Error> {
    ) -> Result<NativeModule, Error> {
//        let mut module = Module::new(Location::unknown(&self.context));
        let mut module = Module::new(Location::unknown(&self.context));
//

//        let mut metadata = MetadataStorage::new();
        let mut metadata = MetadataStorage::new();
//        // Make the runtime library available.
        // Make the runtime library available.
//        metadata.insert(RuntimeBindingsMeta::default());
        metadata.insert(RuntimeBindingsMeta::default());
//

//        let gas_metadata = GasMetadata::new(program, Some(metadata_config))?;
        let gas_metadata = GasMetadata::new(program, Some(metadata_config))?;
//        metadata.insert(gas_metadata);
        metadata.insert(gas_metadata);
//

//        // Create the Sierra program registry
        // Create the Sierra program registry
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;
//

//        crate::compile(
        crate::compile(
//            &self.context,
            &self.context,
//            &module,
            &module,
//            program,
            program,
//            &registry,
            &registry,
//            &mut metadata,
            &mut metadata,
//            None,
            None,
//        )?;
        )?;
//

//        run_pass_manager(&self.context, &mut module)?;
        run_pass_manager(&self.context, &mut module)?;
//

//        Ok(NativeModule::new(module, registry, metadata))
        Ok(NativeModule::new(module, registry, metadata))
//    }
    }
//}
}
//

///// Initialize an MLIR context.
/// Initialize an MLIR context.
//pub fn initialize_mlir() -> Context {
pub fn initialize_mlir() -> Context {
//    let context = Context::new();
    let context = Context::new();
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
//    register_all_passes();
    register_all_passes();
//    register_all_llvm_translations(&context);
    register_all_llvm_translations(&context);
//    context
    context
//}
}
