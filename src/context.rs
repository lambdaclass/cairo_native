use crate::{
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
        Attribute, AttributeLike, Block, Identifier, Location, Module, Region,
    },
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
    Context,
};
use mlir_sys::{
    mlirDisctinctAttrCreate, mlirLLVMDICompileUnitAttrGet, mlirLLVMDIFileAttrGet,
    mlirLLVMDIModuleAttrGet, MlirLLVMDIEmissionKind_MlirLLVMDIEmissionKindFull,
    MlirLLVMDINameTableKind_MlirLLVMDINameTableKindDefault,
};
use std::sync::OnceLock;

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

    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Compiles a sierra program into MLIR and then lowers to LLVM.
    /// Returns the corresponding NativeModule struct.
    ///
    /// If `ignore_debug_names` is true then debug names will not be added to function names.
    /// Mainly useful for the ContractExecutor.
    pub fn compile(
        &self,
        program: &Program,
        ignore_debug_names: bool,
    ) -> Result<NativeModule, Error> {
        static INITIALIZED: OnceLock<()> = OnceLock::new();
        INITIALIZED.get_or_init(|| unsafe {
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmPrinters();
            tracing::debug!("initialized llvm targets");
        });
        let target_triple = get_target_triple();

        let module_region = Region::new();
        module_region.append_block(Block::new(&[]));

        let data_layout_ret = &get_data_layout_rep()?;

        let di_unit_id = unsafe {
            let id = StringAttribute::new(&self.context, "compile_unit_id").to_raw();
            mlirDisctinctAttrCreate(id)
        };

        let op = OperationBuilder::new(
            "builtin.module",
            Location::fused(
                &self.context,
                &[Location::new(&self.context, "program.sierra", 0, 0)],
                {
                    let file_attr = unsafe {
                        Attribute::from_raw(mlirLLVMDIFileAttrGet(
                            self.context.to_raw(),
                            StringAttribute::new(&self.context, "program.sierra").to_raw(),
                            StringAttribute::new(&self.context, "").to_raw(),
                        ))
                    };
                    unsafe {
                        let di_unit = mlirLLVMDICompileUnitAttrGet(
                            self.context.to_raw(),
                            di_unit_id,
                            0x1c, // rust
                            file_attr.to_raw(),
                            StringAttribute::new(&self.context, "cairo-native").to_raw(),
                            false,
                            MlirLLVMDIEmissionKind_MlirLLVMDIEmissionKindFull,
                            MlirLLVMDINameTableKind_MlirLLVMDINameTableKindDefault,
                        );

                        let context = &self.context;

                        let di_module = mlirLLVMDIModuleAttrGet(
                            context.to_raw(),
                            file_attr.to_raw(),
                            di_unit,
                            StringAttribute::new(context, "LLVMDialectModule").to_raw(),
                            StringAttribute::new(context, "").to_raw(),
                            StringAttribute::new(context, "").to_raw(),
                            StringAttribute::new(context, "").to_raw(),
                            0,
                            false,
                        );

                        Attribute::from_raw(di_module)
                    }
                },
            ),
        )
        .add_attributes(&[
            (
                Identifier::new(&self.context, "llvm.target_triple"),
                StringAttribute::new(&self.context, &target_triple).into(),
            ),
            (
                Identifier::new(&self.context, "llvm.data_layout"),
                StringAttribute::new(&self.context, data_layout_ret).into(),
            ),
        ])
        .add_regions([module_region])
        .build()?;
        assert!(op.verify(), "module operation is not valid");

        let mut module = Module::from_operation(op).expect("module failed to create");

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
            unsafe { Attribute::from_raw(di_unit_id) },
            ignore_debug_names,
        )?;

        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
            if x == "1" || x == "true" {
                std::fs::write("dump-prepass.mlir", module.as_operation().to_string())
                    .expect("should work");
                std::fs::write(
                    "dump-prepass-debug-valid.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, false),
                    )?,
                )
                .expect("failed to write dump-prepass-debug-valid.mlir");
                std::fs::write(
                    "dump-prepass-debug-pretty.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, false),
                    )?,
                )
                .expect("failed to writedump-prepass-debug-pretty.mlir");
            }
        }

        run_pass_manager(&self.context, &mut module)?;

        if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
            if x == "1" || x == "true" {
                std::fs::write("dump.mlir", module.as_operation().to_string())
                    .expect("should work");
                std::fs::write(
                    "dump-debug-pretty.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, false),
                    )?,
                )
                .expect("failed to write dump-debug-pretty.mlir");
                std::fs::write(
                    "dump-debug.mlir",
                    module.as_operation().to_string_with_flags(
                        OperationPrintingFlags::new().enable_debug_info(true, false),
                    )?,
                )
                .expect("failed to write dump-debug.mlir");
            }
        }

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
