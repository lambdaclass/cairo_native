use crate::metadata::gas::GasMetadata;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::{ir::Module, Context};
use std::fmt::Debug;

/// A MLIR module in the context of Cairo Native.
/// It is conformed by the MLIR module, the Sierra program registry
/// and the program metadata.
pub struct NativeModule<'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) module: Module<'ctx>,
    pub(crate) registry: ProgramRegistry<CoreType, CoreLibfunc>,
    pub(crate) function_ids: Vec<FunctionId>,
    pub(crate) gas_metadata: GasMetadata,
}

impl<'ctx> NativeModule<'ctx> {
    pub fn new(
        context: &'ctx Context,
        module: Module<'ctx>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        function_ids: impl Into<Vec<FunctionId>>,
        gas_metadata: GasMetadata,
    ) -> Self {
        Self {
            context,
            module,
            registry,
            function_ids: function_ids.into(),
            gas_metadata,
        }
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }

    pub fn function_ids(&self) -> &[FunctionId] {
        &self.function_ids
    }
}

impl Debug for NativeModule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.module.as_operation().to_string())
    }
}
