use crate::{
    executor::ExecutorBase,
    metadata::{gas::GasMetadata, MetadataStorage},
};
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
    pub(crate) executor_base: ExecutorBase,
    pub(crate) gas_metadata: GasMetadata,
}

impl<'ctx> NativeModule<'ctx> {
    pub fn new(
        context: &'ctx Context,
        module: Module<'ctx>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        function_ids: &[FunctionId],
        mut metadata: MetadataStorage,
    ) -> Self {
        let executor_base =
            ExecutorBase::new(context, &module, &registry, &mut metadata, function_ids);

        Self {
            context,
            module,
            registry,
            executor_base,
            gas_metadata: metadata.remove::<GasMetadata>().unwrap(),
        }
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }
}

impl Debug for NativeModule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.module.as_operation().to_string())
    }
}
