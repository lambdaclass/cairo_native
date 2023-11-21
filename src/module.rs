use crate::metadata::{gas::GasMetadata, MetadataStorage};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program::Program,
    program_registry::ProgramRegistry,
};
use melior::ir::Module;
use std::{collections::HashMap, fmt::Debug};

/// A MLIR module in the context of Cairo Native.
/// It is conformed by the MLIR module, the Sierra program registry
/// and the program metadata.
pub struct NativeModule<'m> {
    module: Module<'m>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    initial_gas: HashMap<FunctionId, u128>,

    #[cfg(feature = "with-debug-utils")]
    pub(crate) debug_utils: crate::metadata::debug_utils::DebugUtils,
}

impl<'m> NativeModule<'m> {
    pub fn new(
        module: Module<'m>,
        program: &Program,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: MetadataStorage,
    ) -> Self {
        Self {
            module,
            registry,
            initial_gas: metadata
                .get::<GasMetadata>()
                .map(|gas_meta| {
                    program
                        .funcs
                        .iter()
                        .filter_map(|function| {
                            gas_meta
                                .get_initial_required_gas(&function.id)
                                .map(|gas| (function.id.clone(), gas))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            #[cfg(feature = "with-debug-utils")]
            debug_utils: metadata
                .get::<crate::metadata::debug_utils::DebugUtils>()
                .unwrap()
                .clone(),
        }
    }

    /// Given some contract function's id, returns an option of the required
    /// initial gas to execute it.
    /// If no initial gas is required, `None` is returned.
    pub fn get_required_init_gas(&self, fn_id: &FunctionId) -> Option<u128> {
        self.initial_gas.get(fn_id).copied()
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
