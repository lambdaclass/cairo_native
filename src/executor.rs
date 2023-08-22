use crate::{error::jit_engine::RunnerError, execute, module::NativeModule, utils::create_engine};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::ExecutionEngine;
use serde::{Deserializer, Serializer};

/// A MLIR JIT execution engine in the context of Cairo Native.
pub struct NativeExecutor<'m> {
    engine: ExecutionEngine,
    native_module: NativeModule<'m>,
}

impl<'m> NativeExecutor<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let module = native_module.get_module();
        let engine = create_engine(module);
        Self {
            engine,
            native_module,
        }
    }

    pub fn get_program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        self.native_module.get_program_registry()
    }

    pub fn execute<'de, D, S>(
        &self,
        fn_id: &FunctionId,
        params: D,
        returns: S,
        required_init_gas: Option<u64>,
    ) -> Result<S::Ok, RunnerError<'de, D, S>>
    where
        D: Deserializer<'de>,
        S: Serializer,
    {
        let registry = self.get_program_registry();

        Ok(execute(
            &self.engine,
            registry,
            fn_id,
            params,
            returns,
            required_init_gas,
        )?)
    }
}
