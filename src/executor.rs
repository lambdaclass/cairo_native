use crate::{
    error::jit_engine::RunnerError, execute, invoke::InvokeArg, jit_runner::execute_args,
    module::NativeModule, utils::create_engine,
};
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
    // NativeModule needs to be kept alive with the executor or it will segfault when trying to execute.
    native_module: NativeModule<'m>,
}

impl<'m> NativeExecutor<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let module = native_module.module();
        let engine = create_engine(module, native_module.metadata());
        Self {
            engine,
            native_module,
        }
    }

    pub fn get_program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        self.native_module.program_registry()
    }

    pub fn get_module(&self) -> &NativeModule<'m> {
        &self.native_module
    }

    pub fn get_module_mut(&mut self) -> &mut NativeModule<'m> {
        &mut self.native_module
    }

    pub fn execute<'de, D, S>(
        &self,
        fn_id: &FunctionId,
        params: D,
        returns: S,
        required_init_gas: Option<u128>,
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

    pub fn execute_args(
        &self,
        fn_id: &FunctionId,
        params: &[InvokeArg],
        required_initial_gas: Option<u128>,
        gas: Option<u128>,
    ) -> Vec<InvokeArg> {
        let registry = self.get_program_registry();

        execute_args(
            &self.engine,
            registry,
            fn_id,
            params,
            required_initial_gas,
            gas,
        )
    }
}
