use crate::{
    error::jit_engine::RunnerError, execute, invoke::JITValue, jit_runner::ExecuteResult,
    metadata::syscall_handler::SyscallHandlerMeta, module::NativeModule, utils::create_engine,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::ExecutionEngine;

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

    pub fn execute(
        &self,
        fn_id: &FunctionId,
        params: &[JITValue],
        required_initial_gas: Option<u128>,
        gas: Option<u128>,
    ) -> Result<ExecuteResult, RunnerError> {
        let registry = self.get_program_registry();
        let syscall_handler = self.get_module().get_metadata::<SyscallHandlerMeta>();

        execute(
            &self.engine,
            registry,
            fn_id,
            params,
            required_initial_gas,
            gas,
            syscall_handler,
        )
    }
}
