use crate::{
    error::jit_engine::RunnerError, execute, execute_contract,
    execution_result::ContractExecutionResult, module::NativeModule,
    starknet::StarkNetSyscallHandler, utils::create_engine, values::JitValue, ExecutionResult,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::ExecutionEngine;

/// A MLIR JIT execution engine in the context of Cairo Native.
pub struct NativeJitEngine<'m> {
    engine: ExecutionEngine,
    // NativeModule needs to be kept alive with the executor or it will segfault when trying to execute.
    native_module: NativeModule<'m>,
}

impl<'m> NativeJitEngine<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let module = native_module.module();
        let engine = create_engine(module);

        #[cfg(feature = "with-debug-utils")]
        native_module.debug_utils.register_impls(&engine);

        Self {
            engine,
            native_module,
        }
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        self.native_module.program_registry()
    }

    pub fn module(&self) -> &NativeModule<'m> {
        &self.native_module
    }

    /// Execute a program with the given params.
    ///
    /// See [`execute`].
    pub fn execute(
        &self,
        function_id: &FunctionId,
        params: &[JitValue],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, RunnerError> {
        execute(
            &self.engine,
            self.program_registry(),
            function_id,
            params,
            self.native_module.get_required_init_gas(function_id),
            gas,
        )
    }

    /// Execute a contract with the given calldata.
    ///
    /// See [`execute_contract`].
    pub fn execute_contract<T>(
        &self,
        function_id: &FunctionId,
        params: &[JitValue],
        syscall_handler: &mut T,
        gas: Option<u128>,
    ) -> Result<ContractExecutionResult, RunnerError>
    where
        T: StarkNetSyscallHandler,
    {
        execute_contract(
            &self.engine,
            self.program_registry(),
            function_id,
            params,
            self.native_module.get_required_init_gas(function_id),
            gas,
            syscall_handler,
        )
    }
}
