use crate::{
    error::jit_engine::{ErrorImpl, RunnerError},
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::syscall_handler::SyscallHandlerMeta,
    module::NativeModule,
    utils::{create_engine, generate_function_name},
    values::JitValue,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::ExecutionEngine;
use mlir_sys::{mlirExecutionEngineLookup, MlirExecutionEngine, MlirStringRef};

/// A MLIR JIT execution engine in the context of Cairo Native.
pub struct JitNativeExecutor<'m> {
    engine: ExecutionEngine,
    // NativeModule needs to be kept alive with the executor or it will segfault when trying to execute.
    native_module: NativeModule<'m>,
}

impl<'m> JitNativeExecutor<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let module = native_module.module();
        let engine = create_engine(module, native_module.metadata());
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

    pub fn module_mut(&mut self) -> &mut NativeModule<'m> {
        &mut self.native_module
    }

    /// Execute a program with the given params.
    ///
    /// See [`cairo_native::jit_runner::execute`]
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, RunnerError> {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        let function_ptr = unsafe {
            // FIXME: Code a PR to extract the raw pointer from `ExecutionEngine`s.
            mlirExecutionEngineLookup(
                *std::mem::transmute::<&ExecutionEngine, &MlirExecutionEngine>(&self.engine),
                MlirStringRef {
                    data: function_name.as_ptr() as *const i8,
                    length: function_name.len(),
                },
            )
        };

        let function_signature = &self
            .program_registry()
            .get_function(function_id)
            .unwrap()
            .signature;

        Ok(super::invoke_dynamic(
            self.program_registry(),
            function_ptr,
            function_signature,
            args,
            gas,
        ))
    }

    /// Execute a contract with the given calldata.
    ///
    /// See [`cairo_native::jit_runner::execute_contract`]
    pub fn execute_contract(
        &self,
        _fn_id: &FunctionId,
        _calldata: &[JitValue],
        _gas: u128,
    ) -> Result<ContractExecutionResult, RunnerError> {
        let _registry = self.program_registry();
        let _syscall_handler = self
            .module()
            .get_metadata::<SyscallHandlerMeta>()
            .ok_or(RunnerError::from(ErrorImpl::MissingSyscallHandler))?;

        // Note: It appears this isn't being checked when running contracts on the VM.
        // let required_initial_gas = self.native_module.get_required_init_gas(fn_id);

        todo!()
        // execute_contract(
        //     &self.engine,
        //     registry,
        //     fn_id,
        //     calldata,
        //     None,
        //     gas,
        //     syscall_handler,
        // )
    }
}
