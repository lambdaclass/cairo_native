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
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use libc::c_void;
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
        mut gas: Option<u128>,
    ) -> Result<ExecutionResult, RunnerError> {
        self.process_required_initial_gas(function_id, gas.as_mut());

        Ok(super::invoke_dynamic(
            self.program_registry(),
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            gas,
            None,
        ))
    }

    /// Execute a contract with the given calldata.
    ///
    /// See [`cairo_native::jit_runner::execute_contract`]
    pub fn execute_contract(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        mut gas: Option<u128>,
    ) -> Result<ContractExecutionResult, RunnerError> {
        self.process_required_initial_gas(function_id, gas.as_mut());

        let syscall_handler = self
            .module()
            .get_metadata::<SyscallHandlerMeta>()
            .ok_or(RunnerError::from(ErrorImpl::MissingSyscallHandler))?;

        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            self.program_registry(),
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            gas,
            Some(syscall_handler.as_ptr()),
        ))
    }

    fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        unsafe {
            // FIXME: Code a PR to extract the raw pointer from `ExecutionEngine`s.
            mlirExecutionEngineLookup(
                *std::mem::transmute::<&ExecutionEngine, &MlirExecutionEngine>(&self.engine),
                MlirStringRef {
                    data: function_name.as_ptr() as *const i8,
                    length: function_name.len(),
                },
            )
        }
    }

    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
        &self
            .program_registry()
            .get_function(function_id)
            .unwrap()
            .signature
    }

    fn process_required_initial_gas(&self, function_id: &FunctionId, gas: Option<&mut u128>) {
        if let (Some(gas), Some(required_init_gas)) =
            (gas, self.native_module.get_required_init_gas(function_id))
        {
            if required_init_gas > *gas {
                panic!("Not enough gas");
            }

            *gas -= required_init_gas;
        }
    }
}
