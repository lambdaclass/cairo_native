use crate::{
    error::jit_engine::RunnerError,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{gas::GasMetadata, syscall_handler::SyscallHandlerMeta},
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
use melior::{ir::Module, ExecutionEngine};
use starknet_types_core::felt::Felt;

/// A MLIR JIT execution engine in the context of Cairo Native.
pub struct JitNativeExecutor<'m> {
    engine: ExecutionEngine,

    module: Module<'m>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    gas_metadata: Option<GasMetadata>,
}

impl<'m> JitNativeExecutor<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let NativeModule {
            module,
            registry,
            metadata,
        } = native_module;

        Self {
            engine: create_engine(&module, &metadata),
            module,
            registry,
            gas_metadata: metadata.get::<GasMetadata>().cloned(),
        }
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }

    pub fn module(&self) -> &Module<'m> {
        &self.module
    }

    /// Execute a program with the given params.
    ///
    /// See [`cairo_native::jit_runner::execute`]
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        mut gas: Option<u128>,
        syscall_handler: Option<&SyscallHandlerMeta>,
    ) -> Result<ExecutionResult, RunnerError> {
        self.process_required_initial_gas(function_id, gas.as_mut());

        Ok(super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            gas,
            syscall_handler.map(SyscallHandlerMeta::as_ptr),
        ))
    }

    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        mut gas: Option<u128>,
        syscall_handler: Option<&SyscallHandlerMeta>,
    ) -> Result<ContractExecutionResult, RunnerError> {
        self.process_required_initial_gas(function_id, gas.as_mut());

        // TODO: Check signature for contract interface.
        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            &[JitValue::Struct {
                fields: vec![JitValue::Array(
                    args.iter().cloned().map(JitValue::Felt252).collect(),
                )],
                // TODO: Populate `debug_name`.
                debug_name: None,
            }],
            gas,
            syscall_handler.map(SyscallHandlerMeta::as_ptr),
        ))
    }

    fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        self.engine.lookup(&function_name) as *mut c_void
    }

    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
        &self
            .program_registry()
            .get_function(function_id)
            .unwrap()
            .signature
    }

    fn process_required_initial_gas(&self, function_id: &FunctionId, gas: Option<&mut u128>) {
        if let (Some(gas), Some(required_init_gas)) = (
            gas,
            self.gas_metadata
                .as_ref()
                .and_then(|gas_metadata| gas_metadata.get_initial_required_gas(function_id)),
        ) {
            if required_init_gas > *gas {
                panic!("Not enough gas");
            }

            *gas -= required_init_gas;
        }
    }
}
