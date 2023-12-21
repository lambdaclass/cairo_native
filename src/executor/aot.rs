use crate::{
    error::jit_engine::RunnerError,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{gas::GasMetadata, syscall_handler::SyscallHandlerMeta},
    utils::generate_function_name,
    values::JitValue,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use libc::c_void;
use libloading::Library;
use starknet_types_core::felt::Felt;

pub struct AotNativeExecutor {
    library: Library,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    gas_metadata: Option<GasMetadata>,
}

impl AotNativeExecutor {
    pub fn new(
        library: Library,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        gas_metadata: Option<GasMetadata>,
    ) -> Self {
        Self {
            library,
            registry,
            gas_metadata,
        }
    }
}

impl AotNativeExecutor {
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        mut gas: Option<u128>,
        syscall_handler: Option<&SyscallHandlerMeta>,
    ) -> ExecutionResult {
        self.process_required_initial_gas(function_id, gas.as_mut());

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            gas,
            syscall_handler.map(SyscallHandlerMeta::as_ptr),
        )
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
        unsafe {
            self.library
                .get::<extern "C" fn()>(function_name.as_bytes())
                .unwrap()
                .into_raw()
                .into_raw()
        }
    }

    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
        &self.registry.get_function(function_id).unwrap().signature
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
