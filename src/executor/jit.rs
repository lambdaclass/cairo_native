use crate::{
    error::executor::Result,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{gas::GasMetadata, syscall_handler::SyscallHandlerMeta},
    module::NativeModule,
    utils::{create_engine, generate_function_name},
    values::JitValue,
    OptLevel,
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

impl std::fmt::Debug for JitNativeExecutor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitNativeExecutor")
            .field("module", &self.module)
            .field("gas_metadata", &self.gas_metadata)
            .finish()
    }
}

impl<'m> JitNativeExecutor<'m> {
    /// Create a new JIT-compiled executor from an MLIR module and an optimization level.
    pub fn new(native_module: NativeModule<'m>, opt_level: OptLevel) -> Self {
        let NativeModule {
            module,
            registry,
            metadata,
        } = native_module;

        Self {
            engine: create_engine(&module, &metadata, opt_level),
            module,
            registry,
            gas_metadata: metadata.get::<GasMetadata>().cloned(),
        }
    }

    /// Return the MLIR module.
    pub fn module(&self) -> &Module<'m> {
        &self.module
    }

    /// Return the program registry.
    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }

    /// Return the gas metadata (if any).
    pub fn gas_metadata(&self) -> Option<&GasMetadata> {
        self.gas_metadata.as_ref()
    }

    /// Execute a program with the given params.
    ///
    /// See [`NativeExecutor::invoke_dynamic`](crate::executor::NativeExecutor::invoke_dynamic).
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        mut gas: Option<u128>,
        syscall_handler: Option<&SyscallHandlerMeta>,
    ) -> Result<ExecutionResult> {
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

    /// Execute a contract with the given params, gas and syscall handler.
    ///
    /// See [`NativeExecutor::invoke_contract_dynamic`](crate::executor::NativeExecutor::invoke_contract_dynamic).
    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        mut gas: Option<u128>,
        syscall_handler: Option<&SyscallHandlerMeta>,
    ) -> Result<ContractExecutionResult> {
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

    /// Find the function pointer of an entry point given its function id.
    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
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
