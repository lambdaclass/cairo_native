use crate::{
    error::Error,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::gas::GasMetadata,
    module::NativeModule,
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    utils::{create_engine, generate_function_name},
    values::Value,
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

    gas_metadata: GasMetadata,
}

unsafe impl<'a> Send for JitNativeExecutor<'a> {}
unsafe impl<'a> Sync for JitNativeExecutor<'a> {}

impl std::fmt::Debug for JitNativeExecutor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitNativeExecutor")
            .field("module", &self.module)
            .field("gas_metadata", &self.gas_metadata)
            .finish()
    }
}

impl<'m> JitNativeExecutor<'m> {
    pub fn from_native_module(native_module: NativeModule<'m>, opt_level: OptLevel) -> Self {
        let NativeModule {
            module,
            registry,
            metadata,
        } = native_module;

        Self {
            engine: create_engine(&module, &metadata, opt_level),
            module,
            registry,
            gas_metadata: metadata.get::<GasMetadata>().cloned().unwrap(),
        }
    }

    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }

    pub fn module(&self) -> &Module<'m> {
        &self.module
    }

    /// Execute a program with the given params.
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Value],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.find_symbol_ptr("builtin_costs"),
            self.extract_signature(function_id).unwrap(),
            args,
            available_gas,
            Option::<DummySyscallHandler>::None,
        )
    }

    /// Execute a program with the given params.
    pub fn invoke_dynamic_with_syscall_handler(
        &self,
        function_id: &FunctionId,
        args: &[Value],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.find_symbol_ptr("builtin_costs"),
            self.extract_signature(function_id).unwrap(),
            args,
            available_gas,
            Some(syscall_handler),
        )
    }

    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;
        // TODO: Check signature for contract interface.
        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.find_symbol_ptr("builtin_costs"),
            self.extract_signature(function_id).unwrap(),
            &[Value::Struct {
                fields: vec![Value::Array(
                    args.iter().cloned().map(Value::Felt252).collect(),
                )],
                // TODO: Populate `debug_name`.
                debug_name: None,
            }],
            available_gas,
            Some(syscall_handler),
        )?)
    }

    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
        let function_name = generate_function_name(function_id, false);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        self.engine.lookup(&function_name) as *mut c_void
    }

    pub fn find_symbol_ptr(&self, name: &str) -> Option<*mut c_void> {
        let ptr = self.engine.lookup(name) as *mut c_void;

        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    fn extract_signature(&self, function_id: &FunctionId) -> Option<&FunctionSignature> {
        self.program_registry()
            .get_function(function_id)
            .ok()
            .map(|func| &func.signature)
    }
}
