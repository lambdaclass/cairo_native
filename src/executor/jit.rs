//use crate::{
use crate::{
//    error::Error,
    error::Error,
//    execution_result::{ContractExecutionResult, ExecutionResult},
    execution_result::{ContractExecutionResult, ExecutionResult},
//    metadata::gas::GasMetadata,
    metadata::gas::GasMetadata,
//    module::NativeModule,
    module::NativeModule,
//    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
//    utils::{create_engine, generate_function_name},
    utils::{create_engine, generate_function_name},
//    values::JitValue,
    values::JitValue,
//    OptLevel,
    OptLevel,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreLibfunc, CoreType},
    extensions::core::{CoreLibfunc, CoreType},
//    ids::FunctionId,
    ids::FunctionId,
//    program::FunctionSignature,
    program::FunctionSignature,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use libc::c_void;
use libc::c_void;
//use melior::{ir::Module, ExecutionEngine};
use melior::{ir::Module, ExecutionEngine};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

///// A MLIR JIT execution engine in the context of Cairo Native.
/// A MLIR JIT execution engine in the context of Cairo Native.
//pub struct JitNativeExecutor<'m> {
pub struct JitNativeExecutor<'m> {
//    engine: ExecutionEngine,
    engine: ExecutionEngine,
//

//    module: Module<'m>,
    module: Module<'m>,
//    registry: ProgramRegistry<CoreType, CoreLibfunc>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
//

//    gas_metadata: GasMetadata,
    gas_metadata: GasMetadata,
//}
}
//

//impl std::fmt::Debug for JitNativeExecutor<'_> {
impl std::fmt::Debug for JitNativeExecutor<'_> {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        f.debug_struct("JitNativeExecutor")
        f.debug_struct("JitNativeExecutor")
//            .field("module", &self.module)
            .field("module", &self.module)
//            .field("gas_metadata", &self.gas_metadata)
            .field("gas_metadata", &self.gas_metadata)
//            .finish()
            .finish()
//    }
    }
//}
}
//

//impl<'m> JitNativeExecutor<'m> {
impl<'m> JitNativeExecutor<'m> {
//    pub fn from_native_module(native_module: NativeModule<'m>, opt_level: OptLevel) -> Self {
    pub fn from_native_module(native_module: NativeModule<'m>, opt_level: OptLevel) -> Self {
//        let NativeModule {
        let NativeModule {
//            module,
            module,
//            registry,
            registry,
//            metadata,
            metadata,
//        } = native_module;
        } = native_module;
//

//        Self {
        Self {
//            engine: create_engine(&module, &metadata, opt_level),
            engine: create_engine(&module, &metadata, opt_level),
//            module,
            module,
//            registry,
            registry,
//            gas_metadata: metadata.get::<GasMetadata>().cloned().unwrap(),
            gas_metadata: metadata.get::<GasMetadata>().cloned().unwrap(),
//        }
        }
//    }
    }
//

//    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
    pub fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
//        &self.registry
        &self.registry
//    }
    }
//

//    pub fn module(&self) -> &Module<'m> {
    pub fn module(&self) -> &Module<'m> {
//        &self.module
        &self.module
//    }
    }
//

//    /// Execute a program with the given params.
    /// Execute a program with the given params.
//    ///
    ///
//    /// See [`cairo_native::jit_runner::execute`]
    /// See [`cairo_native::jit_runner::execute`]
//    pub fn invoke_dynamic(
    pub fn invoke_dynamic(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[JitValue],
        args: &[JitValue],
//        gas: Option<u128>,
        gas: Option<u128>,
//    ) -> Result<ExecutionResult, Error> {
    ) -> Result<ExecutionResult, Error> {
//        let available_gas = self
        let available_gas = self
//            .gas_metadata
            .gas_metadata
//            .get_initial_available_gas(function_id, gas)
            .get_initial_available_gas(function_id, gas)
//            .map_err(|_| crate::error::Error::InsufficientGasError)?;
            .map_err(|_| crate::error::Error::InsufficientGasError)?;
//

//        super::invoke_dynamic(
        super::invoke_dynamic(
//            &self.registry,
            &self.registry,
//            self.find_function_ptr(function_id),
            self.find_function_ptr(function_id),
//            self.extract_signature(function_id),
            self.extract_signature(function_id),
//            args,
            args,
//            available_gas,
            available_gas,
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        )
        )
//    }
    }
//

//    /// Execute a program with the given params.
    /// Execute a program with the given params.
//    ///
    ///
//    /// See [`cairo_native::jit_runner::execute`]
    /// See [`cairo_native::jit_runner::execute`]
//    pub fn invoke_dynamic_with_syscall_handler(
    pub fn invoke_dynamic_with_syscall_handler(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[JitValue],
        args: &[JitValue],
//        gas: Option<u128>,
        gas: Option<u128>,
//        syscall_handler: impl StarknetSyscallHandler,
        syscall_handler: impl StarknetSyscallHandler,
//    ) -> Result<ExecutionResult, Error> {
    ) -> Result<ExecutionResult, Error> {
//        let available_gas = self
        let available_gas = self
//            .gas_metadata
            .gas_metadata
//            .get_initial_available_gas(function_id, gas)
            .get_initial_available_gas(function_id, gas)
//            .map_err(|_| crate::error::Error::InsufficientGasError)?;
            .map_err(|_| crate::error::Error::InsufficientGasError)?;
//

//        super::invoke_dynamic(
        super::invoke_dynamic(
//            &self.registry,
            &self.registry,
//            self.find_function_ptr(function_id),
            self.find_function_ptr(function_id),
//            self.extract_signature(function_id),
            self.extract_signature(function_id),
//            args,
            args,
//            available_gas,
            available_gas,
//            Some(syscall_handler),
            Some(syscall_handler),
//        )
        )
//    }
    }
//

//    pub fn invoke_contract_dynamic(
    pub fn invoke_contract_dynamic(
//        &self,
        &self,
//        function_id: &FunctionId,
        function_id: &FunctionId,
//        args: &[Felt],
        args: &[Felt],
//        gas: Option<u128>,
        gas: Option<u128>,
//        syscall_handler: impl StarknetSyscallHandler,
        syscall_handler: impl StarknetSyscallHandler,
//    ) -> Result<ContractExecutionResult, Error> {
    ) -> Result<ContractExecutionResult, Error> {
//        let available_gas = self
        let available_gas = self
//            .gas_metadata
            .gas_metadata
//            .get_initial_available_gas(function_id, gas)
            .get_initial_available_gas(function_id, gas)
//            .map_err(|_| crate::error::Error::InsufficientGasError)?;
            .map_err(|_| crate::error::Error::InsufficientGasError)?;
//        // TODO: Check signature for contract interface.
        // TODO: Check signature for contract interface.
//        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
//            &self.registry,
            &self.registry,
//            self.find_function_ptr(function_id),
            self.find_function_ptr(function_id),
//            self.extract_signature(function_id),
            self.extract_signature(function_id),
//            &[JitValue::Struct {
            &[JitValue::Struct {
//                fields: vec![JitValue::Array(
                fields: vec![JitValue::Array(
//                    args.iter().cloned().map(JitValue::Felt252).collect(),
                    args.iter().cloned().map(JitValue::Felt252).collect(),
//                )],
                )],
//                // TODO: Populate `debug_name`.
                // TODO: Populate `debug_name`.
//                debug_name: None,
                debug_name: None,
//            }],
            }],
//            available_gas,
            available_gas,
//            Some(syscall_handler),
            Some(syscall_handler),
//        )?)
        )?)
//    }
    }
//

//    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
//        let function_name = generate_function_name(function_id);
        let function_name = generate_function_name(function_id);
//        let function_name = format!("_mlir_ciface_{function_name}");
        let function_name = format!("_mlir_ciface_{function_name}");
//

//        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
//        self.engine.lookup(&function_name) as *mut c_void
        self.engine.lookup(&function_name) as *mut c_void
//    }
    }
//

//    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
//        &self
        &self
//            .program_registry()
            .program_registry()
//            .get_function(function_id)
            .get_function(function_id)
//            .unwrap()
            .unwrap()
//            .signature
            .signature
//    }
    }
//}
}
