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
//    utils::generate_function_name,
    utils::generate_function_name,
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
//use educe::Educe;
use educe::Educe;
//use libc::c_void;
use libc::c_void;
//use libloading::Library;
use libloading::Library;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use tempfile::NamedTempFile;
use tempfile::NamedTempFile;
//

//#[derive(Educe)]
#[derive(Educe)]
//#[educe(Debug)]
#[educe(Debug)]
//pub struct AotNativeExecutor {
pub struct AotNativeExecutor {
//    #[educe(Debug(ignore))]
    #[educe(Debug(ignore))]
//    library: Library,
    library: Library,
//    #[educe(Debug(ignore))]
    #[educe(Debug(ignore))]
//    registry: ProgramRegistry<CoreType, CoreLibfunc>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
//

//    gas_metadata: GasMetadata,
    gas_metadata: GasMetadata,
//}
}
//

//impl AotNativeExecutor {
impl AotNativeExecutor {
//    pub fn new(
    pub fn new(
//        library: Library,
        library: Library,
//        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
//        gas_metadata: GasMetadata,
        gas_metadata: GasMetadata,
//    ) -> Self {
    ) -> Self {
//        Self {
        Self {
//            library,
            library,
//            registry,
            registry,
//            gas_metadata,
            gas_metadata,
//        }
        }
//    }
    }
//

//    /// Utility to convert a [`NativeModule`] into an [`AotNativeExecutor`].
    /// Utility to convert a [`NativeModule`] into an [`AotNativeExecutor`].
//    pub fn from_native_module(module: NativeModule, opt_level: OptLevel) -> Self {
    pub fn from_native_module(module: NativeModule, opt_level: OptLevel) -> Self {
//        let NativeModule {
        let NativeModule {
//            module,
            module,
//            registry,
            registry,
//            mut metadata,
            mut metadata,
//        } = module;
        } = module;
//

//        let library_path = NamedTempFile::new().unwrap().into_temp_path();
        let library_path = NamedTempFile::new().unwrap().into_temp_path();
//

//        let object_data = crate::module_to_object(&module, opt_level).unwrap();
        let object_data = crate::module_to_object(&module, opt_level).unwrap();
//        crate::object_to_shared_lib(&object_data, &library_path).unwrap();
        crate::object_to_shared_lib(&object_data, &library_path).unwrap();
//

//        Self {
        Self {
//            library: unsafe { Library::new(library_path).unwrap() },
            library: unsafe { Library::new(library_path).unwrap() },
//            registry,
            registry,
//            gas_metadata: metadata.remove().unwrap(),
            gas_metadata: metadata.remove().unwrap(),
//        }
        }
//    }
    }
//

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
//

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
//        unsafe {
        unsafe {
//            self.library
            self.library
//                .get::<extern "C" fn()>(function_name.as_bytes())
                .get::<extern "C" fn()>(function_name.as_bytes())
//                .unwrap()
                .unwrap()
//                .into_raw()
                .into_raw()
//                .into_raw()
                .into_raw()
//        }
        }
//    }
    }
//

//    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
    fn extract_signature(&self, function_id: &FunctionId) -> &FunctionSignature {
//        &self.registry.get_function(function_id).unwrap().signature
        &self.registry.get_function(function_id).unwrap().signature
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod tests {
mod tests {
//    use super::*;
    use super::*;
//    use crate::{
    use crate::{
//        context::NativeContext,
        context::NativeContext,
//        utils::test::{load_cairo, load_starknet, TestSyscallHandler},
        utils::test::{load_cairo, load_starknet, TestSyscallHandler},
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use rstest::*;
    use rstest::*;
//

//    #[fixture]
    #[fixture]
//    fn program() -> Program {
    fn program() -> Program {
//        let (_, program) = load_cairo! {
        let (_, program) = load_cairo! {
//            use core::starknet::{SyscallResultTrait, get_block_hash_syscall};
            use core::starknet::{SyscallResultTrait, get_block_hash_syscall};
//

//            fn run_test() -> felt252 {
            fn run_test() -> felt252 {
//                42
                42
//            }
            }
//

//            fn get_block_hash() -> felt252 {
            fn get_block_hash() -> felt252 {
//                get_block_hash_syscall(1).unwrap_syscall()
                get_block_hash_syscall(1).unwrap_syscall()
//            }
            }
//        };
        };
//        program
        program
//    }
    }
//

//    #[fixture]
    #[fixture]
//    fn starknet_program() -> Program {
    fn starknet_program() -> Program {
//        let (_, program) = load_starknet! {
        let (_, program) = load_starknet! {
//            #[starknet::interface]
            #[starknet::interface]
//            trait ISimpleStorage<TContractState> {
            trait ISimpleStorage<TContractState> {
//                fn get(self: @TContractState) -> u128;
                fn get(self: @TContractState) -> u128;
//            }
            }
//

//            #[starknet::contract]
            #[starknet::contract]
//            mod contract {
            mod contract {
//                #[storage]
                #[storage]
//                struct Storage {}
                struct Storage {}
//

//                #[abi(embed_v0)]
                #[abi(embed_v0)]
//                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
//                    fn get(self: @ContractState) -> u128 {
                    fn get(self: @ContractState) -> u128 {
//                        42
                        42
//                    }
                    }
//                }
                }
//            }
            }
//        };
        };
//        program
        program
//    }
    }
//

//    #[rstest]
    #[rstest]
//    fn test_invoke_dynamic(program: Program) {
    fn test_invoke_dynamic(program: Program) {
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let module = native_context
        let module = native_context
//            .compile(&program, None)
            .compile(&program, None)
//            .expect("failed to compile context");
            .expect("failed to compile context");
//        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
//

//        // The first function in the program is `run_test`.
        // The first function in the program is `run_test`.
//        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;
        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;
//

//        let result = executor
        let result = executor
//            .invoke_dynamic(entrypoint_function_id, &[], Some(u128::MAX))
            .invoke_dynamic(entrypoint_function_id, &[], Some(u128::MAX))
//            .unwrap();
            .unwrap();
//

//        assert_eq!(result.return_value, JitValue::Felt252(Felt::from(42)));
        assert_eq!(result.return_value, JitValue::Felt252(Felt::from(42)));
//    }
    }
//

//    #[rstest]
    #[rstest]
//    fn test_invoke_dynamic_with_syscall_handler(program: Program) {
    fn test_invoke_dynamic_with_syscall_handler(program: Program) {
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let module = native_context
        let module = native_context
//            .compile(&program, None)
            .compile(&program, None)
//            .expect("failed to compile context");
            .expect("failed to compile context");
//        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
//

//        // The second function in the program is `get_block_hash`.
        // The second function in the program is `get_block_hash`.
//        let entrypoint_function_id = &program.funcs.get(1).expect("should have a function").id;
        let entrypoint_function_id = &program.funcs.get(1).expect("should have a function").id;
//

//        let mut syscall_handler = TestSyscallHandler;
        let mut syscall_handler = TestSyscallHandler;
//        let result = executor
        let result = executor
//            .invoke_dynamic_with_syscall_handler(
            .invoke_dynamic_with_syscall_handler(
//                entrypoint_function_id,
                entrypoint_function_id,
//                &[],
                &[],
//                Some(u128::MAX),
                Some(u128::MAX),
//                syscall_handler.clone(),
                syscall_handler.clone(),
//            )
            )
//            .unwrap();
            .unwrap();
//

//        let expected_value = JitValue::Enum {
        let expected_value = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: JitValue::Struct {
            value: JitValue::Struct {
//                fields: vec![JitValue::Felt252(
                fields: vec![JitValue::Felt252(
//                    syscall_handler.get_block_hash(1, &mut 0).unwrap(),
                    syscall_handler.get_block_hash(1, &mut 0).unwrap(),
//                )],
                )],
//                debug_name: Some("Tuple<felt252>".into()),
                debug_name: Some("Tuple<felt252>".into()),
//            }
            }
//            .into(),
            .into(),
//            debug_name: Some("core::panics::PanicResult::<(core::felt252,)>".into()),
            debug_name: Some("core::panics::PanicResult::<(core::felt252,)>".into()),
//        };
        };
//        assert_eq!(result.return_value, expected_value);
        assert_eq!(result.return_value, expected_value);
//    }
    }
//

//    #[rstest]
    #[rstest]
//    fn test_invoke_contract_dynamic(starknet_program: Program) {
    fn test_invoke_contract_dynamic(starknet_program: Program) {
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let module = native_context
        let module = native_context
//            .compile(&starknet_program, None)
            .compile(&starknet_program, None)
//            .expect("failed to compile context");
            .expect("failed to compile context");
//        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());
//

//        // The last function in the program is the `get` wrapper function.
        // The last function in the program is the `get` wrapper function.
//        let entrypoint_function_id = &starknet_program
        let entrypoint_function_id = &starknet_program
//            .funcs
            .funcs
//            .last()
            .last()
//            .expect("should have a function")
            .expect("should have a function")
//            .id;
            .id;
//

//        let result = executor
        let result = executor
//            .invoke_contract_dynamic(
            .invoke_contract_dynamic(
//                entrypoint_function_id,
                entrypoint_function_id,
//                &[],
                &[],
//                Some(u128::MAX),
                Some(u128::MAX),
//                TestSyscallHandler,
                TestSyscallHandler,
//            )
            )
//            .unwrap();
            .unwrap();
//

//        assert_eq!(result.return_values, vec![Felt::from(42)]);
        assert_eq!(result.return_values, vec![Felt::from(42)]);
//    }
    }
//}
}
