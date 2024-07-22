use crate::{
    error::Error,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::gas::GasMetadata,
    module::NativeModule,
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    utils::generate_function_name,
    values::JitValue,
    OptLevel,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use educe::Educe;
use libc::c_void;
use libloading::Library;
use starknet_types_core::felt::Felt;
use tempfile::NamedTempFile;

#[derive(Educe)]
#[educe(Debug)]
pub struct AotNativeExecutor {
    #[educe(Debug(ignore))]
    library: Library,
    #[educe(Debug(ignore))]
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    gas_metadata: GasMetadata,
}

unsafe impl Send for AotNativeExecutor {}
unsafe impl Sync for AotNativeExecutor {}

impl AotNativeExecutor {
    pub fn new(
        library: Library,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        gas_metadata: GasMetadata,
    ) -> Self {
        Self {
            library,
            registry,
            gas_metadata,
        }
    }

    /// Utility to convert a [`NativeModule`] into an [`AotNativeExecutor`].
    pub fn from_native_module(module: NativeModule, opt_level: OptLevel) -> Self {
        let NativeModule {
            module,
            registry,
            mut metadata,
        } = module;

        let library_path = NamedTempFile::new().unwrap().into_temp_path();

        let object_data = crate::module_to_object(&module, opt_level).unwrap();
        crate::object_to_shared_lib(&object_data, &library_path).unwrap();

        Self {
            library: unsafe { Library::new(library_path).unwrap() },
            registry,
            gas_metadata: metadata.remove().unwrap(),
        }
    }

    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            available_gas,
            Option::<DummySyscallHandler>::None,
        )
    }

    pub fn invoke_dynamic_with_syscall_handler(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
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
            self.extract_signature(function_id),
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
            available_gas,
            Some(syscall_handler),
        )?)
    }

    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::NativeContext,
        starknet_stub::StubSyscallHandler,
        utils::test::{load_cairo, load_starknet},
    };
    use cairo_lang_sierra::program::Program;
    use rstest::*;

    #[fixture]
    fn program() -> Program {
        let (_, program) = load_cairo! {
            use core::starknet::{SyscallResultTrait, get_block_hash_syscall};

            fn run_test() -> felt252 {
                42
            }

            fn get_block_hash() -> felt252 {
                get_block_hash_syscall(1).unwrap_syscall()
            }
        };
        program
    }

    #[fixture]
    fn starknet_program() -> Program {
        let (_, program) = load_starknet! {
            #[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn get(self: @TContractState) -> u128;
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn get(self: @ContractState) -> u128 {
                        42
                    }
                }
            }
        };
        program
    }

    #[rstest]
    fn test_invoke_dynamic(program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program, None)
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());

        // The first function in the program is `run_test`.
        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;

        let result = executor
            .invoke_dynamic(entrypoint_function_id, &[], Some(u128::MAX))
            .unwrap();

        assert_eq!(result.return_value, JitValue::Felt252(Felt::from(42)));
    }

    #[rstest]
    fn test_invoke_dynamic_with_syscall_handler(program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program, None)
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());

        // The second function in the program is `get_block_hash`.
        let entrypoint_function_id = &program.funcs.get(1).expect("should have a function").id;

        let mut syscall_handler = &mut StubSyscallHandler::default();

        let expected_value = syscall_handler.get_block_hash(1, &mut 0).unwrap();

        let result = executor
            .invoke_dynamic_with_syscall_handler(
                entrypoint_function_id,
                &[],
                Some(u128::MAX),
                syscall_handler,
            )
            .unwrap();

        let expected_value = JitValue::Enum {
            tag: 0,
            value: JitValue::Struct {
                fields: vec![JitValue::Felt252(expected_value)],
                debug_name: Some("Tuple<felt252>".into()),
            }
            .into(),
            debug_name: Some("core::panics::PanicResult::<(core::felt252,)>".into()),
        };
        assert_eq!(result.return_value, expected_value);
    }

    #[rstest]
    fn test_invoke_contract_dynamic(starknet_program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&starknet_program, None)
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = executor
            .invoke_contract_dynamic(
                entrypoint_function_id,
                &[],
                Some(u128::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![Felt::from(42)]);
    }
}
