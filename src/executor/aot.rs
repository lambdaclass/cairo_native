use crate::{
    error::Error,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{gas::GasMetadata, MetadataStorage},
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
use melior::{ir::Module, Context};
use starknet_types_core::felt::Felt;
use std::cell::RefCell;
use tempfile::NamedTempFile;

#[derive(Educe)]
#[educe(Debug)]
pub struct AotNativeExecutor<'c> {
    context: &'c Context,

    module: Module<'c>,
    metadata: RefCell<MetadataStorage>,

    #[educe(Debug(ignore))]
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
    #[educe(Debug(ignore))]
    library: Library,

    gas_metadata: GasMetadata,
}

impl<'c> AotNativeExecutor<'c> {
    /// Utility to convert a [`NativeModule`] into an [`AotNativeExecutor`].
    pub fn from_native_module(
        context: &'c Context,
        module: NativeModule<'c>,
        opt_level: OptLevel,
    ) -> Self {
        let NativeModule {
            module,
            registry,
            mut metadata,
        } = module;

        let library_path = NamedTempFile::new().unwrap().into_temp_path();

        let object_data = crate::module_to_object(&module, opt_level).unwrap();
        crate::object_to_shared_lib(&object_data, &library_path).unwrap();

        let gas_metadata = metadata.remove().unwrap();
        Self {
            context,
            module,
            metadata: RefCell::new(metadata),
            registry,
            library: unsafe { Library::new(library_path).unwrap() },
            gas_metadata,
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
            .map_err(|_| crate::error::Error::InsufficientGasError)?;

        Ok(super::invoke_dynamic(
            self.context,
            &self.registry,
            &self.module,
            &mut self.metadata.borrow_mut(),
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            available_gas,
            Option::<DummySyscallHandler>::None,
        ))
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
            .map_err(|_| crate::error::Error::InsufficientGasError)?;

        Ok(super::invoke_dynamic(
            self.context,
            &self.registry,
            &self.module,
            &mut self.metadata.borrow_mut(),
            self.find_function_ptr(function_id),
            self.extract_signature(function_id),
            args,
            available_gas,
            Some(syscall_handler),
        ))
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
            .map_err(|_| crate::error::Error::InsufficientGasError)?;

        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            self.context,
            &self.registry,
            &self.module,
            &mut self.metadata.borrow_mut(),
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
        ))
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
