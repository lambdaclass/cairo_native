use crate::{
    execution_result::ExecutionResult, metadata::gas::GasMetadata, types::TypeBuilder,
    utils::generate_function_name, values::JitValue,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericLibfunc, GenericType,
    },
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use libloading::Library;

pub struct AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    library: Library,
    registry: ProgramRegistry<TType, TLibfunc>,
    gas_metadata: Option<GasMetadata>,
}

impl<TType, TLibfunc> AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
{
    pub fn new(
        library: Library,
        registry: ProgramRegistry<TType, TLibfunc>,
        gas_metadata: Option<GasMetadata>,
    ) -> Self {
        Self {
            library,
            registry,
            gas_metadata,
        }
    }
}

impl AotNativeExecutor<CoreType, CoreLibfunc> {
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        mut gas: Option<u128>,
    ) -> ExecutionResult {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        let function_ptr = unsafe {
            self.library
                .get::<extern "C" fn()>(function_name.as_bytes())
                .unwrap()
        };

        if let (Some(gas), Some(required_init_gas)) = (
            gas.as_mut(),
            self.gas_metadata
                .as_ref()
                .and_then(|gas_metadata| gas_metadata.get_initial_required_gas(function_id)),
        ) {
            if required_init_gas > *gas {
                panic!("Not enough gas");
            }

            *gas -= required_init_gas;
        }

        super::invoke_dynamic(
            &self.registry,
            unsafe { function_ptr.into_raw().into_raw() },
            &self.registry.get_function(function_id).unwrap().signature,
            args,
            gas,
        )
    }
}
