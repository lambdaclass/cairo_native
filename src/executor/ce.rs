use std::{
    ffi::c_void,
    path::{Path, PathBuf},
};

use cairo_lang_sierra::ids::FunctionId;
use educe::Educe;
use libloading::Library;
use starknet_types_core::felt::Felt;
use tempfile::NamedTempFile;

use crate::{
    error::Error, execution_result::ContractExecutionResult, module::NativeModule,
    starknet::StarknetSyscallHandler, utils::generate_function_name, values::JitValue, OptLevel,
};

#[derive(Educe)]
#[educe(Debug)]
pub struct ContractExecutor {
    #[educe(Debug(ignore))]
    library: Library,
    path: PathBuf,
}

unsafe impl Send for ContractExecutor {}
unsafe impl Sync for ContractExecutor {}

impl ContractExecutor {
    /// Create the executor from a native module with the given optimization level.
    /// You can save the library on the desired location later using `save`
    pub fn new(module: NativeModule, opt_level: OptLevel) -> Self {
        let NativeModule {
            module,
            registry,
            mut metadata,
        } = module;

        let library_path = NamedTempFile::new()
            .unwrap()
            .into_temp_path()
            .keep()
            .expect("can only fail on windows");

        let object_data = crate::module_to_object(&module, opt_level).unwrap();
        crate::object_to_shared_lib(&object_data, &library_path).unwrap();

        Self {
            library: unsafe { Library::new(&library_path).unwrap() },
            path: library_path,
        }
    }

    pub fn save(&self, to: &Path) -> std::io::Result<u64> {
        std::fs::copy(&self.path, to)
    }

    /// Load the executor from an already compiled library.
    pub fn load(library_path: &Path) -> Self {
        Self {
            library: unsafe { Library::new(library_path).unwrap() },
            path: library_path.to_path_buf(),
        }
    }

    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        todo!()
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
}
