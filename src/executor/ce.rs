use std::{
    alloc::Layout,
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    path::{Path, PathBuf},
};

use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::core::CoreTypeConcrete,
    ids::FunctionId,
    program::{FunctionSignature, Program},
};
use educe::Educe;
use libloading::Library;
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use tempfile::NamedTempFile;

use crate::{
    arch::AbiArgument,
    error::Error,
    execution_result::ContractExecutionResult,
    executor::invoke_trampoline,
    module::NativeModule,
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    types::TypeBuilder,
    utils::{generate_function_name, get_integer_layout},
    OptLevel,
};

#[derive(Educe)]
#[educe(Debug)]
pub struct ContractExecutor {
    #[educe(Debug(ignore))]
    library: Library,
    path: PathBuf,
    entry_points_info: BTreeMap<u64, EntryPointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntryPointInfo {
    pub num_builtins: u64,
}

unsafe impl Send for ContractExecutor {}
unsafe impl Sync for ContractExecutor {}

impl Clone for ContractExecutor {
    fn clone(&self) -> Self {
        let mut x = Self::load(&self.path);
        x.entry_points_info = self.entry_points_info.clone();
        x
    }
}

impl ContractExecutor {
    /// Create the executor from a native module with the given optimization level.
    /// You can save the library on the desired location later using `save`
    pub fn new(module: NativeModule, opt_level: OptLevel, p: &Program) -> Result<Self, Error> {
        let NativeModule {
            module,
            registry,
            metadata: _,
        } = module;

        let mut infos = BTreeMap::new();

        for x in &p.funcs {
            let mut num_builtins = 0;

            for p in &x.params {
                let ty = registry.get_type(&p.ty)?;
                if ty.is_builtin() && !matches!(ty, CoreTypeConcrete::GasBuiltin(_)) {
                    num_builtins += 1;
                } else {
                    break;
                }
            }

            infos.insert(x.id.id, EntryPointInfo { num_builtins });
        }

        let library_path = NamedTempFile::new()
            .unwrap()
            .into_temp_path()
            .keep()
            .expect("can only fail on windows");

        let object_data = crate::module_to_object(&module, opt_level).unwrap();
        crate::object_to_shared_lib(&object_data, &library_path).unwrap();

        Ok(Self {
            library: unsafe { Library::new(&library_path).unwrap() },
            path: library_path,
            entry_points_info: infos,
        })
    }

    pub fn save(&self, to: &Path) -> std::io::Result<u64> {
        std::fs::copy(&self.path, to)
    }

    /// Load the executor from an already compiled library.
    pub fn load(library_path: &Path) -> Self {
        Self {
            library: unsafe { Library::new(library_path).unwrap() },
            path: library_path.to_path_buf(),
            entry_points_info: todo!(),
        }
    }

    /// Runs the given entry point.
    pub fn run(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        mut syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        let arena = Bump::new();
        let mut invoke_data = Vec::<u8>::new();

        let function_ptr = self.find_function_ptr(function_id);

        //  it can vary from contract to contract thats why we need to store/ load it.
        let num_builtins = self.entry_points_info[&function_id.id].num_builtins;

        for _ in 0..num_builtins {
            0u64.to_bytes(&mut invoke_data)?;
        }

        let gas = gas.unwrap_or(0);
        gas.to_bytes(&mut invoke_data)?;

        let mut syscall_handler = StarknetSyscallHandlerCallbacks::new(&mut syscall_handler);

        (&mut syscall_handler as *mut StarknetSyscallHandlerCallbacks<_>)
            .to_bytes(&mut invoke_data)?;

        let felt_layout = get_integer_layout(252).pad_to_align();
        let ptr: *mut () = unsafe { libc::malloc(felt_layout.size() * args.len()).cast() };
        let len: u32 = args.len().try_into().unwrap();

        ptr.to_bytes(&mut invoke_data)?;
        0u32.to_bytes(&mut invoke_data)?; // start
        len.to_bytes(&mut invoke_data)?; // end
        len.to_bytes(&mut invoke_data)?; // cap

        for (idx, elem) in args.iter().enumerate() {
            let f = elem.to_bytes_le();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f.as_ptr().cast::<u8>(),
                    ptr.byte_add(idx * felt_layout.size()).cast::<u8>(),
                    felt_layout.size(),
                )
            };
        }

        // Pad invoke data to the 16 byte boundary avoid segfaults.
        #[cfg(target_arch = "aarch64")]
        const REGISTER_BYTES: usize = 64;
        #[cfg(target_arch = "x86_64")]
        const REGISTER_BYTES: usize = 48;
        if invoke_data.len() > REGISTER_BYTES {
            invoke_data.resize(
                REGISTER_BYTES + (invoke_data.len() - REGISTER_BYTES).next_multiple_of(16),
                0,
            );
        }

        // Invoke the trampoline.
        #[cfg(target_arch = "x86_64")]
        let mut ret_registers = [0; 2];
        #[cfg(target_arch = "aarch64")]
        let mut ret_registers = [0; 4];

        unsafe {
            invoke_trampoline(
                function_ptr,
                invoke_data.as_ptr().cast(),
                invoke_data.len() >> 3,
                ret_registers.as_mut_ptr(),
            );
        }

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
