//! A specialized executor for Starknet contracts, avoiding the overhead of storing the sierra program registry and
//! enabling efficient serialization of the program/data once compiled.
//!
//! This executor heavily relies on the stability of the contract entry point argument order.
//!
//! Right now this order is like the following:
//!
//! 1. One or more builtins (rangecheck, etc)
//! 2. Gas builtin
//! 3. System builtin (the syscall handler)
//! 4. A Array of felts (calldata)
//!
//! ## How it works:
//!
//! The only variable data we need to know at call time is the builtins order,
//! to save this, when first compiling the sierra program (with [`ContractExecutor::new`]) it iterates through all the user
//! defined functions (this includes the contract wrappers, which are the ones that matter)
//! and saves the builtin arguments in a `Vec`.
//!
//! The API provides two more methods: [`ContractExecutor::save`] and [`ContractExecutor::load`].
//!
//! Save can be used to save the compiled program into the given path, alongside it will be saved
//! a json file with the entry points and their builtins (as seen in the example)
//!
//! ```json
//! {"0":{"builtins":[]},"1":{"builtins":["RangeCheck","Gas","System"]}}
//! ```
//!
//! If the given path is "program.so", then at the same location, "program.json" will be saved.
//!
//! When loading, passing the "program.so" path will make it load the program and the "program.json" alongside it.
//!

use crate::{
    arch::AbiArgument,
    context::NativeContext,
    error::{panic::ToNativeAssertError, Error, Result},
    execution_result::{BuiltinStats, ContractExecutionResult},
    executor::invoke_trampoline,
    metadata::{gas::MetadataComputationConfig, runtime_bindings::setup_runtime},
    module::NativeModule,
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    types::TypeBuilder,
    utils::{
        decode_error_message, generate_function_name, get_integer_layout, libc_free, libc_malloc,
        BuiltinCosts,
    },
    OptLevel,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete, core::CoreTypeConcrete, gas::CostTokenType,
        starknet::StarkNetTypeConcrete,
    },
    ids::FunctionId,
    program::Program,
};
use cairo_lang_starknet_classes::contract_class::ContractEntryPoints;
use cairo_lang_starknet_classes::{
    casm_contract_class::ENTRY_POINT_COST, compiler_version::VersionId,
};
use educe::Educe;
use itertools::chain;
use libloading::Library;
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    cmp::Ordering,
    collections::BTreeMap,
    ffi::c_void,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};
use tempfile::NamedTempFile;

/// Please look at the [module level docs](self).
#[derive(Educe, Clone)]
#[educe(Debug)]
pub struct AotContractExecutor {
    #[educe(Debug(ignore))]
    library: Arc<Library>,
    path: PathBuf,
    contract_info: NativeContractInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NativeContractInfo {
    pub version: ContractInfoVersion,
    pub entry_points: BTreeMap<Felt, EntryPointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContractInfoVersion {
    V0,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntryPointInfo {
    pub function_id: u64,
    pub builtins: Vec<BuiltinType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuiltinType {
    Bitwise,
    EcOp,
    RangeCheck,
    SegmentArena,
    Poseidon,
    Pedersen,
    RangeCheck96,
    CircuitAdd,
    CircuitMul,
    Gas,
    System,
    BuiltinCosts,
}

impl BuiltinType {
    pub const fn size_in_bytes(&self) -> usize {
        size_of::<u64>()
    }
}

impl AotContractExecutor {
    /// Compile and load a program using a temporary shared library.
    pub fn new(
        program: &Program,
        entry_points: &ContractEntryPoints,
        sierra_version: VersionId,
        opt_level: OptLevel,
    ) -> Result<Self> {
        let output_path = NamedTempFile::new()?
            .into_temp_path()
            .keep()
            .to_native_assert_error("can only fail on windows")?;

        let executor = Self::new_into(
            program,
            entry_points,
            sierra_version,
            output_path,
            opt_level,
        )?
        .to_native_assert_error("temporary contract path collision")?;

        fs::remove_file(&executor.path)?;
        fs::remove_file(executor.path.with_extension("json"))?;
        Ok(executor)
    }

    /// Compile and load a program into a shared library.
    ///
    /// This function uses a lockfile to support cache sharing between multiple processes. An
    /// attempt to compile a program while the `output_path` is already locked will result in
    /// `Ok(None)` being returned. When this happens, the user should wait until the lock is
    /// released, at which point they can use `AotContractExecutor::from_path` to load it.
    pub fn new_into(
        program: &Program,
        entry_points: &ContractEntryPoints,
        sierra_version: VersionId,
        output_path: impl Into<PathBuf>,
        opt_level: OptLevel,
    ) -> Result<Option<Self>> {
        let output_path = output_path.into();
        let lock_file = match LockFile::new(&output_path)? {
            Some(x) => x,
            None => return Ok(None),
        };

        let context = NativeContext::new();

        let no_eq_solver = match sierra_version.major.cmp(&1) {
            Ordering::Less => false,
            Ordering::Equal => sierra_version.minor >= 4,
            Ordering::Greater => true,
        };

        // Compile the Sierra program.
        let NativeModule {
            module, registry, ..
        } = context.compile(
            program,
            true,
            Some(MetadataComputationConfig {
                function_set_costs: chain!(
                    entry_points.constructor.iter(),
                    entry_points.external.iter(),
                    entry_points.l1_handler.iter(),
                )
                .map(|x| {
                    (
                        FunctionId::new(x.function_idx as u64),
                        [(CostTokenType::Const, ENTRY_POINT_COST)].into(),
                    )
                })
                .collect(),
                linear_gas_solver: no_eq_solver,
                linear_ap_change_solver: no_eq_solver,
            }),
        )?;

        // Generate mappings between the entry point's selectors and their function indexes.
        let entry_point_mappings = chain!(
            entry_points.constructor.iter(),
            entry_points.external.iter(),
            entry_points.l1_handler.iter(),
        )
        .map(|x| {
            let function_id = x.function_idx as u64;
            let function = registry
                .get_function(&FunctionId::new(function_id))
                .to_native_assert_error("unreachable")?;

            let builtins = function
                .params
                .iter()
                .map(|x| registry.get_type(&x.ty).unwrap())
                .take_while(|ty| ty.is_builtin())
                .filter(|ty| !ty.is_zst(&registry).unwrap())
                .map(|ty| match ty {
                    CoreTypeConcrete::Bitwise(_) => BuiltinType::Bitwise,
                    CoreTypeConcrete::EcOp(_) => BuiltinType::EcOp,
                    CoreTypeConcrete::RangeCheck(_) => BuiltinType::RangeCheck,
                    CoreTypeConcrete::Pedersen(_) => BuiltinType::Pedersen,
                    CoreTypeConcrete::Poseidon(_) => BuiltinType::Poseidon,
                    CoreTypeConcrete::BuiltinCosts(_) => BuiltinType::BuiltinCosts,
                    CoreTypeConcrete::SegmentArena(_) => BuiltinType::SegmentArena,
                    CoreTypeConcrete::RangeCheck96(_) => BuiltinType::RangeCheck96,
                    CoreTypeConcrete::Circuit(CircuitTypeConcrete::AddMod(_)) => {
                        BuiltinType::CircuitAdd
                    }
                    CoreTypeConcrete::Circuit(CircuitTypeConcrete::MulMod(_)) => {
                        BuiltinType::CircuitMul
                    }
                    CoreTypeConcrete::GasBuiltin(_) => BuiltinType::Gas,
                    CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
                        BuiltinType::System
                    }
                    _ => unreachable!("not a builtin"),
                })
                .collect();

            Ok((
                Felt::from(&x.selector),
                EntryPointInfo {
                    function_id: x.function_idx as u64,
                    builtins,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;

        let object_data = crate::module_to_object(&module, opt_level)?;

        // Build the shared library into the lockfile, to avoid using a tmp file.
        crate::object_to_shared_lib(&object_data, &lock_file.0)?;

        // Write the contract info.
        fs::write(
            output_path.with_extension("json"),
            serde_json::to_string(&NativeContractInfo {
                version: ContractInfoVersion::V0,
                entry_points: entry_point_mappings,
            })?,
        )?;

        // Atomically move the built shared library to the correct path. This will avoid data races
        // when loading contracts.
        lock_file.rename(&output_path)?;

        Self::from_path(output_path)
    }

    /// Load a program from a shared library.
    ///
    /// This function will check for the existence of a lockfile. If found, it'll return `Ok(None)`.
    /// When this happens, the user should wait until the lock is released, then try loading it
    /// again.
    pub fn from_path(path: impl Into<PathBuf>) -> Result<Option<Self>> {
        let path = path.into();

        // Note: Library should load first, otherwise there could theoretically be a race condition.
        //   See the `new_into` function's code for details.
        let library = Arc::new(unsafe { Library::new(&path)? });
        let contract_info =
            serde_json::from_str(&fs::read_to_string(path.with_extension("json"))?)?;

        let executor = Self {
            library,
            path,
            contract_info,
        };

        setup_runtime(|x| executor.find_symbol_ptr(x));

        Ok(Some(executor))
    }

    /// Runs the entry point by the given selector.
    ///
    /// - selector: The selector of the entry point to run.
    /// - args: The calldata.
    /// - gas: The gas for the execution.
    /// - builtin_costs: An optional argument to customize the costs of the builtins.
    /// - syscall_handler: The syscall handler implementation to use when executing the contract.
    ///
    /// The entry point gas cost is not deducted from the gas counter.
    pub fn run(
        &self,
        selector: Felt,
        args: &[Felt],
        gas: u64,
        builtin_costs: Option<BuiltinCosts>,
        mut syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult> {
        let arena = Bump::new();
        let mut invoke_data = Vec::<u8>::new();

        let function_id = FunctionId {
            id: self
                .contract_info
                .entry_points
                .get(&selector)
                .ok_or(Error::SelectorNotFound)?
                .function_id,
            debug_name: None,
        };
        let function_ptr = self.find_function_ptr(&function_id, true)?;

        let builtin_costs: [u64; 7] = builtin_costs.unwrap_or_default().into();

        // We may be inside a recursive contract, save the possible saved builtin costs to restore it after our call.
        let old_builtincosts_ptr =
            crate::runtime::cairo_native__set_costs_builtin(builtin_costs.as_ptr());

        //  it can vary from contract to contract thats why we need to store/ load it.
        let builtins_size: usize = self.contract_info.entry_points[&selector]
            .builtins
            .iter()
            .map(|x| x.size_in_bytes())
            .sum();

        // There is always a return ptr because contracts always return more than 1 thing (builtin counters, syscall, enum)
        let return_ptr = arena.alloc_layout(unsafe {
            // 56 = size of enum
            Layout::from_size_align_unchecked(128 + builtins_size, 16)
        });

        return_ptr
            .as_ptr()
            .to_bytes(&mut invoke_data, |_| unreachable!())?;

        let mut syscall_handler = StarknetSyscallHandlerCallbacks::new(&mut syscall_handler);

        for b in &self.contract_info.entry_points[&selector].builtins {
            match b {
                BuiltinType::Gas => {
                    gas.to_bytes(&mut invoke_data, |_| unreachable!())?;
                }
                BuiltinType::BuiltinCosts => {
                    // todo: check if valid
                    builtin_costs
                        .as_ptr()
                        .to_bytes(&mut invoke_data, |_| unreachable!())?;
                }
                BuiltinType::System => {
                    (&mut syscall_handler as *mut StarknetSyscallHandlerCallbacks<_>)
                        .to_bytes(&mut invoke_data, |_| unreachable!())?;
                }
                _ => {
                    0u64.to_bytes(&mut invoke_data, |_| unreachable!())?;
                }
            }
        }

        let felt_layout = get_integer_layout(252).pad_to_align();
        let refcount_offset = crate::types::array::calc_data_prefix_offset(felt_layout);

        let len_u32: u32 = args
            .len()
            .try_into()
            .to_native_assert_error("number of arguments should fit into a u32")?;
        let array_ptr = match args.len() {
            0 => std::ptr::null_mut(),
            _ => unsafe {
                let array_ptr: *mut () =
                    libc_malloc(felt_layout.size() * args.len() + refcount_offset).cast();

                // Write reference count.
                array_ptr.cast::<(u32, u32)>().write((1, len_u32));
                array_ptr.byte_add(refcount_offset)
            },
        };

        for (idx, elem) in args.iter().enumerate() {
            let f = elem.to_bytes_le();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f.as_ptr().cast::<u8>(),
                    array_ptr.byte_add(idx * felt_layout.size()).cast::<u8>(),
                    felt_layout.size(),
                )
            };
        }

        // Make double pointer.
        let array_ptr_ptr = unsafe {
            let array_ptr_ptr = libc_malloc(size_of::<*mut ()>()).cast::<*mut ()>();
            array_ptr_ptr.write(array_ptr);
            array_ptr_ptr
        };

        array_ptr_ptr.to_bytes(&mut invoke_data, |_| unreachable!())?;
        if cfg!(target_arch = "aarch64") {
            0u32.to_bytes(&mut invoke_data, |_| unreachable!())?; // start
            len_u32.to_bytes(&mut invoke_data, |_| unreachable!())?; // end
            len_u32.to_bytes(&mut invoke_data, |_| unreachable!())?; // cap
        } else if cfg!(target_arch = "x86_64") {
            (0u32 as u64).to_bytes(&mut invoke_data, |_| unreachable!())?; // start
            (len_u32 as u64).to_bytes(&mut invoke_data, |_| unreachable!())?; // end
            (len_u32 as u64).to_bytes(&mut invoke_data, |_| unreachable!())?; // cap
        } else {
            unreachable!("unsupported architecture");
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

        // Parse final gas.
        unsafe fn read_value<T>(ptr: &mut NonNull<()>) -> &T {
            let align_offset = ptr
                .cast::<u8>()
                .as_ptr()
                .align_offset(std::mem::align_of::<T>());
            let value_ptr = ptr.cast::<u8>().as_ptr().add(align_offset).cast::<T>();

            *ptr = NonNull::new_unchecked(value_ptr.add(1)).cast();
            &*value_ptr
        }

        let mut remaining_gas = 0;
        let mut builtin_stats = BuiltinStats::default();

        let return_ptr = &mut return_ptr.cast();

        for b in &self.contract_info.entry_points[&selector].builtins {
            match b {
                BuiltinType::Gas => {
                    remaining_gas = unsafe { *read_value::<u64>(return_ptr) };
                }
                BuiltinType::System => {
                    unsafe { read_value::<*mut ()>(return_ptr) };
                }
                BuiltinType::BuiltinCosts => {
                    unsafe { read_value::<*mut ()>(return_ptr) };
                    // ptr holds the builtin costs, but they dont change, so its of no use, but we read to advance the ptr.
                }
                x => {
                    let value = unsafe { *read_value::<u64>(return_ptr) } as usize;

                    match x {
                        BuiltinType::Bitwise => builtin_stats.bitwise = value,
                        BuiltinType::EcOp => builtin_stats.ec_op = value,
                        BuiltinType::RangeCheck => builtin_stats.range_check = value,
                        BuiltinType::SegmentArena => builtin_stats.segment_arena = value,
                        BuiltinType::Poseidon => builtin_stats.poseidon = value,
                        BuiltinType::Pedersen => builtin_stats.pedersen = value,
                        BuiltinType::RangeCheck96 => builtin_stats.range_check_96 = value,
                        BuiltinType::CircuitAdd => builtin_stats.circuit_add = value,
                        BuiltinType::CircuitMul => builtin_stats.circuit_mul = value,
                        BuiltinType::Gas => {}
                        BuiltinType::System => {}
                        BuiltinType::BuiltinCosts => {}
                    }
                }
            }
        }

        // align the pointer
        // layout of the enum type.
        let layout = unsafe { Layout::from_size_align_unchecked(32, 8) };
        let align_offset = return_ptr
            .cast::<u8>()
            .as_ptr()
            .align_offset(layout.align());

        let tag_layout = Layout::from_size_align(1, 1)?;
        let enum_ptr = unsafe {
            NonNull::new(return_ptr.cast::<u8>().as_ptr().add(align_offset))
                .to_native_assert_error("return ptr should not be null")?
        };

        let tag = *unsafe { enum_ptr.cast::<u8>().as_ref() } as usize;
        let tag = tag & 0x01; // Filter out bits that are not part of the enum's tag.

        let value_layout = unsafe { Layout::from_size_align_unchecked(24, 8) };
        let mut value_ptr = unsafe { enum_ptr.byte_add(tag_layout.extend(value_layout)?.1).cast() };

        let array_ptr_ptr = unsafe { *read_value::<*mut NonNull<()>>(&mut value_ptr) };
        let array_start = unsafe { *read_value::<u32>(&mut value_ptr) };
        let array_end = unsafe { *read_value::<u32>(&mut value_ptr) };
        let _array_capacity = unsafe { *read_value::<u32>(&mut value_ptr) };

        let mut array_value = Vec::with_capacity((array_end - array_start) as usize);
        if !array_ptr_ptr.is_null() {
            let array_ptr = unsafe { dbg!(array_ptr_ptr).read() };
            dbg!(array_ptr);

            let elem_stride = felt_layout.pad_to_align().size();
            for i in array_start..array_end {
                let cur_elem_ptr = unsafe { array_ptr.byte_add(elem_stride * i as usize) };

                let mut data = unsafe { cur_elem_ptr.cast::<[u8; 32]>().read() };
                data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

                array_value.push(Felt::from_bytes_le(&data));
            }

            unsafe {
                let array_ptr = array_ptr.byte_sub(refcount_offset);
                assert_eq!(array_ptr.cast::<u32>().read(), 1);
                libc_free(array_ptr.as_ptr().cast());
                libc_free(array_ptr_ptr.cast());
            }
        }

        let error_msg = match tag {
            0 => None,
            _ => {
                Some(decode_error_message(
                    &array_value
                        .iter()
                        .flat_map(|felt| felt.to_bytes_be().to_vec())
                        // remove null chars
                        .filter(|b| *b != 0)
                        .collect::<Vec<_>>(),
                ))
            }
        };

        // Restore the original builtin costs pointer.
        crate::runtime::cairo_native__set_costs_builtin(old_builtincosts_ptr);

        #[cfg(feature = "with-mem-tracing")]
        crate::utils::mem_tracing::report_stats();

        Ok(ContractExecutionResult {
            remaining_gas,
            failure_flag: tag != 0,
            return_values: array_value,
            error_msg,
        })
    }

    pub fn find_function_ptr(
        &self,
        function_id: &FunctionId,
        is_for_contract_executor: bool,
    ) -> Result<*mut c_void> {
        let function_name = generate_function_name(function_id, is_for_contract_executor);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        Ok(unsafe {
            self.library
                .get::<extern "C" fn()>(function_name.as_bytes())?
                .into_raw()
                .into_raw()
        })
    }

    pub fn find_symbol_ptr(&self, name: &str) -> Option<*mut c_void> {
        unsafe {
            self.library
                .get::<*mut ()>(name.as_bytes())
                .ok()
                .map(|x| x.into_raw().into_raw())
        }
    }
}

#[derive(Debug)]
struct LockFile(PathBuf);

impl LockFile {
    pub fn new(path: impl Into<PathBuf>) -> io::Result<Option<Self>> {
        let path: PathBuf = path.into();
        let path = path.with_extension("lock");

        match File::create_new(&path) {
            Ok(_) => Ok(Some(Self(path))),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn rename(self, path: impl AsRef<Path>) -> io::Result<()> {
        fs::rename(&self.0, path.as_ref())?;

        // don't remove lockfile, as we just renamed it
        std::mem::forget(self);

        Ok(())
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{starknet_stub::StubSyscallHandler, utils::test::load_starknet_contract};
    use cairo_lang_starknet_classes::contract_class::{
        version_id_from_serialized_sierra_program, ContractClass,
    };
    use rayon::iter::ParallelBridge;
    use rstest::*;

    // todo add recursive contract test

    #[fixture]
    fn starknet_program() -> ContractClass {
        let (_, program) = load_starknet_contract! {
            #[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn get(self: @TContractState, x: felt252) -> (felt252, felt252);
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn get(self: @ContractState, x: felt252) -> (felt252, felt252) {
                        (x, x * 2)
                    }
                }
            }
        };
        program
    }

    #[fixture]
    fn starknet_program_factorial() -> ContractClass {
        let (_, program) = load_starknet_contract! {
            #[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn get(self: @TContractState, x: felt252) -> felt252;
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn get(self: @ContractState, x: felt252) -> felt252 {
                        factorial(1, x)
                    }
                }

                fn factorial(value: felt252, n: felt252) -> felt252 {
                    if (n == 1) {
                        value
                    } else {
                        factorial(value * n, n - 1)
                    }
                }
            }
        };
        program
    }

    #[fixture]
    fn starknet_program_empty() -> ContractClass {
        let (_, program) = load_starknet_contract! {
            #[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn call(self: @TContractState);
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn call(self: @ContractState) {
                    }
                }
            }
        };
        program
    }

    #[rstest]
    #[case(OptLevel::Default)]
    fn test_contract_executor_parallel(
        starknet_program: ContractClass,
        #[case] optlevel: OptLevel,
    ) {
        use rayon::iter::ParallelIterator;

        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&starknet_program.sierra_program).unwrap();
        let executor = Arc::new(
            AotContractExecutor::new(
                &starknet_program.extract_sierra_program().unwrap(),
                &starknet_program.entry_points_by_type,
                sierra_version,
                optlevel,
            )
            .unwrap(),
        );

        // The last function in the program is the `get` wrapper function.
        let selector = starknet_program
            .entry_points_by_type
            .external
            .last()
            .unwrap()
            .selector
            .clone();

        (0..200).par_bridge().for_each(|n| {
            let result = executor
                .run(
                    Felt::from(&selector),
                    &[n.into()],
                    u64::MAX,
                    None,
                    &mut StubSyscallHandler::default(),
                )
                .unwrap();
            assert_eq!(result.return_values, vec![Felt::from(n), Felt::from(n * 2)]);
            assert_eq!(result.remaining_gas, 18446744073709551615);
        });
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    fn test_contract_executor(starknet_program: ContractClass, #[case] optlevel: OptLevel) {
        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&starknet_program.sierra_program).unwrap();
        let executor = AotContractExecutor::new(
            &starknet_program.extract_sierra_program().unwrap(),
            &starknet_program.entry_points_by_type,
            sierra_version,
            optlevel,
        )
        .unwrap();

        // The last function in the program is the `get` wrapper function.
        let selector = starknet_program
            .entry_points_by_type
            .external
            .last()
            .unwrap()
            .selector
            .clone();

        let result = executor
            .run(
                Felt::from(&selector),
                &[2.into()],
                u64::MAX,
                None,
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![Felt::from(2), Felt::from(4)]);
    }

    #[rstest]
    #[case(OptLevel::Aggressive)]
    fn test_contract_executor_factorial(
        starknet_program_factorial: ContractClass,
        #[case] optlevel: OptLevel,
    ) {
        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&starknet_program_factorial.sierra_program)
                .unwrap();
        let executor = AotContractExecutor::new(
            &starknet_program_factorial.extract_sierra_program().unwrap(),
            &starknet_program_factorial.entry_points_by_type,
            sierra_version,
            optlevel,
        )
        .unwrap();

        // The last function in the program is the `get` wrapper function.
        let selector = starknet_program_factorial
            .entry_points_by_type
            .external
            .last()
            .unwrap()
            .selector
            .clone();

        let result = executor
            .run(
                Felt::from(&selector),
                &[10.into()],
                u64::MAX,
                None,
                &mut StubSyscallHandler::default(),
            )
            .unwrap();
        assert_eq!(result.return_values, vec![Felt::from(3628800)]);
        assert_eq!(result.remaining_gas, 18446744073709537615);
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    fn test_contract_executor_empty(
        starknet_program_empty: ContractClass,
        #[case] optlevel: OptLevel,
    ) {
        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&starknet_program_empty.sierra_program)
                .unwrap();
        let executor = AotContractExecutor::new(
            &starknet_program_empty.extract_sierra_program().unwrap(),
            &starknet_program_empty.entry_points_by_type,
            sierra_version,
            optlevel,
        )
        .unwrap();

        // The last function in the program is the `get` wrapper function.
        // The last function in the program is the `get` wrapper function.
        let selector = starknet_program_empty
            .entry_points_by_type
            .external
            .last()
            .unwrap()
            .selector
            .clone();

        let result = executor
            .run(
                Felt::from(&selector),
                &[],
                u64::MAX,
                None,
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![]);
    }
}
