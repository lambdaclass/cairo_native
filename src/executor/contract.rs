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
    error::Result,
    execution_result::{BuiltinStats, ContractExecutionResult},
    executor::invoke_trampoline,
    module::NativeModule,
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    types::TypeBuilder,
    utils::{decode_error_message, generate_function_name, get_integer_layout},
    OptLevel,
};
use bumpalo::Bump;
use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete, core::CoreTypeConcrete, gas::CostTokenType,
        starknet::StarkNetTypeConcrete, ConcreteType,
    },
    ids::FunctionId,
    program::Program,
};
use educe::Educe;
use libloading::Library;
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    collections::BTreeMap,
    ffi::c_void,
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
    is_temp_path: bool,
    entry_points_info: BTreeMap<u64, EntryPointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct EntryPointInfo {
    builtins: Vec<BuiltinType>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum BuiltinType {
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

impl AotContractExecutor {
    /// Create the executor from a sierra program with the given optimization level.
    /// You can save the library on the desired location later using `save`.
    /// If not saved, the path is treated as
    /// a temporary file an deleted when dropped.
    /// If you loaded a ContractExecutor using [`load`] then it will not be treated as a temp file.
    pub fn new(sierra_program: &Program, opt_level: OptLevel) -> Result<Self> {
        let native_context = NativeContext::new();
        let module = native_context.compile(sierra_program, true)?;

        let NativeModule {
            module,
            registry,
            metadata: _,
        } = module;

        let mut infos = BTreeMap::new();

        for x in &sierra_program.funcs {
            let mut builtins = Vec::new();

            for p in &x.params {
                let ty = registry.get_type(&p.ty)?;
                if ty.is_builtin() {
                    // Skip zero sized builtins
                    if ty.is_zst(&registry)? {
                        continue;
                    }

                    match ty {
                        CoreTypeConcrete::Bitwise(_) => builtins.push(BuiltinType::Bitwise),
                        CoreTypeConcrete::EcOp(_) => builtins.push(BuiltinType::EcOp),
                        CoreTypeConcrete::RangeCheck(_) => builtins.push(BuiltinType::RangeCheck),
                        CoreTypeConcrete::Pedersen(_) => builtins.push(BuiltinType::Pedersen),
                        CoreTypeConcrete::Poseidon(_) => builtins.push(BuiltinType::Poseidon),
                        CoreTypeConcrete::SegmentArena(_) => {
                            builtins.push(BuiltinType::SegmentArena)
                        }
                        CoreTypeConcrete::RangeCheck96(_) => {
                            builtins.push(BuiltinType::RangeCheck96)
                        }
                        CoreTypeConcrete::Circuit(CircuitTypeConcrete::AddMod(_)) => {
                            builtins.push(BuiltinType::CircuitAdd)
                        }
                        CoreTypeConcrete::Circuit(CircuitTypeConcrete::MulMod(_)) => {
                            builtins.push(BuiltinType::CircuitMul)
                        }
                        CoreTypeConcrete::GasBuiltin(_) => builtins.push(BuiltinType::Gas),
                        CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
                            builtins.push(BuiltinType::System)
                        }
                        _ => unreachable!("{:?}", ty.info()),
                    }
                } else {
                    break;
                }
            }

            infos.insert(x.id.id, EntryPointInfo { builtins });
        }

        let library_path = NamedTempFile::new()?
            .into_temp_path()
            .keep()
            .expect("can only fail on windows");

        let object_data = crate::module_to_object(&module, opt_level)?;
        crate::object_to_shared_lib(&object_data, &library_path)?;

        Ok(Self {
            library: Arc::new(unsafe { Library::new(&library_path)? }),
            path: library_path,
            is_temp_path: true,
            entry_points_info: infos,
        })
    }

    /// Save the library to the desired path, alongside it is saved also a json file with additional info.
    pub fn save(&mut self, to: impl AsRef<Path>) -> Result<()> {
        let to = to.as_ref();
        std::fs::copy(&self.path, to)?;

        let info = serde_json::to_string(&self.entry_points_info)?;
        let path = to.with_extension("json");
        std::fs::write(path, info)?;

        self.path = to.to_path_buf();
        self.is_temp_path = false;

        Ok(())
    }

    /// Load the executor from an already compiled library with the additional info json file.
    pub fn load(library_path: &Path) -> Result<Self> {
        let info_str = std::fs::read_to_string(library_path.with_extension("json"))?;
        let info: BTreeMap<u64, EntryPointInfo> = serde_json::from_str(&info_str)?;
        Ok(Self {
            library: Arc::new(unsafe { Library::new(library_path)? }),
            path: library_path.to_path_buf(),
            is_temp_path: false,
            entry_points_info: info,
        })
    }

    /// Runs the given entry point.
    pub fn run(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        builtin_costs: Option<[u64; 7]>,
        mut syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult> {
        let arena = Bump::new();
        let mut invoke_data = Vec::<u8>::new();

        let function_ptr = self.find_function_ptr(function_id, true)?;
        let builtin_costs_ptr = self.find_symbol_ptr("builtin_costs");

        let fallback_builtin_costs = [
            token_gas_cost(CostTokenType::Const) as u64,
            token_gas_cost(CostTokenType::Pedersen) as u64,
            token_gas_cost(CostTokenType::Bitwise) as u64,
            token_gas_cost(CostTokenType::EcOp) as u64,
            token_gas_cost(CostTokenType::Poseidon) as u64,
            token_gas_cost(CostTokenType::AddMod) as u64,
            token_gas_cost(CostTokenType::MulMod) as u64,
        ];

        let builtin_costs = if let Some(builtin_costs) = &builtin_costs {
            builtin_costs.as_slice()
        } else {
            fallback_builtin_costs.as_slice()
        };

        if let Some(builtin_costs_ptr) = builtin_costs_ptr {
            unsafe {
                *builtin_costs_ptr.cast() = builtin_costs.as_ptr();
            }
        }

        //  it can vary from contract to contract thats why we need to store/ load it.
        // substract 2, which are the gas and syscall builtin
        let num_builtins = self.entry_points_info[&function_id.id].builtins.len() - 2;

        // There is always a return ptr because contracts always return more than 1 thing (builtin counters, syscall, enum)
        let return_ptr = arena.alloc_layout(unsafe {
            // 64 = size of enum + syscall + u128 from gas builtin + 8 bytes for each additional builtin counter
            // align is 16 because of the u128
            Layout::from_size_align_unchecked(64 + 8 * num_builtins, 16)
        });

        return_ptr.as_ptr().to_bytes(&mut invoke_data)?;

        let mut syscall_handler = StarknetSyscallHandlerCallbacks::new(&mut syscall_handler);

        for b in &self.entry_points_info[&function_id.id].builtins {
            match b {
                BuiltinType::Gas => {
                    let gas = gas.unwrap_or(0);
                    gas.to_bytes(&mut invoke_data)?;
                }
                BuiltinType::System => {
                    (&mut syscall_handler as *mut StarknetSyscallHandlerCallbacks<_>)
                        .to_bytes(&mut invoke_data)?;
                }
                _ => {
                    0u64.to_bytes(&mut invoke_data)?;
                }
            }
        }

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

        for b in &self.entry_points_info[&function_id.id].builtins {
            match b {
                BuiltinType::Gas => {
                    remaining_gas = unsafe { *read_value::<u128>(return_ptr) };
                }
                BuiltinType::System => {
                    let ptr = return_ptr.cast::<*mut ()>();
                    *return_ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(1)).cast() };
                }
                BuiltinType::BuiltinCosts => {
                    let ptr = return_ptr.cast::<*mut ()>();
                    *return_ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(1)).cast() };
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
                .expect("nonnull is null")
        };

        let tag = *unsafe { enum_ptr.cast::<u8>().as_ref() } as usize;
        // layout of both enum variants, both are a array of felts
        let value_layout = unsafe { Layout::from_size_align_unchecked(24, 8) };
        let value_ptr = unsafe {
            enum_ptr
                .cast::<u8>()
                .add(tag_layout.extend(value_layout)?.1)
        };

        let value_ptr = &mut value_ptr.cast();

        let array_ptr: *mut u8 = unsafe { *read_value(value_ptr) };
        let start: u32 = unsafe { *read_value(value_ptr) };
        let end: u32 = unsafe { *read_value(value_ptr) };
        let _cap: u32 = unsafe { *read_value(value_ptr) };

        let elem_stride = felt_layout.pad_to_align().size();

        // this pointer can be null if the array has a size of 0.
        let data_ptr = unsafe { array_ptr.byte_add(elem_stride * start as usize) };

        assert!(end >= start);
        let num_elems = (end - start) as usize;
        let mut array_value = Vec::with_capacity(num_elems);

        for i in 0..num_elems {
            // safe to create a NonNull because if the array has elements, the data_ptr can't be null.
            let cur_elem_ptr = NonNull::new(unsafe { data_ptr.byte_add(elem_stride * i) }).unwrap();
            let data = unsafe { cur_elem_ptr.cast::<[u8; 32]>().as_ref() };
            let data = Felt::from_bytes_le_slice(data);

            array_value.push(data);
        }

        if !array_ptr.is_null() {
            unsafe { libc::free(array_ptr.cast()) };
        }

        let mut error_msg = None;

        if tag != 0 {
            let bytes_err: Vec<_> = array_value
                .iter()
                .flat_map(|felt| felt.to_bytes_be().to_vec())
                // remove null chars
                .filter(|b| *b != 0)
                .collect();
            let str_error = decode_error_message(&bytes_err);

            error_msg = Some(str_error);
        }

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

impl Drop for AotContractExecutor {
    fn drop(&mut self) {
        if self.is_temp_path {
            std::fs::remove_file(&self.path).ok();
            std::fs::remove_file(self.path.with_extension("json")).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{starknet_stub::StubSyscallHandler, utils::test::load_starknet};
    use cairo_lang_sierra::program::Program;
    use rstest::*;

    #[fixture]
    fn starknet_program() -> Program {
        let (_, program) = load_starknet! {
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
    fn starknet_program_empty() -> Program {
        let (_, program) = load_starknet! {
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
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    fn test_contract_executor(starknet_program: Program, #[case] optlevel: OptLevel) {
        let executor = AotContractExecutor::new(&starknet_program, optlevel).unwrap();

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = executor
            .run(
                entrypoint_function_id,
                &[2.into()],
                Some(u64::MAX as u128),
                None,
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![Felt::from(2), Felt::from(4)]);
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    fn test_contract_executor_empty(starknet_program_empty: Program, #[case] optlevel: OptLevel) {
        let executor = AotContractExecutor::new(&starknet_program_empty, optlevel).unwrap();

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program_empty
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = executor
            .run(
                entrypoint_function_id,
                &[],
                Some(u64::MAX as u128),
                None,
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![]);
    }
}
