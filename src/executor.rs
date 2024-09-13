//! # Executors
//!
//! This module provides methods to execute the programs, either via JIT or compiled ahead
//! of time. It also provides a cache to avoid recompiling previously compiled programs.

pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use crate::{
    arch::{AbiArgument, JitValueWithInfoWrapper},
    error::Error,
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    starknet::{handler::StarknetSyscallHandlerCallbacks, StarknetSyscallHandler},
    types::TypeBuilder,
    utils::RangeExt,
    values::JitValue,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete,
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
        ConcreteType,
    },
    ids::{ConcreteTypeId, FunctionId},
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use libc::c_void;
use num_bigint::BigInt;
use num_traits::One;
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    arch::global_asm,
    ptr::{addr_of_mut, NonNull},
    sync::Arc,
};

mod aot;
pub mod contract;
mod jit;

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));

extern "C" {
    /// Invoke an AOT or JIT-compiled function.
    ///
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    #[cfg_attr(not(target_os = "macos"), link_name = "_invoke_trampoline")]
    fn invoke_trampoline(
        fn_ptr: *const c_void,
        args_ptr: *const u64,
        args_len: usize,
        ret_ptr: *mut u64,
    );
}

/// The cairo native executor, either AOT or JIT based.
#[derive(Debug, Clone)]
pub enum NativeExecutor<'m> {
    Aot(Arc<AotNativeExecutor>),
    Jit(Arc<JitNativeExecutor<'m>>),
}

impl<'a> NativeExecutor<'a> {
    /// Invoke the given function by its function id, with the given arguments and gas.
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
    ) -> Result<ExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => executor.invoke_dynamic(function_id, args, gas),
            NativeExecutor::Jit(executor) => executor.invoke_dynamic(function_id, args, gas),
        }
    }

    /// Invoke the given function by its function id, with the given arguments and gas.
    /// This should be used for programs which require a syscall handler, whose
    /// implementation should be passed on.
    pub fn invoke_dynamic_with_syscall_handler(
        &self,
        function_id: &FunctionId,
        args: &[JitValue],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => executor.invoke_dynamic_with_syscall_handler(
                function_id,
                args,
                gas,
                syscall_handler,
            ),
            NativeExecutor::Jit(executor) => executor.invoke_dynamic_with_syscall_handler(
                function_id,
                args,
                gas,
                syscall_handler,
            ),
        }
    }

    /// Invoke the given function by its function id, with the given arguments and gas.
    /// This should be used for starknet contracts which require a syscall handler, whose
    /// implementation should be passed on.
    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u128>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        match self {
            NativeExecutor::Aot(executor) => {
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
            }
            NativeExecutor::Jit(executor) => {
                executor.invoke_contract_dynamic(function_id, args, gas, syscall_handler)
            }
        }
    }
}

impl<'m> From<AotNativeExecutor> for NativeExecutor<'m> {
    fn from(value: AotNativeExecutor) -> Self {
        Self::Aot(Arc::new(value))
    }
}

impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
    fn from(value: JitNativeExecutor<'m>) -> Self {
        Self::Jit(Arc::new(value))
    }
}

/// Internal method.
///
/// Invokes the given function by constructing the function call depending on the arguments given.
/// Usually calling a function requires knowing it's signature at compile time, but we need to be
/// able to call any given function provided it's signatue (arguments and return type) at runtime,
/// to do so we have a "trampoline" in the given platform assembly (x86_64, aarch64) which
/// constructs the function call in place.
///
/// To pass the arguments, they are stored in a arena.
fn invoke_dynamic(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_ptr: *const c_void,
    function_signature: &FunctionSignature,
    args: &[JitValue],
    gas: u128,
    mut syscall_handler: Option<impl StarknetSyscallHandler>,
) -> Result<ExecutionResult, Error> {
    tracing::info!("Invoking function with signature: {function_signature:?}.");
    let arena = Bump::new();
    let mut invoke_data = Vec::<u8>::new();

    // Generate return pointer (if necessary).
    //
    // Generated when either:
    //   - There are more than one non-zst return values.
    //     - All builtins except GasBuiltin and Starknet are ZST.
    //     - The unit struct is a ZST.
    //   - The return argument is complex.
    let mut ret_types_iter = function_signature
        .ret_types
        .iter()
        .filter(|id| {
            let info = registry.get_type(id).unwrap();
            !(info.is_builtin() && info.is_zst(registry))
        })
        .peekable();

    let num_return_args = ret_types_iter.clone().count();
    let mut return_ptr = if num_return_args > 1
        || ret_types_iter
            .peek()
            .is_some_and(|id| registry.get_type(id).unwrap().is_complex(registry))
    {
        let layout = ret_types_iter.fold(Layout::new::<()>(), |layout, id| {
            let type_info = registry.get_type(id).unwrap();
            layout
                .extend(type_info.layout(registry).unwrap())
                .unwrap()
                .0
        });

        let return_ptr = arena.alloc_layout(layout).cast::<()>();
        return_ptr.as_ptr().to_bytes(&mut invoke_data)?;

        Some(return_ptr)
    } else {
        None
    };

    // The Cairo compiler doesn't specify that the cheatcode syscall needs the syscall handler,
    // so we must always allocate it in case it needs it, regardless of whether it's passed
    // as an argument to the entry point or not.
    let mut syscall_handler = syscall_handler
        .as_mut()
        .map(|syscall_handler| StarknetSyscallHandlerCallbacks::new(syscall_handler));
    // We only care for the previous syscall handler if we actually modify it
    #[cfg(feature = "with-cheatcode")]
    let previous_syscall_handler = syscall_handler.as_mut().map(|syscall_handler| {
        let previous_syscall_handler = crate::starknet::SYSCALL_HANDLER_VTABLE.get();
        let syscall_handler_ptr = std::ptr::addr_of!(*syscall_handler) as *mut ();
        crate::starknet::SYSCALL_HANDLER_VTABLE.set(syscall_handler_ptr);

        previous_syscall_handler
    });

    // Generate argument list.
    let mut iter = args.iter();
    for type_id in function_signature.param_types.iter().filter(|id| {
        let info = registry.get_type(id).unwrap();
        !info.is_zst(registry)
    }) {
        // Process gas requirements and syscall handler.
        match registry.get_type(type_id).unwrap() {
            CoreTypeConcrete::GasBuiltin(_) => gas.to_bytes(&mut invoke_data)?,
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => {
                let syscall_handler = syscall_handler
                    .as_mut()
                    .expect("syscall handler is required");

                (syscall_handler as *mut StarknetSyscallHandlerCallbacks<_>)
                    .to_bytes(&mut invoke_data)?;
            }
            type_info if type_info.is_builtin() => 0u64.to_bytes(&mut invoke_data)?,
            type_info => JitValueWithInfoWrapper {
                value: iter.next().unwrap(),
                type_id,
                info: type_info,

                arena: &arena,
                registry,
            }
            .to_bytes(&mut invoke_data)?,
        }
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

    // If the syscall handler was changed, then reset the previous one.
    // It's only necessary to restore the pointer if it's been modified i.e. if previous_syscall_handler is Some(...)
    #[cfg(feature = "with-cheatcode")]
    if let Some(previous_syscall_handler) = previous_syscall_handler {
        crate::starknet::SYSCALL_HANDLER_VTABLE.set(previous_syscall_handler);
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

    let mut remaining_gas = None;
    let mut builtin_stats = BuiltinStats::default();
    for type_id in &function_signature.ret_types {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::GasBuiltin(_) => {
                remaining_gas = Some(match &mut return_ptr {
                    Some(return_ptr) => unsafe { *read_value::<u128>(return_ptr) },
                    None => {
                        // If there's no return ptr then the function only returned the gas. We don't
                        // need to bother with the syscall handler builtin.
                        ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
                    }
                });
            }
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => match &mut return_ptr {
                Some(return_ptr) => unsafe {
                    let ptr = return_ptr.cast::<*mut ()>();
                    *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
                },
                None => {}
            },
            _ if type_info.is_builtin() => {
                if !type_info.is_zst(registry) {
                    let value = match &mut return_ptr {
                        Some(return_ptr) => unsafe { *read_value::<u64>(return_ptr) },
                        None => ret_registers[0],
                    } as usize;

                    match type_info {
                        CoreTypeConcrete::Bitwise(_) => builtin_stats.bitwise = value,
                        CoreTypeConcrete::EcOp(_) => builtin_stats.ec_op = value,
                        CoreTypeConcrete::RangeCheck(_) => builtin_stats.range_check = value,
                        CoreTypeConcrete::Pedersen(_) => builtin_stats.pedersen = value,
                        CoreTypeConcrete::Poseidon(_) => builtin_stats.poseidon = value,
                        CoreTypeConcrete::SegmentArena(_) => builtin_stats.segment_arena = value,
                        CoreTypeConcrete::RangeCheck96(_) => builtin_stats.range_check_96 = value,
                        CoreTypeConcrete::Circuit(CircuitTypeConcrete::AddMod(_)) => {
                            builtin_stats.circuit_add = value
                        }
                        CoreTypeConcrete::Circuit(CircuitTypeConcrete::MulMod(_)) => {
                            builtin_stats.circuit_mul = value
                        }
                        _ => unreachable!("{type_id:?}"),
                    }
                }
            }
            _ => break,
        }
    }

    // Parse return values.
    let return_value = function_signature
        .ret_types
        .last()
        .and_then(|ret_type| {
            let type_info = registry.get_type(ret_type).unwrap();
            if type_info.is_builtin() {
                None
            } else {
                Some(parse_result(ret_type, registry, return_ptr, ret_registers))
            }
        })
        .unwrap_or_else(|| {
            Ok(JitValue::Struct {
                fields: vec![],
                debug_name: None,
            })
        })?;

    Ok(ExecutionResult {
        remaining_gas,
        return_value,
        builtin_stats,
    })
}

/// Parses the result by reading from the return ptr the given type.
fn parse_result(
    type_id: &ConcreteTypeId,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    mut return_ptr: Option<NonNull<()>>,
    #[cfg(target_arch = "x86_64")] mut ret_registers: [u64; 2],
    #[cfg(target_arch = "aarch64")] mut ret_registers: [u64; 4],
) -> Result<JitValue, Error> {
    let type_info = registry.get_type(type_id).unwrap();
    let debug_name = type_info.info().long_id.to_string();

    // Align the pointer to the actual return value.
    if let Some(return_ptr) = &mut return_ptr {
        let layout = type_info.layout(registry).unwrap();
        let align_offset = return_ptr
            .cast::<u8>()
            .as_ptr()
            .align_offset(layout.align());

        *return_ptr = unsafe {
            NonNull::new(return_ptr.cast::<u8>().as_ptr().add(align_offset))
                .expect("nonnull is null")
                .cast()
        };
    }

    match type_info {
        CoreTypeConcrete::Array(_) => {
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
        }
        CoreTypeConcrete::Box(info) => unsafe {
            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(ret_registers[0] as *mut ()));
            let value = JitValue::from_jit(ptr, &info.ty, registry);
            libc::free(ptr.cast().as_ptr());
            Ok(value)
        },
        CoreTypeConcrete::EcPoint(_) | CoreTypeConcrete::EcState(_) => {
            Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
        }
        CoreTypeConcrete::Felt252(_)
        | CoreTypeConcrete::StarkNet(
            StarkNetTypeConcrete::ClassHash(_)
            | StarkNetTypeConcrete::ContractAddress(_)
            | StarkNetTypeConcrete::StorageAddress(_)
            | StarkNetTypeConcrete::StorageBaseAddress(_),
        ) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            None => {
                #[cfg(target_arch = "x86_64")]
                // Since x86_64's return values hold at most two different 64bit registers,
                // everything bigger than u128 will be returned by memory, therefore making
                // this branch is unreachable on that architecture.
                return Err(Error::ParseAttributeError);

                #[cfg(target_arch = "aarch64")]
                Ok(JitValue::Felt252(
                    starknet_types_core::felt::Felt::from_bytes_le(unsafe {
                        std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
                    }),
                ))
            }
        },
        CoreTypeConcrete::Bytes31(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            None => {
                #[cfg(target_arch = "x86_64")]
                // Since x86_64's return values hold at most two different 64bit registers,
                // everything bigger than u128 will be returned by memory, therefore making
                // this branch is unreachable on that architecture.
                return Err(Error::ParseAttributeError);

                #[cfg(target_arch = "aarch64")]
                Ok(JitValue::Bytes31(unsafe {
                    *std::mem::transmute::<&[u64; 4], &[u8; 31]>(&ret_registers)
                }))
            }
        },
        CoreTypeConcrete::BoundedInt(info) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::from_jit(return_ptr, type_id, registry)),
            None => {
                let mut data = if info.range.offset_bit_width() <= 64 {
                    BigInt::from(ret_registers[0])
                } else {
                    BigInt::from(((ret_registers[1] as u128) << 64) | ret_registers[0] as u128)
                };

                data &= (BigInt::one() << info.range.offset_bit_width()) - BigInt::one();
                data += &info.range.lower;

                Ok(JitValue::BoundedInt {
                    value: data.into(),
                    range: info.range.clone(),
                })
            }
        },
        CoreTypeConcrete::Uint8(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint8(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint8(ret_registers[0] as u8)),
        },
        CoreTypeConcrete::Uint16(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint16(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint16(ret_registers[0] as u16)),
        },
        CoreTypeConcrete::Uint32(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint32(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint32(ret_registers[0] as u32)),
        },
        CoreTypeConcrete::Uint64(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint64(ret_registers[0])),
        },
        CoreTypeConcrete::Uint128(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Uint128(
                ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128,
            )),
        },
        CoreTypeConcrete::Sint8(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint8(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint8(ret_registers[0] as i8)),
        },
        CoreTypeConcrete::Sint16(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint16(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint16(ret_registers[0] as i16)),
        },
        CoreTypeConcrete::Sint32(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Sint32(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint32(ret_registers[0] as i32)),
        },
        CoreTypeConcrete::Sint64(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint64(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint64(ret_registers[0] as i64)),
        },
        CoreTypeConcrete::Sint128(_) => match return_ptr {
            Some(return_ptr) => Ok(JitValue::Uint128(unsafe { *return_ptr.cast().as_ref() })),
            None => Ok(JitValue::Sint128(
                ((ret_registers[1] as i128) << 64) | ret_registers[0] as i128,
            )),
        },
        CoreTypeConcrete::NonZero(info) => {
            parse_result(&info.ty, registry, return_ptr, ret_registers)
        }
        CoreTypeConcrete::Nullable(info) => unsafe {
            let ptr = return_ptr.map_or(ret_registers[0] as *mut (), |x| {
                *x.cast::<*mut ()>().as_ref()
            });
            if ptr.is_null() {
                Ok(JitValue::Null)
            } else {
                let ptr = NonNull::new_unchecked(ptr);
                let value = JitValue::from_jit(ptr, &info.ty, registry);
                libc::free(ptr.as_ptr().cast());
                Ok(value)
            }
        },
        CoreTypeConcrete::Enum(info) => {
            let (_, tag_layout, variant_layouts) =
                crate::types::r#enum::get_layout_for_variants(registry, &info.variants).unwrap();

            let (tag, ptr) = if type_info.is_memory_allocated(registry) || return_ptr.is_some() {
                let ptr = return_ptr.unwrap();

                let tag = unsafe {
                    match tag_layout.size() {
                        0 => 0,
                        1 => *ptr.cast::<u8>().as_ref() as usize,
                        2 => *ptr.cast::<u16>().as_ref() as usize,
                        4 => *ptr.cast::<u32>().as_ref() as usize,
                        8 => *ptr.cast::<u64>().as_ref() as usize,
                        _ => return Err(Error::ParseAttributeError),
                    }
                };

                (
                    tag,
                    Ok(unsafe {
                        NonNull::new_unchecked(
                            ptr.cast::<u8>()
                                .as_ptr()
                                .add(tag_layout.extend(variant_layouts[tag]).unwrap().1),
                        )
                        .cast()
                    }),
                )
            } else {
                match info.variants.len() {
                    0 | 1 => (0, Err(0)),
                    _ => (
                        match tag_layout.size() {
                            1 => ret_registers[0] as u8 as usize,
                            2 => ret_registers[0] as u16 as usize,
                            4 => ret_registers[0] as u32 as usize,
                            8 => ret_registers[0] as usize,
                            _ => return Err(Error::ParseAttributeError),
                        },
                        Err(1),
                    ),
                }
            };
            let value = match ptr {
                Ok(ptr) => Box::new(JitValue::from_jit(ptr, &info.variants[tag], registry)),
                Err(offset) => {
                    ret_registers.copy_within(offset.., 0);
                    Box::new(parse_result(
                        &info.variants[tag],
                        registry,
                        None,
                        ret_registers,
                    )?)
                }
            };

            Ok(JitValue::Enum {
                tag,
                value,
                debug_name: Some(debug_name),
            })
        }
        CoreTypeConcrete::Struct(info) => {
            if info.members.is_empty() {
                Ok(JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: Some(debug_name),
                })
            } else {
                Ok(JitValue::from_jit(return_ptr.unwrap(), type_id, registry))
            }
        }
        CoreTypeConcrete::Felt252Dict(_) | CoreTypeConcrete::SquashedFelt252Dict(_) => unsafe {
            let ptr = return_ptr.unwrap_or(NonNull::new_unchecked(
                addr_of_mut!(ret_registers[0]) as *mut ()
            ));
            let value = JitValue::from_jit(ptr, type_id, registry);
            Ok(value)
        },

        CoreTypeConcrete::Snapshot(info) => {
            parse_result(&info.ty, registry, return_ptr, ret_registers)
        }

        // Builtins are handled before the call to parse_result
        // and should not be reached here.
        CoreTypeConcrete::Bitwise(_)
        | CoreTypeConcrete::Const(_)
        | CoreTypeConcrete::EcOp(_)
        | CoreTypeConcrete::GasBuiltin(_)
        | CoreTypeConcrete::BuiltinCosts(_)
        | CoreTypeConcrete::RangeCheck(_)
        | CoreTypeConcrete::Pedersen(_)
        | CoreTypeConcrete::Poseidon(_)
        | CoreTypeConcrete::SegmentArena(_)
        | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) => unreachable!(),

        CoreTypeConcrete::Felt252DictEntry(_)
        | CoreTypeConcrete::Span(_)
        | CoreTypeConcrete::Uninitialized(_)
        | CoreTypeConcrete::Coupon(_)
        | CoreTypeConcrete::StarkNet(_)
        | CoreTypeConcrete::Uint128MulGuarantee(_)
        | CoreTypeConcrete::Circuit(_)
        | CoreTypeConcrete::RangeCheck96(_) => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::NativeContext;
    use crate::starknet_stub::StubSyscallHandler;
    use crate::utils::test::load_cairo;
    use crate::utils::test::load_starknet;
    use crate::OptLevel;
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
    fn test_invoke_dynamic_aot_native_executor(program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program)
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());

        let native_executor: NativeExecutor = executor.into();

        // The first function in the program is `run_test`.
        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;

        let result = native_executor
            .invoke_dynamic(entrypoint_function_id, &[], Some(u128::MAX))
            .unwrap();

        assert_eq!(result.return_value, JitValue::Felt252(Felt::from(42)));
    }

    #[rstest]
    fn test_invoke_dynamic_jit_native_executor(program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program)
            .expect("failed to compile context");
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default());

        let native_executor: NativeExecutor = executor.into();

        // The first function in the program is `run_test`.
        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;

        let result = native_executor
            .invoke_dynamic(entrypoint_function_id, &[], Some(u128::MAX))
            .unwrap();

        assert_eq!(result.return_value, JitValue::Felt252(Felt::from(42)));
    }

    #[rstest]
    fn test_invoke_contract_dynamic_aot(starknet_program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&starknet_program)
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, OptLevel::default());

        let native_executor: NativeExecutor = executor.into();

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = native_executor
            .invoke_contract_dynamic(
                entrypoint_function_id,
                &[],
                Some(u128::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![Felt::from(42)]);
    }

    #[rstest]
    fn test_invoke_contract_dynamic_jit(starknet_program: Program) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&starknet_program)
            .expect("failed to compile context");
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default());

        let native_executor: NativeExecutor = executor.into();

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = native_executor
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
