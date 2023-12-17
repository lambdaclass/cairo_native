pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use crate::{execution_result::ExecutionResult, types::TypeBuilder, values::JitValue};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
    },
    ids::ConcreteTypeId,
    program::FunctionSignature,
    program_registry::{ProgramRegistry, ProgramRegistryError},
};
use libc::c_void;
use std::{
    alloc::Layout,
    arch::global_asm,
    ptr::{null_mut, NonNull},
};

mod aot;
mod jit;

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));

extern "C" {
    /// Invoke an AOT-compiled function.
    ///
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    fn aot_trampoline(
        fn_ptr: *const c_void,
        args_ptr: *const u64,
        args_len: usize,
        ret_ptr: *mut u64,
    );
}

#[allow(clippy::large_enum_variant)]
pub enum NativeExecutor<'m> {
    Aot(AotNativeExecutor<CoreType, CoreLibfunc>),
    Jit(JitNativeExecutor<'m>),
}

impl<'m> From<AotNativeExecutor<CoreType, CoreLibfunc>> for NativeExecutor<'m> {
    fn from(value: AotNativeExecutor<CoreType, CoreLibfunc>) -> Self {
        Self::Aot(value)
    }
}

impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
    fn from(value: JitNativeExecutor<'m>) -> Self {
        Self::Jit(value)
    }
}

fn invoke_dynamic(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_ptr: *const c_void,
    function_signature: &FunctionSignature,
    args: &[JitValue],
    gas: Option<u128>,
    syscall_handler: Option<NonNull<()>>,
) -> ExecutionResult {
    tracing::info!("Invoking function with signature: {function_signature:?}.");

    let arena = Bump::new();
    let mut invoke_data = Vec::new();

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
            !<CoreTypeConcrete as TypeBuilder<CoreType, CoreLibfunc>>::is_zst(info, registry)
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
        invoke_data.push(return_ptr.as_ptr() as u64);

        Some(return_ptr)
    } else {
        // invoke_data.push(0);
        None
    };

    // Generate argument list.
    let mut iter = args.iter();
    for type_id in function_signature.param_types.iter().filter(|id| {
        let info = registry.get_type(id).unwrap();
        !<CoreTypeConcrete as TypeBuilder<CoreType, CoreLibfunc>>::is_zst(info, registry)
    }) {
        let type_info = registry.get_type(type_id).unwrap();

        // Process gas requirements and syscall handler.
        match type_info {
            CoreTypeConcrete::GasBuiltin(_) => match gas {
                Some(gas) => {
                    invoke_data.push(gas as u64);
                    invoke_data.push((gas >> 64) as u64);
                }
                None => panic!("Gas is required"),
            },
            CoreTypeConcrete::StarkNet(_) => match syscall_handler {
                Some(syscall_handler) => invoke_data.push(syscall_handler.as_ptr() as u64),
                None => panic!("Syscall handler is required"),
            },
            _ => map_arg_to_values(
                &arena,
                &mut invoke_data,
                registry,
                type_info,
                iter.next().unwrap(),
            )
            .unwrap(),
        }
    }

    // Invoke the trampoline.
    #[cfg(target_arch = "x86_64")]
    let mut ret_registers = [0; 2];
    #[cfg(target_arch = "aarch64")]
    let mut ret_registers = [0; 4];

    unsafe {
        aot_trampoline(
            function_ptr,
            invoke_data.as_ptr(),
            invoke_data.len(),
            ret_registers.as_mut_ptr(),
        );
    }

    // Parse final gas.
    let mut remaining_gas = None;
    for type_id in &function_signature.ret_types {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::GasBuiltin(_) => {
                remaining_gas = Some(match &mut return_ptr {
                    Some(return_ptr) => unsafe {
                        let ptr = return_ptr.cast::<u128>();
                        *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
                        *ptr.as_ref()
                    },
                    None => {
                        // If there's no return ptr then the function only returned the gas. We don't
                        // need to bother with the syscall handler builtin.
                        ((ret_registers[1] as u128) << 64) | ret_registers[0] as u128
                    }
                });
            }
            CoreTypeConcrete::StarkNet(_) => match &mut return_ptr {
                Some(return_ptr) => unsafe {
                    let ptr = return_ptr.cast::<*mut ()>();
                    *return_ptr = NonNull::new_unchecked(ptr.as_ptr().add(1)).cast();
                },
                None => {}
            },
            _ if <CoreTypeConcrete as TypeBuilder<CoreType, CoreLibfunc>>::is_builtin(
                type_info,
            ) => {}
            _ => break,
        }
    }

    // Parse return values.
    let return_value = parse_result(
        function_signature.ret_types.last().unwrap(),
        registry,
        return_ptr,
        ret_registers,
    );

    // FIXME: Arena deallocation.
    std::mem::forget(arena);

    ExecutionResult {
        remaining_gas,
        return_value,
    }
}

fn map_arg_to_values(
    arena: &Bump,
    invoke_data: &mut Vec<u64>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_info: &CoreTypeConcrete,
    value: &JitValue,
) -> Result<(), Box<ProgramRegistryError>> {
    // TODO: Find out if builtins push an argument or not. My guess is that they do.
    match (type_info, value) {
        (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
            // TODO: Assert that `info.ty` matches all the values' types.

            let type_info = program_registry.get_type(&info.ty)?;
            let type_layout = type_info.layout(program_registry).unwrap().pad_to_align();

            // This needs to be a heap-allocated pointer because it's the actual array data.
            let ptr = if values.is_empty() {
                null_mut()
            } else {
                unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) }
            };

            for (idx, value) in values.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        value
                            .to_jit(arena, program_registry, &info.ty)
                            .unwrap()
                            .cast()
                            .as_ptr(),
                        (ptr as usize + type_layout.size() * idx) as *mut u8,
                        type_layout.size(),
                    );
                }
            }

            invoke_data.push(ptr as u64);
            invoke_data.push(values.len() as u64);
            invoke_data.push(values.len() as u64);
        }
        (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
            invoke_data.extend(a.to_le_digits());
            invoke_data.extend(b.to_le_digits());
        }
        (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
            invoke_data.extend(a.to_le_digits());
            invoke_data.extend(b.to_le_digits());
            invoke_data.extend(c.to_le_digits());
            invoke_data.extend(d.to_le_digits());
        }
        (CoreTypeConcrete::Enum(info), JitValue::Enum { tag, value, .. }) => {
            if type_info.is_memory_allocated(program_registry) {
                let (layout, tag_layout, variant_layouts) =
                    crate::types::r#enum::get_layout_for_variants(program_registry, &info.variants)
                        .unwrap();

                let ptr = arena.alloc_layout(layout);
                unsafe {
                    match tag_layout.size() {
                        0 => {}
                        1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                        2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                        4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                        8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                        _ => unreachable!(),
                    }
                }

                let offset = tag_layout.extend(variant_layouts[*tag]).unwrap().1;
                let payload_ptr = value
                    .to_jit(arena, program_registry, &info.variants[*tag])
                    .unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        payload_ptr.cast::<u8>().as_ptr(),
                        ptr.cast::<u8>().as_ptr().add(offset),
                        variant_layouts[*tag].size(),
                    );
                }

                invoke_data.push(ptr.as_ptr() as u64);
            } else {
                // Write the tag.
                match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                    0 => {}
                    _ => invoke_data.push(*tag as u64),
                }

                // Write the payload.
                let type_info = program_registry.get_type(&info.variants[*tag]).unwrap();
                map_arg_to_values(arena, invoke_data, program_registry, type_info, value)?;
            }
        }
        (CoreTypeConcrete::Felt252(_), JitValue::Felt252(value)) => {
            invoke_data.extend(value.to_le_digits());
        }
        (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => {
            #[cfg(not(feature = "cairo-native-runtime"))]
            unimplemented!("enable the `cairo-native-runtime` feature to use felt252 dicts");

            #[cfg(feature = "cairo-native-runtime")]
            // invoke_data.push(value.iter().fold(
            //     unsafe { cairo_native_runtime::cairo_native__alloc_dict() },
            //     |ptr, (key, val)| unsafe {
            //         cairo_native_runtime::cairo_native__dict_insert(
            //             ptr,
            //             &key.to_le_bytes(),
            //             todo!(),
            //         )
            //     },
            // ) as u64);
            todo!("Flatten felt252_dict into Vec<u64> for the AOT interface's arguments'.")
        }
        (CoreTypeConcrete::Struct(info), JitValue::Struct { fields, .. }) => {
            for (field_type_id, field_value) in info.members.iter().zip(fields) {
                map_arg_to_values(
                    arena,
                    invoke_data,
                    program_registry,
                    program_registry.get_type(field_type_id)?,
                    field_value,
                )?;
            }
        }
        (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => {
            invoke_data.push(*value as u64);
            invoke_data.push((value >> 64) as u64);
        }
        (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
            invoke_data.push(*value);
        }
        (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Sint32(_), JitValue::Sint32(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Sint16(_), JitValue::Sint16(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Sint8(_), JitValue::Sint8(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::NonZero(info), _) => {
            // TODO: Check that the value is indeed non-zero.
            let type_info = program_registry.get_type(&info.ty)?;
            map_arg_to_values(arena, invoke_data, program_registry, type_info, value)?;
        }
        (CoreTypeConcrete::Snapshot(info), _) => {
            let type_info = program_registry.get_type(&info.ty)?;
            map_arg_to_values(arena, invoke_data, program_registry, type_info, value)?;
        }
        (CoreTypeConcrete::Bitwise(_), _)
        | (CoreTypeConcrete::BuiltinCosts(_), _)
        | (CoreTypeConcrete::EcOp(_), _)
        | (CoreTypeConcrete::Pedersen(_), _)
        | (CoreTypeConcrete::Poseidon(_), _)
        | (CoreTypeConcrete::RangeCheck(_), _)
        | (CoreTypeConcrete::SegmentArena(_), _) => {}
        (sierra_ty, arg_ty) => match sierra_ty {
            CoreTypeConcrete::Array(_) => todo!("Array {arg_ty:?}"),
            CoreTypeConcrete::Bitwise(_) => todo!("Bitwise {arg_ty:?}"),
            CoreTypeConcrete::Box(_) => todo!("Box {arg_ty:?}"),
            CoreTypeConcrete::EcOp(_) => todo!("EcOp {arg_ty:?}"),
            CoreTypeConcrete::EcPoint(_) => todo!("EcPoint {arg_ty:?}"),
            CoreTypeConcrete::EcState(_) => todo!("EcState {arg_ty:?}"),
            CoreTypeConcrete::Felt252(_) => todo!("Felt252 {arg_ty:?}"),
            CoreTypeConcrete::GasBuiltin(_) => todo!("GasBuiltin {arg_ty:?}"),
            CoreTypeConcrete::BuiltinCosts(_) => todo!("BuiltinCosts {arg_ty:?}"),
            CoreTypeConcrete::Uint8(_) => todo!("Uint8 {arg_ty:?}"),
            CoreTypeConcrete::Uint16(_) => todo!("Uint16 {arg_ty:?}"),
            CoreTypeConcrete::Uint32(_) => todo!("Uint32 {arg_ty:?}"),
            CoreTypeConcrete::Uint64(_) => todo!("Uint64 {arg_ty:?}"),
            CoreTypeConcrete::Uint128(_) => todo!("Uint128 {arg_ty:?}"),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!("Uint128MulGuarantee {arg_ty:?}"),
            CoreTypeConcrete::Sint8(_) => todo!("Sint8 {arg_ty:?}"),
            CoreTypeConcrete::Sint16(_) => todo!("Sint16 {arg_ty:?}"),
            CoreTypeConcrete::Sint32(_) => todo!("Sint32 {arg_ty:?}"),
            CoreTypeConcrete::Sint64(_) => todo!("Sint64 {arg_ty:?}"),
            CoreTypeConcrete::Sint128(_) => todo!("Sint128 {arg_ty:?}"),
            CoreTypeConcrete::NonZero(_) => todo!("NonZero {arg_ty:?}"),
            CoreTypeConcrete::Nullable(_) => todo!("Nullable {arg_ty:?}"),
            CoreTypeConcrete::RangeCheck(_) => todo!("RangeCheck {arg_ty:?}"),
            CoreTypeConcrete::Uninitialized(_) => todo!("Uninitialized {arg_ty:?}"),
            CoreTypeConcrete::Enum(_) => todo!("Enum {arg_ty:?}"),
            CoreTypeConcrete::Struct(_) => todo!("Struct {arg_ty:?}"),
            CoreTypeConcrete::Felt252Dict(_) => todo!("Felt252Dict {arg_ty:?}"),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!("Felt252DictEntry {arg_ty:?}"),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!("SquashedFelt252Dict {arg_ty:?}"),
            CoreTypeConcrete::Pedersen(_) => todo!("Pedersen {arg_ty:?}"),
            CoreTypeConcrete::Poseidon(_) => todo!("Poseidon {arg_ty:?}"),
            CoreTypeConcrete::Span(_) => todo!("Span {arg_ty:?}"),
            CoreTypeConcrete::StarkNet(_) => todo!("StarkNet {arg_ty:?}"),
            CoreTypeConcrete::SegmentArena(_) => todo!("SegmentArena {arg_ty:?}"),
            CoreTypeConcrete::Snapshot(_) => todo!("Snapshot {arg_ty:?}"),
            CoreTypeConcrete::Bytes31(_) => todo!("Bytes31 {arg_ty:?}"),
        },
    }

    Ok(())
}

fn parse_result(
    type_id: &ConcreteTypeId,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    return_ptr: Option<NonNull<()>>,
    #[cfg(target_arch = "x86_64")] ret_registers: [u64; 2],
    #[cfg(target_arch = "aarch64")] ret_registers: [u64; 4],
) -> JitValue {
    let type_info = registry.get_type(type_id).unwrap();

    match type_info {
        CoreTypeConcrete::Array(_) => JitValue::from_jit(return_ptr.unwrap(), type_id, registry),
        CoreTypeConcrete::Box(_) => todo!(),
        CoreTypeConcrete::EcPoint(_) => JitValue::from_jit(return_ptr.unwrap(), type_id, registry),
        CoreTypeConcrete::EcState(_) => JitValue::from_jit(return_ptr.unwrap(), type_id, registry),
        CoreTypeConcrete::Felt252(_)
        | CoreTypeConcrete::StarkNet(
            StarkNetTypeConcrete::ClassHash(_)
            | StarkNetTypeConcrete::ContractAddress(_)
            | StarkNetTypeConcrete::StorageAddress(_)
            | StarkNetTypeConcrete::StorageBaseAddress(_),
        ) => {
            #[cfg(target_arch = "x86_64")]
            let value = JitValue::from_jit(return_ptr.unwrap(), type_id, registry);

            #[cfg(target_arch = "aarch64")]
            let value = JitValue::Felt252(Felt::from_bytes_le(unsafe {
                std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
            }));

            value
        }
        CoreTypeConcrete::Uint8(_) => {
            assert!(return_ptr.is_none());
            JitValue::Uint8(ret_registers[0] as u8)
        }
        CoreTypeConcrete::Uint16(_) => {
            assert!(return_ptr.is_none());
            JitValue::Uint16(ret_registers[0] as u16)
        }
        CoreTypeConcrete::Uint32(_) => {
            assert!(return_ptr.is_none());
            JitValue::Uint32(ret_registers[0] as u32)
        }
        CoreTypeConcrete::Uint64(_) => {
            assert!(return_ptr.is_none());
            JitValue::Uint64(ret_registers[0])
        }
        CoreTypeConcrete::Uint128(_) => {
            assert!(return_ptr.is_none());
            JitValue::Uint128(((ret_registers[1] as u128) << 64) | ret_registers[0] as u128)
        }
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => {
            assert!(return_ptr.is_none());
            JitValue::Sint8(ret_registers[0] as i8)
        }
        CoreTypeConcrete::Sint16(_) => {
            assert!(return_ptr.is_none());
            JitValue::Sint16(ret_registers[0] as i16)
        }
        CoreTypeConcrete::Sint32(_) => {
            assert!(return_ptr.is_none());
            JitValue::Sint32(ret_registers[0] as i32)
        }
        CoreTypeConcrete::Sint64(_) => todo!(),
        CoreTypeConcrete::Sint128(_) => todo!(),
        CoreTypeConcrete::NonZero(info) => {
            parse_result(&info.ty, registry, return_ptr, ret_registers)
        }
        CoreTypeConcrete::Nullable(_) => todo!(),
        CoreTypeConcrete::Uninitialized(_) => todo!(),
        CoreTypeConcrete::Enum(info) => {
            let (_, tag_layout, variant_layouts) =
                crate::types::r#enum::get_layout_for_variants(registry, &info.variants).unwrap();

            let (tag, ptr) = if type_info.is_memory_allocated(registry) {
                let ptr = return_ptr.unwrap();

                let tag = unsafe {
                    match tag_layout.size() {
                        0 => 0,
                        1 => *ptr.cast::<u8>().as_ref() as usize,
                        2 => *ptr.cast::<u16>().as_ref() as usize,
                        4 => *ptr.cast::<u32>().as_ref() as usize,
                        8 => *ptr.cast::<u64>().as_ref() as usize,
                        _ => unreachable!(),
                    }
                };

                (tag, unsafe {
                    NonNull::new_unchecked(
                        ptr.cast::<u8>()
                            .as_ptr()
                            .add(tag_layout.extend(variant_layouts[tag]).unwrap().1),
                    )
                    .cast()
                })
            } else {
                // TODO: Shouldn't the pointer be always `None` within this block?
                match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                    0 => (0, return_ptr.unwrap_or_else(NonNull::dangling)),
                    _ => (
                        match tag_layout.size() {
                            0 => 0,
                            1 => ret_registers[0] as u8 as usize,
                            2 => ret_registers[0] as u16 as usize,
                            4 => ret_registers[0] as u32 as usize,
                            8 => ret_registers[0] as usize,
                            _ => unreachable!(),
                        },
                        return_ptr.unwrap_or_else(NonNull::dangling),
                    ),
                }
            };
            let value = Box::new(JitValue::from_jit(ptr, &info.variants[tag], registry));

            JitValue::Enum {
                tag,
                value,
                debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
            }
        }
        CoreTypeConcrete::Struct(info) => {
            if info.members.is_empty() {
                JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: type_id.debug_name.as_deref().map(ToString::to_string),
                }
            } else {
                JitValue::from_jit(return_ptr.unwrap(), type_id, registry)
            }
        }
        CoreTypeConcrete::Felt252Dict(_) => todo!(),
        CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
        CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
        CoreTypeConcrete::Span(_) => todo!(),
        CoreTypeConcrete::Snapshot(_) => todo!(),
        CoreTypeConcrete::Bytes31(_) => todo!(),
        _ => unreachable!(),
    }
}
