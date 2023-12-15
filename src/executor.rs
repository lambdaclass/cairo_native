pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use crate::{
    execution_result::ExecutionResult,
    types::TypeBuilder,
    values::{JitValue, ValueBuilder},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    ids::ConcreteTypeId,
    program::FunctionSignature,
    program_registry::{ProgramRegistry, ProgramRegistryError},
};
use libc::c_void;
use starknet_types_core::felt::Felt;
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
        ret_ptr: &mut [u64; 4],
    );
}

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
    mut gas: Option<u128>,
) -> ExecutionResult {
    tracing::info!("Invoking function with signature: {function_signature:?}.");

    let arena = Bump::new();
    let mut invoke_data = Vec::new();

    // Generate return pointer (if necessary).
    let return_ptr = if function_signature.ret_types.len() > 1
        || function_signature
            .ret_types
            .last()
            .is_some_and(|id| registry.get_type(id).unwrap().is_complex())
    {
        let layout = function_signature
            .ret_types
            .iter()
            .fold(Layout::new::<()>(), |layout, id| {
                let type_info = registry.get_type(id).unwrap();
                layout
                    .extend(type_info.layout(registry).unwrap())
                    .unwrap()
                    .0
            });

        let return_ptr = arena.alloc_layout(layout).cast::<()>();
        invoke_data.push(return_ptr.as_ptr() as u64);

        Some((layout, return_ptr))
    } else {
        invoke_data.push(0);
        None
    };

    // Generate argument list.
    let mut values_iter = args.iter().peekable();
    for type_id in &function_signature.param_types {
        let value = values_iter.peek().unwrap();
        if map_arg_to_values(&arena, &mut invoke_data, registry, type_id, value, gas).unwrap() {
            values_iter.next().unwrap();
        }
    }

    // Invoke the trampoline.
    let mut ret_registers = [0; 4];
    unsafe {
        aot_trampoline(
            function_ptr,
            invoke_data.as_ptr(),
            invoke_data.len(),
            &mut ret_registers,
        );
    }

    // Parse return values.
    let type_info = registry
        .get_type(function_signature.ret_types.last().unwrap())
        .unwrap();
    let return_value = match type_info {
        CoreTypeConcrete::Array(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                return_ptr,
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => unreachable!("Array<T> is complex"),
        },
        CoreTypeConcrete::EcPoint(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => unreachable!("EcPoint is complex"),
        },
        CoreTypeConcrete::EcState(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => unreachable!("EcState is complex"),
        },
        CoreTypeConcrete::Enum(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => unreachable!("Enum<...> is complex"),
        },
        CoreTypeConcrete::Felt252(_) => match return_ptr {
            Some(_) => todo!(),
            None => JitValue::Felt252(Felt::from_bytes_le(unsafe {
                std::mem::transmute::<&[u64; 4], &[u8; 32]>(&ret_registers)
            })),
        },
        CoreTypeConcrete::Felt252Dict(_) => todo!(),
        CoreTypeConcrete::Struct(info) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                return_ptr,
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None if info.members.is_empty() => JitValue::Struct {
                fields: Vec::new(),
                debug_name: function_signature
                    .ret_types
                    .last()
                    .unwrap()
                    .debug_name
                    .as_deref()
                    .map(ToString::to_string),
            },
            None => unreachable!("Struct<...> is complex"),
        },
        CoreTypeConcrete::Uint128(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => JitValue::Uint128(ret_registers[0] as u128 | (ret_registers[1] as u128) << 64),
        },
        CoreTypeConcrete::Uint64(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => JitValue::Uint64(ret_registers[0]),
        },
        CoreTypeConcrete::Uint32(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => JitValue::Uint32(ret_registers[0] as u32),
        },
        CoreTypeConcrete::Uint16(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => JitValue::Uint16(ret_registers[0] as u16),
        },
        CoreTypeConcrete::Uint8(_) => match return_ptr {
            Some((_, return_ptr)) => JitValue::from_jit(
                unsafe { NonNull::new_unchecked(return_ptr.as_ptr()) },
                function_signature.ret_types.last().unwrap(),
                registry,
            ),
            None => JitValue::Uint8(ret_registers[0] as u8),
        },
        _ => todo!("unsupported return type"),
    };

    // TODO: Handle gas.
    ExecutionResult {
        remaining_gas: gas,
        return_value,
    }
}

fn map_arg_to_values(
    arena: &Bump,
    invoke_data: &mut Vec<u64>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_id: &ConcreteTypeId,
    value: &JitValue,
    gas: Option<u128>,
) -> Result<bool, Box<ProgramRegistryError>> {
    let type_info = program_registry.get_type(type_id)?;

    // TODO: Find out if builtins push an argument or not. My guess is that they do.
    match (type_info, value) {
        (CoreTypeConcrete::Array(info), JitValue::Array(values)) => {
            // TODO: Assert that `info.ty` matches all the values' types.

            let type_info = match program_registry.get_type(type_id)? {
                CoreTypeConcrete::Array(type_info) => program_registry.get_type(&type_info.ty)?,
                _ => unreachable!(),
            };
            let type_layout = type_info.layout(program_registry).unwrap().pad_to_align();

            // This needs to be a heap-allocated pointer because it's the actual array data.
            let ptr = unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) };

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
        (CoreTypeConcrete::Enum(_info), JitValue::Enum { .. }) => {
            todo!()
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
                    field_type_id,
                    field_value,
                    gas,
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
        (CoreTypeConcrete::Bitwise(_), _)
        | (CoreTypeConcrete::BuiltinCosts(_), _)
        | (CoreTypeConcrete::EcOp(_), _)
        | (CoreTypeConcrete::Pedersen(_), _)
        | (CoreTypeConcrete::Poseidon(_), _)
        | (CoreTypeConcrete::RangeCheck(_), _)
        | (CoreTypeConcrete::SegmentArena(_), _) => return Ok(false),
        (CoreTypeConcrete::GasBuiltin(_), _) => {
            let value = gas.unwrap();
            invoke_data.push(value as u64);
            invoke_data.push((value >> 64) as u64);
        }
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

    Ok(true)
}
