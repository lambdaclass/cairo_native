use crate::{
    types::{r#enum::get_layout_for_variants, TypeBuilder},
    utils::{generate_function_name, get_integer_layout},
    values::{JitValue, ValueBuilder},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        GenericLibfunc, GenericType,
    },
    ids::{ConcreteTypeId, FunctionId},
    program_registry::{ProgramRegistry, ProgramRegistryError},
};
use libc::c_void;
use libloading::Library;
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    arch::global_asm,
    io::{Cursor, Read},
    iter::repeat,
    ptr::{null_mut, NonNull},
};

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("../arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("../arch/x86_64.s"));

// #[cfg(target_arch = "aarch64")]
// const NUM_REGISTER_ARGS: usize = 8;
// #[cfg(target_arch = "x86_64")]
// const NUM_REGISTER_ARGS: usize = 6;

extern "C" {
    /// Invoke an AOT-compiled function.
    ///
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    fn aot_trampoline(
        fn_ptr: *mut c_void,
        args_ptr: *const u64,
        args_len: usize,
        ret_ptr: &mut [u64; 4],
    );
}

pub struct AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    library: Library,
    registry: ProgramRegistry<TType, TLibfunc>,
}

impl<TType, TLibfunc> AotNativeExecutor<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder,
{
    pub fn new(library: Library, registry: ProgramRegistry<TType, TLibfunc>) -> Self {
        Self { library, registry }
    }
}

impl AotNativeExecutor<CoreType, CoreLibfunc> {
    pub fn invoke_dynamic(&self, function_id: &FunctionId, args: &[JitValue]) -> JitValue {
        let function_name = generate_function_name(function_id);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        let function_ptr = unsafe {
            self.library
                .get::<extern "C" fn()>(function_name.as_bytes())
                .unwrap()
        };

        let function_signature = &self.registry.get_function(function_id).unwrap().signature;
        let mut invoke_data = Vec::new();

        // Generate return pointer (if necessary).
        let return_ptr = if function_signature.ret_types.len() > 1
            || function_signature
                .ret_types
                .last()
                .is_some_and(|id| self.registry.get_type(id).unwrap().is_complex())
        {
            let layout =
                function_signature
                    .ret_types
                    .iter()
                    .fold(Layout::new::<()>(), |layout, id| {
                        let type_info = self.registry.get_type(id).unwrap();
                        layout
                            .extend(type_info.layout(&self.registry).unwrap())
                            .unwrap()
                            .0
                    });

            let return_ptr = unsafe { std::alloc::alloc(layout) };
            invoke_data.push(return_ptr as u64);

            Some((layout, return_ptr))
        } else {
            invoke_data.push(0);
            None
        };

        for (type_id, value) in function_signature.param_types.iter().zip(args) {
            map_arg_to_values(&mut invoke_data, &self.registry, type_id, value).unwrap();
        }

        // Invoke the trampoline.
        let mut ret_registers = [0; 4];
        unsafe {
            aot_trampoline(
                function_ptr.into_raw().into_raw(),
                invoke_data.as_ptr(),
                invoke_data.len(),
                &mut ret_registers,
            );
        }

        // Parse return values.
        let type_info = self
            .registry
            .get_type(function_signature.ret_types.last().unwrap())
            .unwrap();
        let return_value = match type_info {
            CoreTypeConcrete::Array(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => unreachable!("Array<T> is complex"),
            },
            CoreTypeConcrete::EcPoint(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => unreachable!("EcPoint is complex"),
            },
            CoreTypeConcrete::EcState(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => unreachable!("EcState is complex"),
            },
            CoreTypeConcrete::Enum(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
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
            CoreTypeConcrete::Struct(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => {
                    JitValue::Uint128(ret_registers[0] as u128 | (ret_registers[1] as u128) << 64)
                }
            },
            CoreTypeConcrete::Uint64(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => JitValue::Uint64(ret_registers[0]),
            },
            CoreTypeConcrete::Uint32(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => JitValue::Uint32(ret_registers[0] as u32),
            },
            CoreTypeConcrete::Uint16(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => JitValue::Uint16(ret_registers[0] as u16),
            },
            CoreTypeConcrete::Uint8(_) => match return_ptr {
                Some((_, return_ptr)) => JitValue::from_jit(
                    unsafe { NonNull::new_unchecked(return_ptr as *mut ()) },
                    function_signature.ret_types.last().unwrap(),
                    &self.registry,
                ),
                None => JitValue::Uint8(ret_registers[0] as u8),
            },
            _ => todo!("unsupported return type"),
        };

        if let Some((layout, return_ptr)) = return_ptr {
            unsafe {
                std::alloc::dealloc(return_ptr, layout);
            }
        }

        return_value
    }
}

fn map_arg_to_values(
    invoke_data: &mut Vec<u64>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_id: &ConcreteTypeId,
    value: &JitValue,
) -> Result<(), Box<ProgramRegistryError>> {
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

            let ptr = unsafe { libc::realloc(null_mut(), type_layout.size() * values.len()) };

            let bump = Bump::new();
            for (idx, value) in values.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        value
                            .to_jit(&bump, program_registry, &info.ty)
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
            let default_variant_idx = info
                .variants
                .iter()
                .map(|type_id| program_registry.get_type(type_id).unwrap())
                .enumerate()
                .fold((0, 0), |acc, (i, ty)| {
                    let ty_align = ty.layout(program_registry).unwrap().align();
                    if ty_align > acc.1 {
                        (i, ty_align)
                    } else {
                        acc
                    }
                })
                .0;

            let (layout, tag_layout, payload_layouts) =
                get_layout_for_variants(program_registry, &info.variants).unwrap();

            let (partial_layout, payload_offset) = tag_layout
                .extend(payload_layouts[default_variant_idx])
                .unwrap();

            // If the payload is the default variant, then the conversion is straighforward. If not,
            // hell's gates will have to be opened.
            if *tag == default_variant_idx {
                // Insert tag.
                invoke_data.push(*tag as u64);

                // Insert pre-payload padding.
                invoke_data.extend(repeat(0).take(payload_offset - tag_layout.size()));

                // Insert payload.
                map_arg_to_values(
                    invoke_data,
                    program_registry,
                    &info.variants[default_variant_idx],
                    value,
                )?;

                // Insert post-payload padding.
                invoke_data.extend(repeat(0).take(layout.size() - partial_layout.size()));
            } else {
                let mut binary_data = Vec::new();

                // Insert pre-payload padding. Since the payload alignment is less or equal to the
                // default branch, we can just insert bytes.
                invoke_data.extend(repeat(0).take(payload_offset - tag_layout.size()));

                // Write the payload into a byte array.
                map_arg_to_bytes(
                    &mut binary_data,
                    program_registry,
                    &info.variants[default_variant_idx],
                    value,
                )?;

                // Load arguments from the byte array into invoke_args.
                let mut cursor = Cursor::new(binary_data);
                map_bytes_to_values(invoke_data, program_registry, &[], &mut cursor)?;
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
                map_arg_to_values(invoke_data, program_registry, field_type_id, field_value)?;
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

fn map_arg_to_bytes(
    binary_data: &mut Vec<u8>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_id: &ConcreteTypeId,
    value: &JitValue,
) -> Result<(), Box<ProgramRegistryError>> {
    let type_info = program_registry.get_type(type_id)?;

    match (type_info, value) {
        (CoreTypeConcrete::Array(_), JitValue::Array(_values)) => todo!(),
        (CoreTypeConcrete::EcPoint(_), JitValue::EcPoint(a, b)) => {
            if binary_data.len() & (get_integer_layout(252).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(252))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(a.to_bytes_le());
            binary_data.extend(b.to_bytes_le());
        }
        (CoreTypeConcrete::EcState(_), JitValue::EcState(a, b, c, d)) => {
            if binary_data.len() & (get_integer_layout(252).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(252))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(a.to_bytes_le());
            binary_data.extend(b.to_bytes_le());
            binary_data.extend(c.to_bytes_le());
            binary_data.extend(d.to_bytes_le());
        }
        (CoreTypeConcrete::Enum(_info), JitValue::Enum { .. }) => todo!(),
        (CoreTypeConcrete::Felt252(_), JitValue::Felt252(value)) => {
            if binary_data.len() & (get_integer_layout(252).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(252))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_bytes_le());
        }
        (CoreTypeConcrete::Felt252Dict(_), JitValue::Felt252Dict { .. }) => todo!(),
        (CoreTypeConcrete::Struct(_info), JitValue::Struct { .. }) => todo!(),
        (CoreTypeConcrete::Uint128(_), JitValue::Uint128(value)) => {
            if binary_data.len() & (get_integer_layout(128).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(128))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_le_bytes());
        }
        (CoreTypeConcrete::Uint64(_), JitValue::Uint64(value)) => {
            if binary_data.len() & (get_integer_layout(64).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(64))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_le_bytes());
        }
        (CoreTypeConcrete::Uint32(_), JitValue::Uint32(value)) => {
            if binary_data.len() & (get_integer_layout(32).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(32))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_le_bytes());
        }
        (CoreTypeConcrete::Uint16(_), JitValue::Uint16(value)) => {
            if binary_data.len() & (get_integer_layout(16).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(16))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_le_bytes());
        }
        (CoreTypeConcrete::Uint8(_), JitValue::Uint8(value)) => {
            if binary_data.len() & (get_integer_layout(8).align() - 1) != 0 {
                binary_data.resize(
                    Layout::from_size_align(binary_data.len(), 16)
                        .unwrap()
                        .extend(get_integer_layout(8))
                        .unwrap()
                        .1,
                    0,
                );
            }

            binary_data.extend(value.to_le_bytes());
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

    Ok(())
}

#[allow(dead_code)]
enum EnumTypeId<'a> {
    Padding(usize),
    Payload(&'a ConcreteTypeId),
}

fn map_bytes_to_values(
    invoke_data: &mut Vec<u64>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_ids: &[EnumTypeId],
    value: &mut Cursor<Vec<u8>>,
) -> Result<(), Box<ProgramRegistryError>> {
    for type_id in type_ids {
        match type_id {
            EnumTypeId::Padding(len) => {
                for _ in 0..*len {
                    let mut bytes = [0; 1];
                    value.read_exact(&mut bytes).unwrap();
                    invoke_data.push(u8::from_le_bytes(bytes) as u64);
                }
            }
            EnumTypeId::Payload(type_id) => {
                let type_info = program_registry.get_type(type_id)?;

                // TODO: Alignment?
                match type_info {
                    CoreTypeConcrete::Array(_) => todo!(),
                    CoreTypeConcrete::EcPoint(_) => {
                        let mut bytes = [0; 32];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                    }
                    CoreTypeConcrete::EcState(_) => {
                        let mut bytes = [0; 32];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                    }
                    CoreTypeConcrete::Enum(_) => todo!(),
                    CoreTypeConcrete::Felt252(_) => {
                        let mut bytes = [0; 32];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.extend(Felt::from_bytes_le(&bytes).to_le_digits());
                    }
                    CoreTypeConcrete::Felt252Dict(_) => todo!(),
                    CoreTypeConcrete::Struct(_) => todo!(),
                    CoreTypeConcrete::Uint128(_) => {
                        let mut bytes = [0; 16];
                        value.read_exact(&mut bytes).unwrap();

                        let value = u128::from_le_bytes(bytes);
                        invoke_data.push((value >> 64) as u64);
                        invoke_data.push(value as u64);
                    }
                    CoreTypeConcrete::Uint64(_) => {
                        let mut bytes = [0; 8];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.push(u64::from_le_bytes(bytes));
                    }
                    CoreTypeConcrete::Uint32(_) => {
                        let mut bytes = [0; 4];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.push(u32::from_le_bytes(bytes) as u64);
                    }
                    CoreTypeConcrete::Uint16(_) => {
                        let mut bytes = [0; 2];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.push(u16::from_le_bytes(bytes) as u64);
                    }
                    CoreTypeConcrete::Uint8(_) => {
                        let mut bytes = [0; 1];
                        value.read_exact(&mut bytes).unwrap();
                        invoke_data.push(u8::from_le_bytes(bytes) as u64);
                    }
                    _ => todo!(),
                }
            }
        };
    }

    Ok(())
}
