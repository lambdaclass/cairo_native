use crate::{
    types::TypeBuilder,
    utils::generate_function_name,
    values::{JITValue, ValueBuilder},
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
use std::{alloc::Layout, arch::global_asm, ptr::null_mut};

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));

// #[cfg(target_arch = "aarch64")]
// const NUM_REGISTER_ARGS: usize = 8;
// #[cfg(target_arch = "x86_64")]
// const NUM_REGISTER_ARGS: usize = 6;

extern "C" {
    fn aot_trampoline(fn_ptr: *mut c_void, args_ptr: *const u64, args_len: usize);
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
    pub fn invoke_dynamic(&self, function_id: &FunctionId, args: &[JITValue]) -> Vec<JITValue> {
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
                .first()
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

        // TODO: Invoke the trampoline.
        unsafe {
            aot_trampoline(
                function_ptr.into_raw().into_raw(),
                invoke_data.as_ptr(),
                invoke_data.len(),
            );
        }

        if let Some((layout, return_ptr)) = return_ptr {
            unsafe {
                std::alloc::dealloc(return_ptr, layout);
            }
        }

        vec![]
    }
}

fn map_arg_to_values(
    invoke_data: &mut Vec<u64>,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    type_id: &ConcreteTypeId,
    value: &JITValue,
) -> Result<(), Box<ProgramRegistryError>> {
    let type_info = program_registry.get_type(type_id)?;

    // TODO: Find out if builtins push an argument or not. My guess is that they do.
    match (type_info, value) {
        (CoreTypeConcrete::Array(info), JITValue::Array(values)) => {
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
        (CoreTypeConcrete::EcPoint(_), JITValue::EcPoint(a, b)) => {
            invoke_data.extend(a.to_le_digits());
            invoke_data.extend(b.to_le_digits());
        }
        (CoreTypeConcrete::EcState(_), JITValue::EcState(a, b, c, d)) => {
            invoke_data.extend(a.to_le_digits());
            invoke_data.extend(b.to_le_digits());
            invoke_data.extend(c.to_le_digits());
            invoke_data.extend(d.to_le_digits());
        }
        (CoreTypeConcrete::Enum(info), JITValue::Enum { tag, value, .. }) => {
            // TODO: Why does `MyEnum::A` require its payload on the stack but `MyEnum::B` is parsed
            //   on the next available register? Why does `MyEnum::A`'s MSB get replaced with 0xEF?
            invoke_data.push(*tag as u64);
            // if invoke_data.len() < NUM_REGISTER_ARGS {
            //     invoke_data.resize(NUM_REGISTER_ARGS, 0);
            // }
            map_arg_to_values(invoke_data, program_registry, &info.variants[*tag], value)?;
        }
        (CoreTypeConcrete::Felt252(_), JITValue::Felt252(value)) => {
            invoke_data.extend(value.to_le_digits());
        }
        (CoreTypeConcrete::Felt252Dict(_), JITValue::Felt252Dict { .. }) => {
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
        (CoreTypeConcrete::Struct(info), JITValue::Struct { fields, .. }) => {
            for (field_type_id, field_value) in info.members.iter().zip(fields) {
                map_arg_to_values(invoke_data, program_registry, field_type_id, field_value)?;
            }
        }
        (CoreTypeConcrete::Uint128(_), JITValue::Uint128(value)) => {
            invoke_data.push(*value as u64);
            invoke_data.push((value >> 64) as u64);
        }
        (CoreTypeConcrete::Uint64(_), JITValue::Uint64(value)) => {
            invoke_data.push(*value);
        }
        (CoreTypeConcrete::Uint32(_), JITValue::Uint32(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Uint16(_), JITValue::Uint16(value)) => {
            invoke_data.push(*value as u64);
        }
        (CoreTypeConcrete::Uint8(_), JITValue::Uint8(value)) => {
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
