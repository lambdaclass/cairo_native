use crate::types::TypeBuilder;

use super::ValueBuilder;
use crate::values::ValueSerializer;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{ser::SerializeSeq, Deserializer, Serializer};
use std::{alloc::Layout, fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    _deserializer: D,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &InfoAndTypeConcreteType,
) -> Result<(), D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    todo!();
    /*
    let payload_ty = registry.get_type(&info.info.).unwrap();
    let payload_layout = payload_ty.layout(registry);

    let value = <Number as Deserialize>::deserialize(deserializer)?;
    let value: u32 = value.to_string().parse().unwrap();
    std::ptr::write(ptr.cast::<u32>().as_mut(), value);
    */
    // Ok(())
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoAndTypeConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let elem_ty = registry.get_type(&info.ty).unwrap();

    let elem_layout = elem_ty.layout(registry);
    let elem_stride = elem_layout.pad_to_align().size();

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let ptr_layout = Layout::new::<*mut ()>();
    let len_layout = crate::utils::get_integer_layout(32);

    let len_value = *ptr
        .map_addr(|addr| addr.unchecked_add(ptr_layout.extend(len_layout).unwrap().1))
        .cast::<u32>()
        .as_ref();

    let data_ptr = *ptr.cast::<NonNull<()>>().as_ref();

    let mut ser = serializer.serialize_seq(Some(len_value.try_into().unwrap()))?;
    for i in 0..(len_value as usize) {
        let cur_elem_ptr = data_ptr.map_addr(|addr| addr.unchecked_add(elem_stride * i));

        ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
            cur_elem_ptr.cast(),
            registry,
            elem_ty,
        ))?;
    }
    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    _f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &InfoAndTypeConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    todo!()
}
