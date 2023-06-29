use super::{ValueBuilder, ValueSerializer};
use crate::types::TypeBuilder;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{enm::EnumConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{ser::SerializeSeq, Deserializer, Serializer};
use std::{fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    _deserializer: D,
    _arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &EnumConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    todo!()
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &EnumConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let tag_layout = crate::utils::get_integer_layout(
        (info.variants.len().next_power_of_two().next_multiple_of(8) >> 3)
            .try_into()
            .unwrap(),
    );
    let tag_value = match tag_layout.size() {
        1 => *ptr.cast::<u8>().as_ref() as usize,
        2 => *ptr.cast::<u16>().as_ref() as usize,
        4 => *ptr.cast::<u32>().as_ref() as usize,
        8 => *ptr.cast::<u64>().as_ref() as usize,
        _ => unreachable!(),
    };

    let payload_ty = registry.get_type(&info.variants[tag_value]).unwrap();
    let payload_layout = payload_ty.layout(registry);

    type ParamSerializer<'a, TType, TLibfunc> =
        <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

    let mut ser = serializer.serialize_seq(Some(2))?;
    ser.serialize_element(&tag_value)?;
    ser.serialize_element(&ParamSerializer::<TType, TLibfunc>::new(
        ptr.map_addr(|addr| addr.unchecked_add(tag_layout.extend(payload_layout).unwrap().1)),
        registry,
        payload_ty,
    ))?;
    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    _f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &EnumConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    todo!()
}
