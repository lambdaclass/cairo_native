use super::{ValueBuilder, ValueSerializer};
use crate::{types::TypeBuilder, utils::debug_with};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{ser::SerializeMap, Deserializer, Serializer};
use std::{collections::HashMap, fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    _deserializer: D,
    _arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &InfoAndTypeConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    todo!()
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
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    let ptr = ptr.cast::<HashMap<[u8; 32], NonNull<()>>>();
    let map = Box::from_raw(ptr.as_ptr());
    let target_type = registry.get_type(&info.ty).unwrap();

    let mut ser = serializer.serialize_map(Some(map.len()))?;

    for (key, val_ptr) in map.iter() {
        type ParamSerializer<'a, TType, TLibfunc> =
            <<TType as GenericType>::Concrete as ValueBuilder<TType, TLibfunc>>::Serializer<'a>;

        ser.serialize_entry(
            key,
            &ParamSerializer::<TType, TLibfunc>::new(*val_ptr, registry, target_type),
        )?;
    }

    ser.end()
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoAndTypeConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    let mut fmt = f.debug_map();
    let ptr = ptr.cast::<HashMap<[u8; 32], NonNull<()>>>();
    let map = Box::from_raw(ptr.as_ptr());
    let target_type = registry.get_type(&info.ty).unwrap();

    for (key, val_ptr) in map.iter() {
        fmt.entry(
            key,
            &debug_with(|f| target_type.debug_fmt(f, &info.ty, registry, *val_ptr)),
        );
    }

    fmt.finish()
}
