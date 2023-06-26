use super::ValueBuilder;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry, ids::ConcreteTypeId,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{ptr::NonNull, fmt};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> Result<(), D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    *ptr.cast::<u64>().as_mut() = <u64 as Deserialize>::deserialize(deserializer)?;

    Ok(())
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    <u64 as Serialize>::serialize(ptr.cast::<u64>().as_ref(), serializer)
}

pub unsafe fn debug_fmt<TType, TLibfunc>(
    f: &mut fmt::Formatter,
    _id: &ConcreteTypeId,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> fmt::Result
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
{
    fmt::Debug::fmt(ptr.cast::<u64>().as_ref(), f)
}
