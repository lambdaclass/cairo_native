use super::ValueBuilder;
use crate::types::TypeBuilder;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ptr::NonNull;

pub unsafe fn deserialize_address<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    registry: &ProgramRegistry<TType, TLibfunc>,
    info: &InfoOnlyConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    super::felt252::deserialize(deserializer, arena, registry, info)
}

pub unsafe fn serialize_address<TType, TLibfunc, S>(
    serializer: S,
    registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    info: &InfoOnlyConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    super::felt252::serialize(serializer, registry, ptr, info)
}

pub unsafe fn deserialize_system<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &InfoOnlyConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    <u64 as Deserialize>::deserialize(deserializer)
        .map(|x| arena.alloc(NonNull::new(x as *mut ()).unwrap()).cast())
}

pub unsafe fn serialize_system<TType, TLibfunc, S>(
    serializer: S,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    <u64 as Serialize>::serialize(ptr.cast::<u64>().as_ref(), serializer)
}
