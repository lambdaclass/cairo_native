use super::ValueBuilder;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ptr::NonNull;

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> Result<(), D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    <() as Deserialize>::deserialize(deserializer)
}

pub unsafe fn serialize<TType, TLibfunc, S>(
    serializer: S,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> Result<S::Ok, S::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    <() as Serialize>::serialize(&(), serializer)
}
