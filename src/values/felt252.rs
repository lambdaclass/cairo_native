use super::ValueBuilder;
use crate::types::felt252::PRIME;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use num_bigint::BigUint;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Number;
use std::{alloc::Layout, fmt, ptr::NonNull, str::FromStr};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
    deserializer: D,
    arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &InfoOnlyConcreteType,
) -> Result<NonNull<()>, D::Error>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: ValueBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
{
    let ptr = arena
        .alloc_layout(Layout::from_size_align(32, 8).unwrap())
        .cast();

    let value = <Number as Deserialize>::deserialize(deserializer)?;
    let value: BigUint = value.to_string().parse().unwrap();
    assert!(value < *PRIME);

    let mut bytes = value.to_bytes_le();
    bytes.resize(32, 0);
    ptr.cast::<[u8; 32]>().as_mut().copy_from_slice(&bytes);

    Ok(ptr)
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
    let value = BigUint::from_bytes_le(ptr.cast::<[u8; 32]>().as_ref());
    <Number as Serialize>::serialize(&Number::from_str(&value.to_string()).unwrap(), serializer)
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
    fmt::Debug::fmt(&BigUint::from_bytes_le(ptr.cast::<[u8; 32]>().as_ref()), f)
}
