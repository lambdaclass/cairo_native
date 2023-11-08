use super::ValueBuilder;
use crate::types::TypeBuilder;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{alloc::Layout, fmt, ptr::NonNull};

pub unsafe fn deserialize<'de, TType, TLibfunc, D>(
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
    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();

    let value = u8::deserialize(deserializer)?;
    *ptr.cast::<u8>().as_mut() = value;

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
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    S: Serializer,
{
    ptr.cast::<u8>().as_ref().serialize(serializer)
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
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    fmt::Debug::fmt(ptr.cast::<u8>().as_ref(), f)
}

pub unsafe fn from_rust_to_jit<TType, TLibfunc>(
    arena: &Bump,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _info: &InfoOnlyConcreteType,
    value: u8,
) -> NonNull<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
    *ptr.cast::<u8>().as_mut() = value;

    ptr
}

pub unsafe fn from_jit_to_rust<TType, TLibfunc>(
    _registry: &ProgramRegistry<TType, TLibfunc>,
    ptr: NonNull<()>,
    _info: &InfoOnlyConcreteType,
) -> u8
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
{
    *ptr.cast::<u8>().as_ref()
}
