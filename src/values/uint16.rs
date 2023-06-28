use super::ValueBuilder;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Number;
use std::{fmt, ptr::NonNull, str::FromStr};

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
    let value = <Number as Deserialize>::deserialize(deserializer)?;
    let value: u16 = value.to_string().parse().unwrap();
    std::ptr::write(ptr.cast::<u16>().as_mut(), value);
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
    let value = ptr.cast::<u16>().as_ref();
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
    fmt::Debug::fmt(ptr.cast::<u16>().as_ref(), f)
}
