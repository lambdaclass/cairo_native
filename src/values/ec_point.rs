use super::ValueBuilder;
use crate::{
    types::{felt252::PRIME, TypeBuilder},
    utils::{get_integer_layout, layout_repeat},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use num_bigint::BigUint;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ptr::NonNull;

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
    let ptr = arena
        .alloc_layout(layout_repeat(&get_integer_layout(252), 2).unwrap().0)
        .cast();

    let data = <[[u32; 8]; 2] as Deserialize>::deserialize(deserializer)?;
    assert!(BigUint::new(data[0].to_vec()) < *PRIME);
    assert!(BigUint::new(data[1].to_vec()) < *PRIME);

    ptr.cast::<[[u32; 8]; 2]>().as_mut().copy_from_slice(&data);
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
    ptr.cast::<[[u32; 8]; 2]>().as_ref().serialize(serializer)
}
