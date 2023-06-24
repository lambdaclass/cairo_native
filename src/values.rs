use crate::types::TypeBuilder;
use cairo_lang_sierra::{
    extensions::{core::CoreTypeConcrete, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt, ptr::NonNull};

pub mod array;
pub mod bitwise;
pub mod r#box;
pub mod builtin_costs;
pub mod ec_op;
pub mod ec_point;
pub mod ec_state;
pub mod r#enum;
pub mod felt252;
pub mod felt252_dict;
pub mod felt252_dict_entry;
pub mod gas_builtin;
pub mod non_zero;
pub mod nullable;
pub mod pedersen;
pub mod poseidon;
pub mod range_check;
pub mod segment_arena;
pub mod snapshot;
pub mod squashed_felt252_dict;
pub mod stark_net;
pub mod r#struct;
pub mod uint128;
pub mod uint128_mul_guarantee;
pub mod uint16;
pub mod uint32;
pub mod uint64;
pub mod uint8;
pub mod uninitialized;

pub trait ValueBuilder
where
    Self: TypeBuilder,
{
    type Error: Error;

    unsafe fn deserialize<'de, T>(
        &self,
        de: impl Deserialize<'de>,
        ptr: NonNull<()>,
    ) -> Result<(), <Self as ValueBuilder>::Error>;
    unsafe fn serialize(
        &self,
        ser: impl Serialize,
        ptr: NonNull<()>,
    ) -> Result<(), <Self as ValueBuilder>::Error>;

    unsafe fn debug_fmt<TType, TLibfunc>(
        &self,
        f: &mut fmt::Formatter,
        id: &ConcreteTypeId,
        registry: &ProgramRegistry<TType, TLibfunc>,
        ptr: NonNull<()>,
    ) -> fmt::Result
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder;
}

impl ValueBuilder for CoreTypeConcrete {
    type Error = std::convert::Infallible;

    unsafe fn deserialize<'de, T>(
        &self,
        _de: impl Deserialize<'de>,
        _ptr: NonNull<()>,
    ) -> Result<(), <Self as ValueBuilder>::Error> {
        todo!()
    }

    unsafe fn serialize(
        &self,
        _ser: impl Serialize,
        _ptr: NonNull<()>,
    ) -> Result<(), <Self as ValueBuilder>::Error> {
        todo!()
    }

    unsafe fn debug_fmt<TType, TLibfunc>(
        &self,
        f: &mut fmt::Formatter,
        id: &ConcreteTypeId,
        registry: &ProgramRegistry<TType, TLibfunc>,
        ptr: NonNull<()>,
    ) -> fmt::Result
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: ValueBuilder,
    {
        match self {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(info) => self::felt252::debug_fmt(f, id, registry, ptr, info),
            CoreTypeConcrete::GasBuiltin(_) => todo!(),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => todo!(),
            CoreTypeConcrete::Struct(info) => self::r#struct::debug_fmt(f, id, registry, ptr, info),
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }
    }
}
