use crate::types::TypeBuilder;
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        GenericLibfunc, GenericType,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use serde::{de::DeserializeSeed, Serialize};
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

pub trait ValueBuilder<TType, TLibfunc>
where
    Self: TypeBuilder,
    TType: GenericType<Concrete = Self>,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    type Deserializer<'a>: ValueDeserializer<'a, TType, TLibfunc>;
    type Serializer<'a>: ValueSerializer<'a, TType, TLibfunc>;

    type Error: Error;

    fn is_complex(&self) -> bool;

    unsafe fn debug_fmt(
        &self,
        f: &mut fmt::Formatter,
        id: &ConcreteTypeId,
        registry: &ProgramRegistry<TType, TLibfunc>,
        ptr: NonNull<()>,
    ) -> fmt::Result;
}

pub trait ValueDeserializer<'a, TType, TLibfunc>
where
    Self: for<'de> DeserializeSeed<'de, Value = NonNull<()>>,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a <TType as GenericType>::Concrete,
    ) -> Self;
}

pub trait ValueSerializer<'a, TType, TLibfunc>
where
    Self: Serialize,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    fn new(
        ptr: NonNull<()>,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a <TType as GenericType>::Concrete,
    ) -> Self;
}

impl ValueBuilder<CoreType, CoreLibfunc> for CoreTypeConcrete {
    type Deserializer<'a> = CoreTypeDeserializer<'a, CoreType, CoreLibfunc>;
    type Serializer<'a> = CoreTypeSerializer<'a, CoreType, CoreLibfunc>;

    type Error = std::convert::Infallible;

    fn is_complex(&self) -> bool {
        match self {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => false,
            CoreTypeConcrete::GasBuiltin(_) => false,
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => false,
            CoreTypeConcrete::Uint16(_) => false,
            CoreTypeConcrete::Uint32(_) => false,
            CoreTypeConcrete::Uint64(_) => false,
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => false,
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => true,
            CoreTypeConcrete::Struct(_) => true,
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

    unsafe fn debug_fmt(
        &self,
        f: &mut fmt::Formatter,
        id: &ConcreteTypeId,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        ptr: NonNull<()>,
    ) -> fmt::Result {
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
            CoreTypeConcrete::Uint8(info) => self::uint8::debug_fmt(f, id, registry, ptr, info),
            CoreTypeConcrete::Uint16(info) => self::uint16::debug_fmt(f, id, registry, ptr, info),
            CoreTypeConcrete::Uint32(info) => self::uint32::debug_fmt(f, id, registry, ptr, info),
            CoreTypeConcrete::Uint64(info) => self::uint64::debug_fmt(f, id, registry, ptr, info),
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

pub struct CoreTypeDeserializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    arena: &'a Bump,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a <TType as GenericType>::Concrete,
}

impl<'a> ValueDeserializer<'a, CoreType, CoreLibfunc>
    for CoreTypeDeserializer<'a, CoreType, CoreLibfunc>
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
        info: &'a CoreTypeConcrete,
    ) -> Self {
        Self {
            arena,
            registry,
            info,
        }
    }
}

impl<'a, 'de> DeserializeSeed<'de> for CoreTypeDeserializer<'a, CoreType, CoreLibfunc> {
    type Value = NonNull<()>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ptr = self
            .arena
            .alloc_layout(self.info.layout(self.registry))
            .cast::<()>();
        match self.info {
            CoreTypeConcrete::Array(info) => unsafe {
                self::array::deserialize(deserializer, self.registry, ptr, info)?
            },
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(info) => unsafe {
                self::felt252::deserialize(deserializer, self.registry, ptr, info)?
            },
            CoreTypeConcrete::GasBuiltin(info) => unsafe {
                self::gas_builtin::deserialize(deserializer, self.registry, ptr, info)?
            },
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(info) => {
                unsafe { self::uint8::deserialize(deserializer, self.registry, ptr, info) }?
            }
            CoreTypeConcrete::Uint16(info) => {
                unsafe { self::uint16::deserialize(deserializer, self.registry, ptr, info) }?
            }
            CoreTypeConcrete::Uint32(info) => {
                unsafe { self::uint32::deserialize(deserializer, self.registry, ptr, info) }?
            }
            CoreTypeConcrete::Uint64(info) => {
                unsafe { self::uint64::deserialize(deserializer, self.registry, ptr, info) }?
            }
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(info) => unsafe {
                self::range_check::deserialize(deserializer, self.registry, ptr, info)?
            },
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => todo!(),
            CoreTypeConcrete::Struct(_) => todo!(),
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

        Ok(ptr)
    }
}

pub struct CoreTypeSerializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    ptr: NonNull<()>,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a <TType as GenericType>::Concrete,
}

impl<'a> ValueSerializer<'a, CoreType, CoreLibfunc>
    for CoreTypeSerializer<'a, CoreType, CoreLibfunc>
{
    fn new(
        ptr: NonNull<()>,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
        info: &'a <CoreType as GenericType>::Concrete,
    ) -> Self {
        Self {
            ptr,
            registry,
            info,
        }
    }
}

impl<'a> Serialize for CoreTypeSerializer<'a, CoreType, CoreLibfunc> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.info {
            CoreTypeConcrete::Array(info) => unsafe {
                self::array::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(info) => unsafe {
                self::felt252::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::GasBuiltin(info) => unsafe {
                self::gas_builtin::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(info) => unsafe {
                self::uint8::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uint16(info) => unsafe {
                self::uint16::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uint32(info) => unsafe {
                self::uint32::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uint64(info) => unsafe {
                self::uint64::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(info) => unsafe {
                self::range_check::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(info) => unsafe {
                self::r#enum::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Struct(info) => unsafe {
                self::r#struct::serialize(serializer, self.registry, self.ptr, info)
            },
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
