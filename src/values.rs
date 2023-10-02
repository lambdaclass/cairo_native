//! # JIT params and return values de/serialization

use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
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

/// The [`ValueBuilder`] trait is implemented any de/serializable value, which is the `TType`
/// generic.
pub trait ValueBuilder<TType, TLibfunc>
where
    TType: GenericType<Concrete = Self>,
    TLibfunc: GenericLibfunc,
{
    /// Value deserializer from [serde] into the JIT ABI.
    type Deserializer<'a>: ValueDeserializer<'a, TType, TLibfunc>;
    /// Value serializer from the JIT ABI into [serde].
    type Serializer<'a>: ValueSerializer<'a, TType, TLibfunc>;

    /// Error type returned from the de/serializers.
    type Error: Error;

    /// Return whether the type is considered complex or simple.
    ///
    /// Complex types are always passed by pointer (both as params and return values) and require a
    /// stack allocation. Examples of complex values include structs and enums, but not felts since
    /// LLVM considers them integers.
    fn is_complex(&self) -> bool;

    /// Write a representation of the argument in a human-friendly format for debugging.
    unsafe fn debug_fmt(
        &self,
        f: &mut fmt::Formatter,
        id: &ConcreteTypeId,
        registry: &ProgramRegistry<TType, TLibfunc>,
        ptr: NonNull<()>,
    ) -> fmt::Result;
}

/// Deserialize a value from [serde] into the JIT ABI.
pub trait ValueDeserializer<'a, TType, TLibfunc>
where
    Self: for<'de> DeserializeSeed<'de, Value = NonNull<()>>,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    fn new(
        arena: &'a Bump,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a <TType as GenericType>::Concrete,
    ) -> Self;
}

/// Serialize a value from the JIT ABI into [serde].
pub trait ValueSerializer<'a, TType, TLibfunc>
where
    Self: Serialize,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
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
            CoreTypeConcrete::Array(_) => true,
            CoreTypeConcrete::Bitwise(_) => false,
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => false,
            CoreTypeConcrete::EcPoint(_) => true,
            CoreTypeConcrete::EcState(_) => true,
            CoreTypeConcrete::Felt252(_) => false,
            CoreTypeConcrete::GasBuiltin(_) => false,
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => false,
            CoreTypeConcrete::Uint16(_) => false,
            CoreTypeConcrete::Uint32(_) => false,
            CoreTypeConcrete::Uint64(_) => false,
            CoreTypeConcrete::Uint128(_) => false,
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => false,
            CoreTypeConcrete::RangeCheck(_) => false,
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => true,
            CoreTypeConcrete::Struct(_) => true,
            CoreTypeConcrete::Felt252Dict(_) => false,
            CoreTypeConcrete::Felt252DictEntry(_) => true,
            CoreTypeConcrete::SquashedFelt252Dict(_) => false,
            CoreTypeConcrete::Pedersen(_) => false,
            CoreTypeConcrete::Poseidon(_) => false,
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::ClassHash(_) => false,
                StarkNetTypeConcrete::ContractAddress(_) => false,
                StarkNetTypeConcrete::StorageBaseAddress(_) => false,
                StarkNetTypeConcrete::StorageAddress(_) => false,
                StarkNetTypeConcrete::System(_) => false,
                StarkNetTypeConcrete::Secp256Point(_) => todo!(),
            },
            CoreTypeConcrete::SegmentArena(_) => false,
            CoreTypeConcrete::Snapshot(_) => false,
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }

    unsafe fn debug_fmt(
        &self,
        _f: &mut fmt::Formatter,
        _id: &ConcreteTypeId,
        _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        _ptr: NonNull<()>,
    ) -> fmt::Result {
        todo!()
        // match self {
        //     CoreTypeConcrete::Array(_) => todo!(),
        //     CoreTypeConcrete::Bitwise(_) => todo!(),
        //     CoreTypeConcrete::Box(_) => todo!(),
        //     CoreTypeConcrete::EcOp(_) => todo!(),
        //     CoreTypeConcrete::EcPoint(_) => todo!(),
        //     CoreTypeConcrete::EcState(_) => todo!(),
        //     CoreTypeConcrete::Felt252(info) => self::felt252::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::GasBuiltin(_) => todo!(),
        //     CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        //     CoreTypeConcrete::Uint8(info) => self::uint8::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::Uint16(info) => self::uint16::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::Uint32(info) => self::uint32::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::Uint64(info) => self::uint64::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::Uint128(_) => todo!(),
        //     CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        //     CoreTypeConcrete::NonZero(_) => todo!(),
        //     CoreTypeConcrete::Nullable(_) => todo!(),
        //     CoreTypeConcrete::RangeCheck(_) => todo!(),
        //     CoreTypeConcrete::Uninitialized(_) => todo!(),
        //     CoreTypeConcrete::Enum(_) => todo!(),
        //     CoreTypeConcrete::Struct(info) => self::r#struct::debug_fmt(f, id, registry, ptr, info),
        //     CoreTypeConcrete::Felt252Dict(_) => todo!(),
        //     CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
        //     CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
        //     CoreTypeConcrete::Pedersen(_) => todo!(),
        //     CoreTypeConcrete::Poseidon(_) => todo!(),
        //     CoreTypeConcrete::Span(_) => todo!(),
        //     CoreTypeConcrete::StarkNet(_) => todo!(),
        //     CoreTypeConcrete::SegmentArena(_) => todo!(),
        //     CoreTypeConcrete::Snapshot(_) => todo!(),
        // }
    }
}

/// Deserializer for Cairo's [`CoreType`].
pub struct CoreTypeDeserializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
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
        unsafe {
            match self.info {
                CoreTypeConcrete::Array(info) => {
                    self::array::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Bitwise(info) => {
                    self::bitwise::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Box(_) => todo!(),
                CoreTypeConcrete::EcOp(info) => {
                    self::ec_op::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::EcPoint(info) => {
                    self::ec_point::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::EcState(info) => {
                    self::ec_state::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Felt252(info) => {
                    self::felt252::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::GasBuiltin(info) => {
                    self::gas_builtin::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::BuiltinCosts(_) => todo!(),
                CoreTypeConcrete::Uint8(info) => {
                    self::uint8::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uint16(info) => {
                    self::uint16::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uint32(info) => {
                    self::uint32::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uint64(info) => {
                    self::uint64::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uint128(info) => {
                    self::uint128::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
                CoreTypeConcrete::NonZero(info) => {
                    self::non_zero::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Nullable(info) => {
                    self::nullable::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::RangeCheck(info) => {
                    self::range_check::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Uninitialized(_) => todo!(),
                CoreTypeConcrete::Enum(info) => {
                    self::r#enum::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Struct(info) => {
                    self::r#struct::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Felt252Dict(info) => {
                    self::felt252_dict::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
                CoreTypeConcrete::SquashedFelt252Dict(info) => {
                    self::squashed_felt252_dict::deserialize(
                        deserializer,
                        self.arena,
                        self.registry,
                        info,
                    )
                }
                CoreTypeConcrete::Pedersen(info) => {
                    self::pedersen::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Poseidon(info) => {
                    self::poseidon::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Span(_) => todo!(),
                CoreTypeConcrete::StarkNet(selector) => match selector {
                    StarkNetTypeConcrete::ClassHash(info) => self::stark_net::deserialize_address(
                        deserializer,
                        self.arena,
                        self.registry,
                        info,
                    ),
                    StarkNetTypeConcrete::ContractAddress(info) => {
                        self::stark_net::deserialize_address(
                            deserializer,
                            self.arena,
                            self.registry,
                            info,
                        )
                    }
                    StarkNetTypeConcrete::StorageBaseAddress(info) => {
                        self::stark_net::deserialize_address(
                            deserializer,
                            self.arena,
                            self.registry,
                            info,
                        )
                    }
                    StarkNetTypeConcrete::StorageAddress(info) => {
                        self::stark_net::deserialize_address(
                            deserializer,
                            self.arena,
                            self.registry,
                            info,
                        )
                    }
                    StarkNetTypeConcrete::System(info) => self::stark_net::deserialize_system(
                        deserializer,
                        self.arena,
                        self.registry,
                        info,
                    ),
                    StarkNetTypeConcrete::Secp256Point(_) => todo!(),
                },
                CoreTypeConcrete::SegmentArena(info) => {
                    self::segment_arena::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Snapshot(info) => {
                    self::snapshot::deserialize(deserializer, self.arena, self.registry, info)
                }
                CoreTypeConcrete::Sint8(_) => todo!(),
                CoreTypeConcrete::Sint16(_) => todo!(),
                CoreTypeConcrete::Sint32(_) => todo!(),
                CoreTypeConcrete::Sint64(_) => todo!(),
                CoreTypeConcrete::Sint128(_) => todo!(),
                CoreTypeConcrete::Bytes31(_) => todo!(),
            }
        }
    }
}

/// Serializer for Cairo's [`CoreType`].
pub struct CoreTypeSerializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
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
            CoreTypeConcrete::Bitwise(info) => unsafe {
                self::bitwise::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(info) => unsafe {
                self::ec_op::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::EcPoint(info) => unsafe {
                self::ec_point::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::EcState(info) => unsafe {
                self::ec_state::serialize(serializer, self.registry, self.ptr, info)
            },
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
            CoreTypeConcrete::Uint128(info) => unsafe {
                self::uint128::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(info) => unsafe {
                self::non_zero::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Nullable(info) => unsafe {
                self::nullable::serialize(serializer, self.registry, self.ptr, info)
            },
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
            CoreTypeConcrete::Felt252Dict(info) => unsafe {
                self::felt252_dict::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(info) => unsafe {
                self::squashed_felt252_dict::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Pedersen(info) => unsafe {
                self::pedersen::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Poseidon(info) => unsafe {
                self::poseidon::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::ClassHash(info) => unsafe {
                    self::stark_net::serialize_address(serializer, self.registry, self.ptr, info)
                },
                StarkNetTypeConcrete::ContractAddress(info) => unsafe {
                    self::stark_net::serialize_address(serializer, self.registry, self.ptr, info)
                },
                StarkNetTypeConcrete::StorageBaseAddress(info) => unsafe {
                    self::stark_net::serialize_address(serializer, self.registry, self.ptr, info)
                },
                StarkNetTypeConcrete::StorageAddress(info) => unsafe {
                    self::stark_net::serialize_address(serializer, self.registry, self.ptr, info)
                },
                StarkNetTypeConcrete::System(info) => unsafe {
                    self::stark_net::serialize_system(serializer, self.registry, self.ptr, info)
                },
                StarkNetTypeConcrete::Secp256Point(_) => todo!(),
            },
            CoreTypeConcrete::SegmentArena(info) => unsafe {
                self::segment_arena::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Snapshot(info) => unsafe {
                self::snapshot::serialize(serializer, self.registry, self.ptr, info)
            },
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }
}
