//! A Rusty interface to provide parameters to JIT calls.

use std::slice::Iter;

use bumpalo::Bump;
use cairo_felt::Felt252;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        enm::EnumConcreteType,
        types::InfoOnlyConcreteType,
        GenericLibfunc, GenericType,
    },
    ids::ConcreteTypeId,
    program::Function,
    program_registry::ProgramRegistry,
};
use num_bigint::{BigInt, BigUint};
use serde::Deserialize;
use serde::{
    de::{DeserializeSeed, Visitor},
    ser::SerializeSeq,
    Serialize,
};

use crate::{metadata::syscall_handler::SyscallHandlerMeta, utils::felt252_bigint};

#[derive(Debug, Clone)]
pub enum InvokeArg {
    Felt252(Felt252),
    Array(Vec<Self>),  // all elements need to be same type
    Struct(Vec<Self>), // element types can differ
    Span(Vec<Self>),   // like a array, used specially when passing parameters to contracts
    Enum { tag: u64, value: Box<Self> },
    Box(Box<Self>), // can't be null
    Nullable(Option<Box<Self>>),
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
}

#[derive(Debug, Default)]
pub struct InvokeContext<'s> {
    pub gas: Option<u128>,
    // Starknet syscall handler
    pub system: Option<&'s SyscallHandlerMeta>,
    pub bitwise: bool,
    pub range_check: bool,
    pub pedersen: bool,
    // call args
    pub args: Vec<InvokeArg>,
}

#[derive(Debug, Default)]
pub struct InvokeResult {
    pub gas: Option<u128>,
    pub outputs: Vec<InvokeArg>,
}

// Conversions

impl From<Felt252> for InvokeArg {
    fn from(value: Felt252) -> Self {
        InvokeArg::Felt252(value)
    }
}

impl From<u8> for InvokeArg {
    fn from(value: u8) -> Self {
        InvokeArg::Uint8(value)
    }
}

impl From<u16> for InvokeArg {
    fn from(value: u16) -> Self {
        InvokeArg::Uint16(value)
    }
}

impl From<u32> for InvokeArg {
    fn from(value: u32) -> Self {
        InvokeArg::Uint32(value)
    }
}

impl From<u64> for InvokeArg {
    fn from(value: u64) -> Self {
        InvokeArg::Uint64(value)
    }
}

impl From<u128> for InvokeArg {
    fn from(value: u128) -> Self {
        InvokeArg::Uint128(value)
    }
}

// Serialization

impl<'s> Serialize for InvokeContext<'s> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut params_seq = serializer.serialize_seq(None)?;

        // TODO: check order
        if self.bitwise {
            params_seq.serialize_element(&())?;
        }
        if self.range_check {
            params_seq.serialize_element(&())?;
        }
        if self.pedersen {
            params_seq.serialize_element(&())?;
        }

        if let Some(gas) = self.gas {
            params_seq.serialize_element(&gas)?;
        }

        if let Some(system) = self.system {
            params_seq.serialize_element(&(system.as_ptr().as_ptr() as usize))?;
        }

        for arg in &self.args {
            params_seq.serialize_element(arg)?;
        }

        params_seq.end()
    }
}

impl Serialize for InvokeArg {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            InvokeArg::Felt252(value) => {
                let value = felt252_bigint(value.to_bigint());
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in &value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Array(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Struct(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Span(value) => {
                InvokeArg::Struct(vec![InvokeArg::Array(value.clone())]).serialize(serializer)
            }
            InvokeArg::Enum { tag, value } => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(tag)?;
                seq.serialize_element(value.as_ref())?;
                seq.end()
            }
            InvokeArg::Box(value) => serializer.serialize_some(value.as_ref()),
            InvokeArg::Nullable(value) => match value {
                Some(value) => serializer.serialize_some(value.as_ref()),
                None => serializer.serialize_none(),
            },
            InvokeArg::Uint8(value) => serializer.serialize_u8(*value),
            InvokeArg::Uint16(value) => serializer.serialize_u16(*value),
            InvokeArg::Uint32(value) => serializer.serialize_u32(*value),
            InvokeArg::Uint64(value) => serializer.serialize_u64(*value),
            InvokeArg::Uint128(value) => serializer.serialize_u128(*value),
        }
    }
}

/// Deserialize a value from [serde] into InvokeArg.
pub trait InvokeValueDeserializer<'a, TType, TLibfunc>
where
    Self: for<'de> DeserializeSeed<'de, Value = InvokeArg>,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    fn new(
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a <TType as GenericType>::Concrete,
    ) -> Self;
}

/// Serialize a value from the InvokeArg into [serde].
pub trait InvokeValueSerializer<'a, TType, TLibfunc>
where
    Self: Serialize,
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    fn new(
        value: InvokeArg,
        registry: &'a ProgramRegistry<TType, TLibfunc>,
        info: &'a <TType as GenericType>::Concrete,
    ) -> Self;
}

/// The [`InvokeValueBuilder`] trait is implemented any de/serializable value, which is the `TType`
/// generic.
pub trait InvokeValueBuilder<TType, TLibfunc>
where
    TType: GenericType<Concrete = Self>,
    TLibfunc: GenericLibfunc,
{
    /// Value deserializer from [serde] into the JIT ABI.
    type Deserializer<'a>: InvokeValueDeserializer<'a, TType, TLibfunc>;
    /// Value serializer from the JIT ABI into [serde].
    type Serializer<'a>: InvokeValueSerializer<'a, TType, TLibfunc>;

    /// Error type returned from the de/serializers.
    type Error: std::error::Error;
}

/// Deserializer for Cairo's [`CoreType`].
pub struct InvokeCoreTypeDeserializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a <TType as GenericType>::Concrete,
}

impl<'a> InvokeValueDeserializer<'a, CoreType, CoreLibfunc>
    for InvokeCoreTypeDeserializer<'a, CoreType, CoreLibfunc>
{
    fn new(
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
        info: &'a CoreTypeConcrete,
    ) -> Self {
        Self { registry, info }
    }
}

impl InvokeValueBuilder<CoreType, CoreLibfunc> for CoreTypeConcrete {
    type Deserializer<'a> = InvokeCoreTypeDeserializer<'a, CoreType, CoreLibfunc>;
    type Serializer<'a> = InvokeCoreTypeSerializer<'a, CoreType, CoreLibfunc>;

    type Error = std::convert::Infallible;
}

impl<'a, 'de> DeserializeSeed<'de> for InvokeCoreTypeDeserializer<'a, CoreType, CoreLibfunc> {
    type Value = InvokeArg;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match self.info {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => {
                let felt = <[u32; 8]>::deserialize(deserializer)?;
                let felt = Felt252::from(BigUint::new(felt.to_vec()));
                Ok(InvokeArg::Felt252(felt))
            }
            CoreTypeConcrete::GasBuiltin(_) => todo!(),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => Ok(InvokeArg::Uint8(u8::deserialize(deserializer)?)),
            CoreTypeConcrete::Uint16(_) => Ok(InvokeArg::Uint16(u16::deserialize(deserializer)?)),
            CoreTypeConcrete::Uint32(_) => Ok(InvokeArg::Uint32(u32::deserialize(deserializer)?)),
            CoreTypeConcrete::Uint64(_) => Ok(InvokeArg::Uint64(u64::deserialize(deserializer)?)),
            CoreTypeConcrete::Uint128(_) => {
                Ok(InvokeArg::Uint128(u128::deserialize(deserializer)?))
            }
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
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
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }
}

/// Serializer for Cairo's [`CoreType`].
pub struct InvokeCoreTypeSerializer<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    value: InvokeArg,
    registry: &'a ProgramRegistry<TType, TLibfunc>,
    info: &'a <TType as GenericType>::Concrete,
}

impl<'a> InvokeValueSerializer<'a, CoreType, CoreLibfunc>
    for InvokeCoreTypeSerializer<'a, CoreType, CoreLibfunc>
{
    fn new(
        value: InvokeArg,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
        info: &'a <CoreType as GenericType>::Concrete,
    ) -> Self {
        Self {
            value,
            registry,
            info,
        }
    }
}

impl<'a> Serialize for InvokeCoreTypeSerializer<'a, CoreType, CoreLibfunc> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.info {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => todo!(),
            CoreTypeConcrete::GasBuiltin(_) => todo!(),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
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
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }
}

pub struct InvokeArgVisitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
{
    pub registry: &'a ProgramRegistry<TType, TLibfunc>,
    pub types: Vec<ConcreteTypeId>,
}

impl<'a, 'de, TType, TLibfunc> Visitor<'de> for InvokeArgVisitor<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: InvokeValueBuilder<TType, TLibfunc>,
{
    type Value = Vec<InvokeArg>;

    fn expecting(&self, _formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        todo!()
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();

        for type_id in &self.types {
            let ty = self.registry.get_type(type_id).unwrap();

            type ParamDeserializer<'a, TType, TLibfunc> =
                <<TType as GenericType>::Concrete as InvokeValueBuilder<TType, TLibfunc>>::Deserializer<
                    'a,
                >;
            let deserializer = ParamDeserializer::<TType, TLibfunc>::new(self.registry, ty);

            let value = seq
                .next_element_seed::<ParamDeserializer<TType, TLibfunc>>(deserializer)?
                .unwrap();

            values.push(value);
        }

        Ok(values)
    }
}
