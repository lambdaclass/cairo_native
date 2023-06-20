use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::{core::CoreTypeConcrete, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};
use std::error::Error;

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

pub trait TypeBuilder {
    type Error: Error;

    fn build<'ctx, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Result<Type<'ctx>, Self::Error>
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder;

    fn variants(&self) -> Option<&[ConcreteTypeId]>;
}

impl TypeBuilder for CoreTypeConcrete {
    type Error = std::convert::Infallible;

    fn build<'ctx, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Result<Type<'ctx>, Self::Error>
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
    {
        match self {
            Self::Array(info) => self::array::build(context, module, registry, metadata, info),
            Self::Bitwise(_) => todo!(),
            Self::Box(_) => todo!(),
            Self::BuiltinCosts(info) => self::builtin_costs::build(context, module, registry, metadata, info),
            Self::EcOp(_) => todo!(),
            Self::EcPoint(_) => todo!(),
            Self::EcState(_) => todo!(),
            Self::Enum(info) => self::r#enum::build(context, module, registry, metadata, info),
            Self::Felt252(info) => self::felt252::build(context, module, registry, metadata, info),
            Self::Felt252Dict(_) => todo!(),
            Self::Felt252DictEntry(_) => todo!(),
            Self::GasBuiltin(info) => {
                self::gas_builtin::build(context, module, registry, metadata, info)
            }
            Self::NonZero(info) => self::non_zero::build(context, module, registry, metadata, info),
            Self::Nullable(_) => todo!(),
            Self::Pedersen(_) => todo!(),
            Self::Poseidon(_) => todo!(),
            Self::RangeCheck(info) => {
                self::range_check::build(context, module, registry, metadata, info)
            }
            Self::SegmentArena(_) => todo!(),
            Self::Snapshot(_) => todo!(),
            Self::Span(_) => todo!(),
            Self::SquashedFelt252Dict(_) => todo!(),
            Self::StarkNet(_) => todo!(),
            Self::Struct(info) => self::r#struct::build(context, module, registry, metadata, info),
            Self::Uint128(_) => todo!(),
            Self::Uint128MulGuarantee(_) => todo!(),
            Self::Uint16(_) => todo!(),
            Self::Uint32(info) => self::uint32::build(context, module, registry, metadata, info),
            Self::Uint64(_) => todo!(),
            Self::Uint8(info) => self::uint8::build(context, module, registry, metadata, info),
            Self::Uninitialized(_) => todo!(),
        }
    }

    fn variants(&self) -> Option<&[ConcreteTypeId]> {
        match self {
            Self::Enum(info) => Some(&info.variants),
            _ => None,
        }
    }
}
