use cairo_lang_sierra::{
    extensions::{core::CoreTypeConcrete, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{ir::Type, Context};
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

pub struct TypeBuilderContext<'ctx, 'this, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    context: &'ctx Context,
    registry: &'this ProgramRegistry<TType, TLibfunc>,
}

impl<'ctx, 'this, TType, TLibfunc> Clone for TypeBuilderContext<'ctx, 'this, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<'ctx, 'this, TType, TLibfunc> Copy for TypeBuilderContext<'ctx, 'this, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
}

impl<'ctx, 'this, TType, TLibfunc> TypeBuilderContext<'ctx, 'this, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    pub fn new(context: &'ctx Context, registry: &'this ProgramRegistry<TType, TLibfunc>) -> Self {
        Self { context, registry }
    }

    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    pub fn registry(&self) -> &ProgramRegistry<TType, TLibfunc> {
        self.registry
    }
}

pub trait TypeBuilder {
    type Error: Error;

    fn build<'ctx, TType, TLibfunc>(
        &self,
        context: TypeBuilderContext<'ctx, '_, TType, TLibfunc>,
    ) -> Result<Type<'ctx>, Self::Error>
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder;
}

impl TypeBuilder for CoreTypeConcrete {
    type Error = std::convert::Infallible;

    fn build<'ctx, TType, TLibfunc>(
        &self,
        context: TypeBuilderContext<'ctx, '_, TType, TLibfunc>,
    ) -> Result<Type<'ctx>, Self::Error>
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
    {
        match self {
            Self::Array(info) => self::array::build(context, info),
            Self::Bitwise(_) => todo!(),
            Self::Box(_) => todo!(),
            Self::BuiltinCosts(_) => todo!(),
            Self::EcOp(_) => todo!(),
            Self::EcPoint(_) => todo!(),
            Self::EcState(_) => todo!(),
            Self::Enum(_) => todo!(),
            Self::Felt252(info) => self::felt252::build(context, info),
            Self::Felt252Dict(_) => todo!(),
            Self::Felt252DictEntry(_) => todo!(),
            Self::GasBuiltin(_) => todo!(),
            Self::NonZero(_) => todo!(),
            Self::Nullable(_) => todo!(),
            Self::Pedersen(_) => todo!(),
            Self::Poseidon(_) => todo!(),
            Self::RangeCheck(_) => todo!(),
            Self::SegmentArena(_) => todo!(),
            Self::Snapshot(_) => todo!(),
            Self::Span(_) => todo!(),
            Self::SquashedFelt252Dict(_) => todo!(),
            Self::StarkNet(_) => todo!(),
            Self::Struct(info) => self::r#struct::build(context, info),
            Self::Uint128(_) => todo!(),
            Self::Uint128MulGuarantee(_) => todo!(),
            Self::Uint16(_) => todo!(),
            Self::Uint32(_) => todo!(),
            Self::Uint64(_) => todo!(),
            Self::Uint8(_) => todo!(),
            Self::Uninitialized(_) => todo!(),
        }
    }
}
