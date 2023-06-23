//! # Compiler type infrastructure
//!
//! Contains type generation stuff (aka. conversion from Sierra to MLIR types).

use crate::{metadata::MetadataStorage, utils::get_integer_layout};
use cairo_lang_sierra::{
    extensions::{core::CoreTypeConcrete, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};
use std::{alloc::Layout, error::Error};

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

/// Generation of MLIR types from their Sierra counterparts.
///
/// All possible Sierra types must implement it. It is already implemented for all the core Sierra
/// types, contained in [CoreTypeConcrete].
pub trait TypeBuilder {
    /// Error type returned by this trait's methods.
    type Error: Error;

    /// Build the MLIR type.
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

    /// Generate the layout of the MLIR type.
    ///
    /// Used in both the compiler and the interface when calling the compiled code.
    fn layout<TType, TLibfunc>(&self, registry: &ProgramRegistry<TType, TLibfunc>) -> Layout
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder;

    /// If the type is a variant type, return all possible variants.
    ///
    /// TODO: How is it used?
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
            Self::BuiltinCosts(info) => {
                self::builtin_costs::build(context, module, registry, metadata, info)
            }
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
            Self::Uint16(info) => self::uint16::build(context, module, registry, metadata, info),
            Self::Uint32(info) => self::uint32::build(context, module, registry, metadata, info),
            Self::Uint64(info) => self::uint64::build(context, module, registry, metadata, info),
            Self::Uint8(info) => self::uint8::build(context, module, registry, metadata, info),
            Self::Uninitialized(_) => todo!(),
        }
    }

    fn layout<TType, TLibfunc>(&self, registry: &ProgramRegistry<TType, TLibfunc>) -> Layout
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
    {
        match self {
            CoreTypeConcrete::Array(_) => {
                Layout::new::<*mut ()>()
                    .extend(get_integer_layout(32))
                    .unwrap()
                    .0
                    .extend(get_integer_layout(32))
                    .unwrap()
                    .0
            }
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => get_integer_layout(252),
            CoreTypeConcrete::GasBuiltin(_) => get_integer_layout(64),
            CoreTypeConcrete::BuiltinCosts(_) => Layout::new::<()>(), // TODO: Figure out builtins layout
            CoreTypeConcrete::Uint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Uint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Uint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Uint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Uint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Uint128MulGuarantee(_) => Layout::new::<()>(), // TODO: Figure out builtins layout
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => Layout::new::<()>(), // TODO: Figure out builtins layout
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(info) => {
                let tag_layout =
                    get_integer_layout(info.variants.len().next_power_of_two().trailing_zeros());

                info.variants.iter().fold(tag_layout, |acc, id| {
                    let layout = tag_layout
                        .extend(registry.get_type(id).unwrap().layout(registry))
                        .unwrap()
                        .0;

                    Layout::from_size_align(
                        acc.size().max(layout.size()),
                        acc.align().max(layout.align()),
                    )
                    .unwrap()
                })
            }
            CoreTypeConcrete::Struct(info) => info
                .members
                .iter()
                .fold(Option::<Layout>::None, |acc, id| {
                    Some(match acc {
                        Some(layout) => {
                            layout
                                .extend(registry.get_type(id).unwrap().layout(registry))
                                .unwrap()
                                .0
                        }
                        None => registry.get_type(id).unwrap().layout(registry),
                    })
                })
                .unwrap_or(Layout::from_size_align(0, 1).unwrap()),
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => Layout::new::<()>(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }
    }

    fn variants(&self) -> Option<&[ConcreteTypeId]> {
        match self {
            Self::Enum(info) => Some(&info.variants),
            _ => None,
        }
    }
}
