//! # Compiler type infrastructure
//!
//! Contains type generation stuff (aka. conversion from Sierra to MLIR types).

use crate::{error::CoreTypeBuilderError, metadata::MetadataStorage, utils::get_integer_layout};
use cairo_lang_sierra::{
    extensions::{
        core::CoreTypeConcrete, starknet::StarkNetTypeConcrete, GenericLibfunc, GenericType,
    },
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
pub trait TypeBuilder<TType, TLibfunc>
where
    TType: GenericType<Concrete = Self>,
    TLibfunc: GenericLibfunc,
{
    /// Error type returned by this trait's methods.
    type Error: Error;

    /// Build the MLIR type.
    fn build<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Result<Type<'ctx>, Self::Error>;

    /// Generate the layout of the MLIR type.
    ///
    /// Used in both the compiler and the interface when calling the compiled code.
    fn layout(&self, registry: &ProgramRegistry<TType, TLibfunc>) -> Result<Layout, Self::Error>;

    /// If the type is a variant type, return all possible variants.
    ///
    /// TODO: How is it used?
    fn variants(&self) -> Option<&[ConcreteTypeId]>;
}

impl<TType, TLibfunc> TypeBuilder<TType, TLibfunc> for CoreTypeConcrete
where
    TType: GenericType<Concrete = Self>,
    TLibfunc: GenericLibfunc,
{
    type Error = CoreTypeBuilderError;

    fn build<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Result<Type<'ctx>, Self::Error> {
        match self {
            Self::Array(info) => self::array::build(context, module, registry, metadata, info),
            Self::Bitwise(info) => self::bitwise::build(context, module, registry, metadata, info),
            Self::Box(info) => self::r#box::build(context, module, registry, metadata, info),
            Self::BuiltinCosts(info) => {
                self::builtin_costs::build(context, module, registry, metadata, info)
            }
            Self::EcOp(_) => todo!(),
            Self::EcPoint(_) => todo!(),
            Self::EcState(_) => todo!(),
            Self::Enum(info) => self::r#enum::build(context, module, registry, metadata, info),
            Self::Felt252(info) => self::felt252::build(context, module, registry, metadata, info),
            Self::Felt252Dict(info) => {
                self::felt252_dict::build(context, module, registry, metadata, info)
            }
            Self::Felt252DictEntry(info) => {
                self::felt252_dict_entry::build(context, module, registry, metadata, info)
            }
            Self::GasBuiltin(info) => {
                self::gas_builtin::build(context, module, registry, metadata, info)
            }
            Self::NonZero(info) => self::non_zero::build(context, module, registry, metadata, info),
            Self::Nullable(_) => todo!(),
            Self::Pedersen(info) => {
                self::pedersen::build(context, module, registry, metadata, info)
            }
            Self::Poseidon(_) => todo!(),
            Self::RangeCheck(info) => {
                self::range_check::build(context, module, registry, metadata, info)
            }
            Self::SegmentArena(info) => {
                self::segment_arena::build(context, module, registry, metadata, info)
            }
            Self::Snapshot(info) => {
                self::snapshot::build(context, module, registry, metadata, info)
            }
            Self::Span(_) => todo!(),
            Self::SquashedFelt252Dict(info) => {
                self::squashed_felt252_dict::build(context, module, registry, metadata, info)
            }
            Self::StarkNet(selector) => {
                self::stark_net::build(context, module, registry, metadata, selector)
            }
            Self::Struct(info) => self::r#struct::build(context, module, registry, metadata, info),
            Self::Uint128(info) => self::uint128::build(context, module, registry, metadata, info),
            Self::Uint128MulGuarantee(_) => todo!(),
            Self::Uint16(info) => self::uint16::build(context, module, registry, metadata, info),
            Self::Uint32(info) => self::uint32::build(context, module, registry, metadata, info),
            Self::Uint64(info) => self::uint64::build(context, module, registry, metadata, info),
            Self::Uint8(info) => self::uint8::build(context, module, registry, metadata, info),
            Self::Uninitialized(info) => {
                self::uninitialized::build(context, module, registry, metadata, info)
            }
        }
    }

    fn layout(&self, registry: &ProgramRegistry<TType, TLibfunc>) -> Result<Layout, Self::Error> {
        Ok(match self {
            CoreTypeConcrete::Array(_) => {
                Layout::new::<*mut ()>()
                    .extend(get_integer_layout(32))?
                    .0
                    .extend(get_integer_layout(32))?
                    .0
            }
            CoreTypeConcrete::Bitwise(_) => Layout::new::<()>(),
            CoreTypeConcrete::Box(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => get_integer_layout(252),
            CoreTypeConcrete::GasBuiltin(_) => get_integer_layout(64),
            CoreTypeConcrete::BuiltinCosts(_) => Layout::new::<()>(), // TODO: Figure out builtins layout.
            CoreTypeConcrete::Uint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Uint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Uint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Uint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Uint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Uint128MulGuarantee(_) => Layout::new::<()>(), // TODO: Figure out builtins layout.
            CoreTypeConcrete::NonZero(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => Layout::new::<()>(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(info) => {
                let tag_layout =
                    get_integer_layout(info.variants.len().next_power_of_two().trailing_zeros());

                info.variants.iter().try_fold(tag_layout, |acc, id| {
                    let layout = tag_layout
                        .extend(registry.get_type(id)?.layout(registry)?)?
                        .0;

                    Result::<_, Self::Error>::Ok(Layout::from_size_align(
                        acc.size().max(layout.size()),
                        acc.align().max(layout.align()),
                    )?)
                })?
            }
            CoreTypeConcrete::Struct(info) => info
                .members
                .iter()
                .try_fold(Option::<Layout>::None, |acc, id| {
                    Result::<_, Self::Error>::Ok(Some(match acc {
                        Some(layout) => layout.extend(registry.get_type(id)?.layout(registry)?)?.0,
                        None => registry.get_type(id)?.layout(registry)?,
                    }))
                })?
                .unwrap_or(Layout::from_size_align(0, 1)?),
            CoreTypeConcrete::Felt252Dict(_) => get_integer_layout(64), // ptr
            CoreTypeConcrete::Felt252DictEntry(_) => {
                get_integer_layout(252)
                    .extend(Layout::new::<*mut ()>())
                    .unwrap()
                    .0
            }
            CoreTypeConcrete::SquashedFelt252Dict(_) => get_integer_layout(64), // ptr
            CoreTypeConcrete::Pedersen(_) => Layout::new::<()>(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(info) => match info {
                StarkNetTypeConcrete::ClassHash(_) => get_integer_layout(252),
                StarkNetTypeConcrete::ContractAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageBaseAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::System(_) => Layout::new::<()>(),
                StarkNetTypeConcrete::Secp256Point(_) => todo!(),
            },
            CoreTypeConcrete::SegmentArena(_) => Layout::new::<()>(),
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty)?.layout(registry)?,
        })
    }

    fn variants(&self) -> Option<&[ConcreteTypeId]> {
        match self {
            Self::Enum(info) => Some(&info.variants),
            _ => None,
        }
    }
}
