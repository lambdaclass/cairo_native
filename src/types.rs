//! # Compiler type infrastructure
//!
//! Contains type generation stuff (aka. conversion from Sierra to MLIR types).

use crate::{
    error::CoreTypeBuilderError,
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    utils::{get_integer_layout, layout_repeat, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreConcreteLibfunc, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
        GenericLibfunc, GenericType,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm::{self, r#type::opaque_pointer},
    ir::{attribute::DenseI64ArrayAttribute, Block, Location, Value},
};
use melior::{
    ir::{Module, Type},
    Context,
};
use std::{alloc::Layout, error::Error, ops::Deref};

pub mod array;
pub mod bitwise;
pub mod r#box;
pub mod builtin_costs;
pub mod bytes31;
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
        self_ty: &ConcreteTypeId,
    ) -> Result<Type<'ctx>, Self::Error>;

    /// Generate the layout of the MLIR type.
    ///
    /// Used in both the compiler and the interface when calling the compiled code.
    fn layout(&self, registry: &ProgramRegistry<TType, TLibfunc>) -> Result<Layout, Self::Error>;

    /// If the type is an integer (felt not included) type, return its width in bits.
    ///
    /// TODO: How is it used?
    fn integer_width(&self) -> Option<usize>;

    /// If the type is a variant type, return all possible variants.
    ///
    /// TODO: How is it used?
    fn variants(&self) -> Option<&[ConcreteTypeId]>;

    #[allow(clippy::too_many_arguments)]
    fn build_drop<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
        self_ty: &ConcreteTypeId,
    ) -> Result<(), Self::Error>;
}

impl<TType, TLibfunc> TypeBuilder<TType, TLibfunc> for CoreTypeConcrete
where
    TType: 'static + GenericType<Concrete = Self>,
    TLibfunc: 'static + GenericLibfunc<Concrete = CoreConcreteLibfunc>,
    // TODO: Find a way to remove the `Concrete = CoreConcreteLibfunc` requirement on `TLibfunc` and
    //   instead add the following restriction without causing an overflow evaluating requirement
    //   error:
    // <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
{
    type Error = CoreTypeBuilderError;

    fn build<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
        self_ty: &ConcreteTypeId,
    ) -> Result<Type<'ctx>, Self::Error> {
        match self {
            Self::Array(info) => self::array::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Bitwise(info) => self::bitwise::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Box(info) => self::r#box::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::BuiltinCosts(info) => self::builtin_costs::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::EcOp(info) => self::ec_op::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::EcPoint(info) => self::ec_point::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::EcState(info) => self::ec_state::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Enum(info) => self::r#enum::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Felt252(info) => self::felt252::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Felt252Dict(info) => self::felt252_dict::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Felt252DictEntry(info) => self::felt252_dict_entry::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::GasBuiltin(info) => self::gas_builtin::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::NonZero(info) => self::non_zero::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Nullable(info) => self::nullable::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Pedersen(info) => self::pedersen::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Poseidon(info) => self::poseidon::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::RangeCheck(info) => self::range_check::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::SegmentArena(info) => self::segment_arena::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Snapshot(info) => self::snapshot::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Span(_) => todo!("implement span type"),
            Self::SquashedFelt252Dict(info) => self::squashed_felt252_dict::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::StarkNet(selector) => self::stark_net::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, selector),
            ),
            Self::Struct(info) => self::r#struct::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint128(info) => self::uint128::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint128MulGuarantee(info) => self::uint128_mul_guarantee::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint16(info) => self::uint16::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint32(info) => self::uint32::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint64(info) => self::uint64::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uint8(info) => self::uint8::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            Self::Uninitialized(info) => self::uninitialized::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Sint8(info) => self::uint8::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Sint16(info) => self::uint16::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Sint32(info) => self::uint32::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Sint64(info) => self::uint64::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Sint128(info) => self::uint128::build(
                context,
                module,
                registry,
                metadata,
                WithSelf::new(self_ty, info),
            ),
            CoreTypeConcrete::Bytes31(info) => {
                self::bytes31::build(context, module, registry, metadata, info)
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
            CoreTypeConcrete::Box(_) => Layout::new::<*mut ()>(),
            CoreTypeConcrete::EcOp(_) => Layout::new::<()>(),
            CoreTypeConcrete::EcPoint(_) => layout_repeat(&get_integer_layout(252), 2)?.0,
            CoreTypeConcrete::EcState(_) => layout_repeat(&get_integer_layout(252), 4)?.0,
            CoreTypeConcrete::Felt252(_) => get_integer_layout(252),
            CoreTypeConcrete::GasBuiltin(_) => get_integer_layout(128),
            CoreTypeConcrete::BuiltinCosts(_) => Layout::new::<()>(), // TODO: Figure out builtins layout.
            CoreTypeConcrete::Uint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Uint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Uint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Uint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Uint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Uint128MulGuarantee(_) => Layout::new::<()>(), // TODO: Figure out builtins layout.
            CoreTypeConcrete::NonZero(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::Nullable(_) => Layout::new::<*mut ()>(),
            CoreTypeConcrete::RangeCheck(_) => Layout::new::<()>(),
            CoreTypeConcrete::Uninitialized(info) => {
                registry.get_type(&info.ty)?.layout(registry)?
            }
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
            CoreTypeConcrete::Felt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
            CoreTypeConcrete::Felt252DictEntry(_) => {
                get_integer_layout(252)
                    .extend(Layout::new::<*mut std::ffi::c_void>())
                    .unwrap()
                    .0
                    .extend(Layout::new::<*mut std::ffi::c_void>())
                    .unwrap()
                    .0
            }
            CoreTypeConcrete::SquashedFelt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
            CoreTypeConcrete::Pedersen(_) => Layout::new::<()>(),
            CoreTypeConcrete::Poseidon(_) => Layout::new::<()>(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(info) => match info {
                StarkNetTypeConcrete::ClassHash(_) => get_integer_layout(252),
                StarkNetTypeConcrete::ContractAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageBaseAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::System(_) => Layout::new::<*mut ()>(),
                StarkNetTypeConcrete::Secp256Point(_) => todo!("implement Secp256Point type"),
            },
            CoreTypeConcrete::SegmentArena(_) => Layout::new::<()>(),
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::Sint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Sint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Sint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Sint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Sint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Bytes31(_) => get_integer_layout(248),
        })
    }

    fn integer_width(&self) -> Option<usize> {
        match self {
            Self::Uint8(_) => Some(8),
            Self::Uint16(_) => Some(16),
            Self::Uint32(_) => Some(32),
            Self::Uint64(_) => Some(64),
            Self::Uint128(_) => Some(128),
            _ => None,
        }
    }

    fn variants(&self) -> Option<&[ConcreteTypeId]> {
        match self {
            Self::Enum(info) => Some(&info.variants),
            _ => None,
        }
    }

    fn build_drop<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
        self_ty: &ConcreteTypeId,
    ) -> Result<(), Self::Error> {
        match self {
            CoreTypeConcrete::Array(_info) => {
                let array_ty = registry.build_type(context, helper, registry, metadata, self_ty)?;

                let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);

                let array_val = entry.argument(0)?.into();

                let op = entry.append_operation(llvm::extract_value(
                    context,
                    array_val,
                    DenseI64ArrayAttribute::new(context, &[0]),
                    ptr_ty,
                    location,
                ));
                let ptr: Value = op.result(0)?.into();

                let ptr = entry
                    .append_operation(llvm::bitcast(ptr, opaque_pointer(context), location))
                    .result(0)?
                    .into();

                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
            }
            CoreTypeConcrete::Felt252Dict(_) | CoreTypeConcrete::SquashedFelt252Dict(_) => {
                let runtime: &mut RuntimeBindingsMeta = metadata.get_mut().unwrap();
                let ptr = entry.argument(0)?.into();

                runtime.dict_alloc_free(context, helper, ptr, entry, location)?;
            }
            CoreTypeConcrete::Box(_) | CoreTypeConcrete::Nullable(_) => {
                if metadata.get::<ReallocBindingsMeta>().is_none() {
                    metadata.insert(ReallocBindingsMeta::new(context, helper));
                }

                let ptr = entry.argument(0)?.into();
                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
            }
            _ => {}
        };
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WithSelf<'a, T> {
    self_ty: &'a ConcreteTypeId,
    inner: &'a T,
}

impl<'a, T> WithSelf<'a, T> {
    pub fn new(self_ty: &'a ConcreteTypeId, inner: &'a T) -> Self {
        Self { self_ty, inner }
    }

    pub fn self_ty(&self) -> &ConcreteTypeId {
        self.self_ty
    }
}

impl<'a, T> AsRef<T> for WithSelf<'a, T> {
    fn as_ref(&self) -> &T {
        self.inner
    }
}

impl<'a, T> Deref for WithSelf<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.inner
    }
}
