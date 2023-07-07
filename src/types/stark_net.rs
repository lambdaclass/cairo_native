//! # StarkNet types
//!
//! ## ClassHash
//! Type for Starknet class hash, a value in the range [0, 2 ** 251).
//!
//! ## ContractAddress
//! Type for Starknet contract address, a value in the range [0, 2 ** 251).
//!
//! ## StorageBaseAddress
//! Type for Starknet storage base address, a value in the range [0, 2 ** 251 - 256).
//!
//! ## StorageAddress
//! Type for Starknet storage base address, a value in the range [0, 2 ** 251).
//!
//! ## System
//! Type for Starknet system object.
//! Used to make system calls.
//!
//! ## Secp256Point
//! TODO

// TODO: Maybe the types used here can be i251 instead of i252.

use super::TypeBuilder;
use crate::{
    error::types::{Error, Result},
    metadata::MetadataStorage,
};
use cairo_lang_sierra::{
    extensions::{
        starknet::StarkNetTypeConcrete, types::InfoOnlyConcreteType, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    selector: &StarkNetTypeConcrete,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        StarkNetTypeConcrete::ClassHash(info) => {
            build_class_hash(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::ContractAddress(info) => {
            build_contract_address(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::StorageBaseAddress(info) => {
            build_storage_base_address(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::StorageAddress(info) => {
            build_storage_address(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::System(info) => {
            build_system(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::Secp256Point(_) => todo!(),
    }
}

pub fn build_class_hash<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // it's a felt252 value
    super::felt252::build(context, module, registry, metadata, info)
}

pub fn build_contract_address<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // it's a felt252 value
    super::felt252::build(context, module, registry, metadata, info)
}

pub fn build_storage_base_address<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // it's a felt252 value
    super::felt252::build(context, module, registry, metadata, info)
}

pub fn build_storage_address<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // it's a felt252 value
    super::felt252::build(context, module, registry, metadata, info)
}

pub fn build_system<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // builtin
    Ok(llvm::r#type::array(IntegerType::new(context, 8).into(), 0))
}
