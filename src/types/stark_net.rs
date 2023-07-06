//! # StarkNet type
//!
//! TODO

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
    info: &StarkNetTypeConcrete,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    match info {
        StarkNetTypeConcrete::ClassHash(_) => todo!(),
        StarkNetTypeConcrete::ContractAddress(_) => todo!(),
        StarkNetTypeConcrete::StorageBaseAddress(_) => todo!(),
        StarkNetTypeConcrete::StorageAddress(_) => todo!(),
        StarkNetTypeConcrete::System(info) => {
            build_system(context, module, registry, metadata, info)
        }
        StarkNetTypeConcrete::Secp256Point(_) => todo!(),
    }
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
    // TODO: should it be transparent like this in MLIR?
    Ok(llvm::r#type::array(IntegerType::new(context, 8).into(), 0))
}
