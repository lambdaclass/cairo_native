//! # Nullable type
//!
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//!
//! This is so we only check if the ptr is nullptr for nulability, instead of using a enum in this case.
//! TODO

use super::TypeBuilder;
use crate::{
    error::types::{Error, Result},
    metadata::MetadataStorage,
};
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Module, Type},
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
    info: &InfoAndTypeConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
    let inner = registry
        .get_type(&info.ty)?
        .build(context, module, registry, metadata)?;

    Ok(llvm::r#type::pointer(inner, 0))
}
