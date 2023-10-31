//! # Nullable type
//!
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//!
//! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
//! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.

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
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: &InfoAndTypeConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // nullable is represented as a pointer, like a box, used to check if its null (when it can be null).
    Ok(llvm::r#type::opaque_pointer(context))
}
