//! # Non-zero type
//!
//! The non-zero generic type guarantees that whatever value it has is not zero.
//!
//! ## Layout
//!
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! to the following:
//!
//! ```
//! #[repr(transparent)]
//! pub struct NonZero<T>(pub T);
//! ```

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
    registry
        .get_type(&info.ty)?
        .build(context, module, registry, metadata)
}
