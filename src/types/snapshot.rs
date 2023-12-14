//! # Snapshot type
//!
//! The type snapshot for a given type `T`.
//!
//! ## Layout
//!
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! to the following:
//!
//! ```
//! #[repr(transparent)]
//! pub struct Snapshot<T>(pub T);
//! ```

use super::{TypeBuilder, WithSelf};
use crate::{
    error::types::{Error, Result},
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    utils::ProgramRegistryExt,
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
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    // This type is like a `Cow<T>` that clones whenever the original type is modified to keep the
    // original data. Since implementing that is complicated we can just clone the entire value for
    // now.
    match metadata.get_mut::<EnumSnapshotVariantsMeta>() {
        Some(x) => x,
        None => metadata
            .insert(EnumSnapshotVariantsMeta::default())
            .unwrap(),
    }
    .set_mapping(info.self_ty, registry.get_type(&info.ty)?.variants());

    registry.build_type(context, module, registry, metadata, &info.ty)
}
