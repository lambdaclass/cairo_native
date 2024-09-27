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
    error::Result,
    metadata::{
        enum_snapshot_variants::EnumSnapshotVariantsMeta, snapshot_clones::SnapshotClonesMeta,
        MetadataStorage,
    },
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    // This type is like a `Cow<T>` that clones whenever the original type is modified to keep the
    // original data. Since implementing that is complicated we can just clone the entire value for
    // now.

    // Register enum variants for the snapshot.
    if let Some(variants) = registry.get_type(&info.ty)?.variants() {
        metadata
            .get_or_insert_with(EnumSnapshotVariantsMeta::default)
            .set_mapping(info.self_ty, variants);
    }

    // Ensure the inner type is built and register the snapshot clone logic builder.
    let self_ty = registry.build_type(context, module, registry, metadata, &info.ty)?;
    if let Some(snapshot_clones_meta) = metadata.get_mut::<SnapshotClonesMeta>() {
        snapshot_clones_meta.register_dup(info.self_ty.clone(), &info.ty);
    }

    Ok(self_ty)
}
