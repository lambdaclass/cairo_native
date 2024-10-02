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
        dup_overrides::DupOverridesMeta, enum_snapshot_variants::EnumSnapshotVariantsMeta,
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
    dialect::func,
    ir::{Block, Location, Module, Region, Type},
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

    // Register clone override (if required).
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            registry.build_type(context, module, registry, metadata, &info.ty)?;

            // The following unwrap is unreachable because `register_with` will always insert it before
            // calling this closure.
            metadata
                .get::<DupOverridesMeta>()
                .unwrap()
                .is_overriden(&info.ty)
                .then(|| build_dup(context, module, registry, metadata, &info))
                .transpose()
        },
    )?;

    registry.build_type(context, module, registry, metadata, &info.ty)
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);

    let inner_ty = registry.build_type(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(inner_ty, location)]));

    // The following unwrap is unreachable because the registration logic will always insert it.
    let values = metadata
        .get::<DupOverridesMeta>()
        .unwrap()
        .invoke_override(
            context,
            &entry,
            location,
            &info.ty,
            entry.argument(0)?.into(),
        )?;

    entry.append_operation(func::r#return(&[values.0, values.1], location));
    Ok(region)
}
