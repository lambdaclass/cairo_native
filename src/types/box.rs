//! # Box type
//!
//! The type box for a given type `T`.
//!
//! ## Layout
//!
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! to the following:
//!
//! ```
//! #[repr(transparent)]
//! pub struct Box<T>(pub T);
//! ```

use super::WithSelf;
use crate::{
    block_ext::BlockExt,
    error::Result,
    ffi::get_mlir_layout,
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
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
    dialect::{
        llvm::{self, r#type::opaque_pointer},
        ods,
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    metadata
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .register(
            info.self_ty().clone(),
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        );

    Ok(opaque_pointer(context))
}

#[allow(clippy::too_many_arguments)]
fn snapshot_take<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
    src_value: Value<'ctx, 'this>,
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let inner_snapshot_take = metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.ty));

    let inner_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
    let inner_layout = get_mlir_layout(helper, inner_ty);

    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;

    let ptr = entry
        .append_operation(llvm::nullptr(opaque_pointer(context), location))
        .result(0)?
        .into();
    let dst_ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(
            context, ptr, value_len, location,
        ))
        .result(0)?
        .into();

    match inner_snapshot_take {
        Some(inner_snapshot_take) => {
            let value = entry.load(
                context,
                location,
                src_value,
                inner_ty,
                Some(inner_layout.align()),
            )?;

            let (entry, value) =
                inner_snapshot_take(context, registry, entry, location, helper, metadata, value)?;

            entry.store(
                context,
                location,
                dst_ptr,
                value,
                Some(inner_layout.align()),
            );
        }
        None => {
            entry.append_operation(
                ods::llvm::intr_memcpy(
                    context,
                    dst_ptr,
                    src_value,
                    value_len,
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                    location,
                )
                .into(),
            );
        }
    }

    Ok((entry, dst_ptr))
}
