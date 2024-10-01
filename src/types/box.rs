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
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
    },
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
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
        llvm::{self, r#type::pointer},
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
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    SnapshotClonesMeta::register_with(metadata, info.self_ty().clone(), |metadata| {
        registry.build_type(context, module, registry, metadata, &info.ty)?;

        Ok(Some((
            snapshot_take,
            InfoAndTypeConcreteType {
                info: info.info.clone(),
                ty: info.ty.clone(),
            },
        )))
    })?;

    Ok(llvm::r#type::pointer(context, 0))
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

    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, info.self_ty())?;

    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;

    let ptr = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;
    let dst_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context, ptr, value_len, location,
    ))?;

    match inner_snapshot_take {
        Some(inner_snapshot_take) => {
            let value = entry.load(context, location, src_value, inner_ty)?;

            let (entry, value) =
                inner_snapshot_take(context, registry, entry, location, helper, metadata, value)?;

            entry.store(context, location, dst_ptr, value)?;
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
