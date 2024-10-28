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
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
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
    dialect::{func, llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Region, Type},
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
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_dup`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;
    DropOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_drop`.

            Ok(Some(build_drop(
                context, module, registry, metadata, &info,
            )?))
        },
    )?;

    Ok(llvm::r#type::pointer(context, 0))
}

fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let inner_ty = registry.get_type(&info.ty)?;
    let inner_len = inner_ty.layout(registry)?.pad_to_align().size();
    let inner_ty = inner_ty.build(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let inner_len_val = entry.const_int(context, location, inner_len, 64)?;

    let src_value = entry.argument(0)?.into();
    let dst_value = entry.append_op_result(ReallocBindingsMeta::realloc(
        context,
        null_ptr,
        inner_len_val,
        location,
    ))?;

    match metadata.get::<DupOverridesMeta>() {
        Some(dup_override_meta) if dup_override_meta.is_overriden(&info.ty) => {
            let value = entry.load(context, location, src_value, inner_ty)?;
            let values =
                dup_override_meta.invoke_override(context, &entry, location, &info.ty, value)?;
            entry.store(context, location, src_value, values.0)?;
            entry.store(context, location, dst_value, values.1)?;
        }
        _ => {
            entry.append_operation(
                ods::llvm::intr_memcpy_inline(
                    context,
                    dst_value,
                    src_value,
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), inner_len as i64),
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                    location,
                )
                .into(),
            );
        }
    }

    entry.append_operation(func::r#return(&[src_value, dst_value], location));
    Ok(region)
}

fn build_drop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let inner_ty = registry.build_type(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    let value = entry.argument(0)?.into();
    match metadata.get::<DropOverridesMeta>() {
        Some(drop_override_meta) if drop_override_meta.is_overriden(&info.ty) => {
            let value = entry.load(context, location, value, inner_ty)?;
            drop_override_meta.invoke_override(context, &entry, location, &info.ty, value)?;
        }
        _ => {}
    }

    entry.append_operation(ReallocBindingsMeta::free(context, value, location));
    entry.append_operation(func::r#return(&[], location));
    Ok(region)
}
