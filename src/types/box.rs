////! # Box type
//! # Box type
////!
//!
////! The type box for a given type `T`.
//! The type box for a given type `T`.
////!
//!
////! ## Layout
//! ## Layout
////!
//!
////! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
////! to the following:
//! to the following:
////!
//!
////! ```
//! ```
////! #[repr(transparent)]
//! #[repr(transparent)]
////! pub struct Box<T>(pub T);
//! pub struct Box<T>(pub T);
////! ```
//! ```
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    libfuncs::LibfuncHelper,
    libfuncs::LibfuncHelper,
//    metadata::{
    metadata::{
//        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
        realloc_bindings::ReallocBindingsMeta, snapshot_clones::SnapshotClonesMeta, MetadataStorage,
//    },
    },
//    types::TypeBuilder,
    types::TypeBuilder,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoAndTypeConcreteType,
        types::InfoAndTypeConcreteType,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        llvm::{self, r#type::pointer},
        llvm::{self, r#type::pointer},
//        ods,
        ods,
//    },
    },
//    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value},
//    Context,
    Context,
//};
};
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoAndTypeConcreteType>,
    info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    metadata
    metadata
//        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
        .get_or_insert_with::<SnapshotClonesMeta>(SnapshotClonesMeta::default)
//        .register(
        .register(
//            info.self_ty().clone(),
            info.self_ty().clone(),
//            snapshot_take,
            snapshot_take,
//            InfoAndTypeConcreteType {
            InfoAndTypeConcreteType {
//                info: info.info.clone(),
                info: info.info.clone(),
//                ty: info.ty.clone(),
                ty: info.ty.clone(),
//            },
            },
//        );
        );
//

//    Ok(llvm::r#type::pointer(context, 0))
    Ok(llvm::r#type::pointer(context, 0))
//}
}
//

//#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
//fn snapshot_take<'ctx, 'this>(
fn snapshot_take<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoAndTypeConcreteType>,
    info: WithSelf<InfoAndTypeConcreteType>,
//    src_value: Value<'ctx, 'this>,
    src_value: Value<'ctx, 'this>,
//) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
) -> Result<(&'this Block<'ctx>, Value<'ctx, 'this>)> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let inner_snapshot_take = metadata
    let inner_snapshot_take = metadata
//        .get::<SnapshotClonesMeta>()
        .get::<SnapshotClonesMeta>()
//        .and_then(|meta| meta.wrap_invoke(&info.ty));
        .and_then(|meta| meta.wrap_invoke(&info.ty));
//

//    let inner_type = registry.get_type(&info.ty)?;
    let inner_type = registry.get_type(&info.ty)?;
//    let inner_layout = inner_type.layout(registry)?;
    let inner_layout = inner_type.layout(registry)?;
//    let inner_ty = inner_type.build(context, helper, registry, metadata, info.self_ty())?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, info.self_ty())?;
//

//    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;
    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_ptr = entry
    let dst_ptr = entry
//        .append_operation(ReallocBindingsMeta::realloc(
        .append_operation(ReallocBindingsMeta::realloc(
//            context, ptr, value_len, location,
            context, ptr, value_len, location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    match inner_snapshot_take {
    match inner_snapshot_take {
//        Some(inner_snapshot_take) => {
        Some(inner_snapshot_take) => {
//            let value = entry.load(
            let value = entry.load(
//                context,
                context,
//                location,
                location,
//                src_value,
                src_value,
//                inner_ty,
                inner_ty,
//                Some(inner_layout.align()),
                Some(inner_layout.align()),
//            )?;
            )?;
//

//            let (entry, value) =
            let (entry, value) =
//                inner_snapshot_take(context, registry, entry, location, helper, metadata, value)?;
                inner_snapshot_take(context, registry, entry, location, helper, metadata, value)?;
//

//            entry.store(
            entry.store(
//                context,
                context,
//                location,
                location,
//                dst_ptr,
                dst_ptr,
//                value,
                value,
//                Some(inner_layout.align()),
                Some(inner_layout.align()),
//            );
            );
//        }
        }
//        None => {
        None => {
//            entry.append_operation(
            entry.append_operation(
//                ods::llvm::intr_memcpy(
                ods::llvm::intr_memcpy(
//                    context,
                    context,
//                    dst_ptr,
                    dst_ptr,
//                    src_value,
                    src_value,
//                    value_len,
                    value_len,
//                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            );
            );
//        }
        }
//    }
    }
//

//    Ok((entry, dst_ptr))
    Ok((entry, dst_ptr))
//}
}
