////! # Array type
//! # Array type
////!
//!
////! An array type is a dynamically allocated list of items.
//! An array type is a dynamically allocated list of items.
////!
//!
////! ## Layout
//! ## Layout
////!
//!
////! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
////! its capacity:
//! its capacity:
////!
//!
////! | Index | Type           | Description              |
//! | Index | Type           | Description              |
////! | ----- | -------------- | ------------------------ |
//! | ----- | -------------- | ------------------------ |
////! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
////! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array start offset[^2].  |
////! |   1   | `i32`          | Array end offset[^2].    |
//! |   1   | `i32`          | Array end offset[^2].    |
////! |   2   | `i32`          | Allocated capacity[^2].  |
//! |   2   | `i32`          | Allocated capacity[^2].  |
////!
//!
////! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
////! [^2]: Those numbers are number of items, **not bytes**.
//! [^2]: Those numbers are number of items, **not bytes**.
//

//use super::{TypeBuilder, WithSelf};
use super::{TypeBuilder, WithSelf};
//use crate::{
use crate::{
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
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
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
//        arith, cf,
        arith, cf,
//        llvm::{self, r#type::pointer, LoadStoreOptions},
        llvm::{self, r#type::pointer, LoadStoreOptions},
//        ods,
        ods,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location, Module, Type, Value, ValueLike,
        Block, Location, Module, Type, Value, ValueLike,
//    },
    },
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

//    let ptr_ty = llvm::r#type::pointer(context, 0);
    let ptr_ty = llvm::r#type::pointer(context, 0);
//    let len_ty = IntegerType::new(context, 32).into();
    let len_ty = IntegerType::new(context, 32).into();
//

//    Ok(llvm::r#type::r#struct(
    Ok(llvm::r#type::r#struct(
//        context,
        context,
//        &[ptr_ty, len_ty, len_ty, len_ty],
        &[ptr_ty, len_ty, len_ty, len_ty],
//        false,
        false,
//    ))
    ))
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

//    let elem_snapshot_take = metadata
    let elem_snapshot_take = metadata
//        .get::<SnapshotClonesMeta>()
        .get::<SnapshotClonesMeta>()
//        .and_then(|meta| meta.wrap_invoke(&info.ty));
        .and_then(|meta| meta.wrap_invoke(&info.ty));
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//    let elem_stride = elem_layout.pad_to_align().size();
    let elem_stride = elem_layout.pad_to_align().size();
//    let elem_ty = elem_ty.build(context, helper, registry, metadata, &info.ty)?;
    let elem_ty = elem_ty.build(context, helper, registry, metadata, &info.ty)?;
//

//    let src_ptr = entry
    let src_ptr = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            src_value,
            src_value,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let array_start = entry
    let array_start = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            src_value,
            src_value,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            IntegerType::new(context, 32).into(),
            IntegerType::new(context, 32).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let array_end = entry
    let array_end = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            src_value,
            src_value,
//            DenseI64ArrayAttribute::new(context, &[2]),
            DenseI64ArrayAttribute::new(context, &[2]),
//            IntegerType::new(context, 32).into(),
            IntegerType::new(context, 32).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let elem_stride = entry
    let elem_stride = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(
            IntegerAttribute::new(
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                elem_stride.try_into()?,
                elem_stride.try_into()?,
//            )
            )
//            .into(),
            .into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let array_ty = registry.build_type(context, helper, registry, metadata, info.self_ty())?;
    let array_ty = registry.build_type(context, helper, registry, metadata, info.self_ty())?;
//

//    let array_len: Value = entry
    let array_len: Value = entry
//        .append_operation(arith::subi(array_end, array_start, location))
        .append_operation(arith::subi(array_end, array_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let k0 = entry
    let k0 = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(array_len.r#type(), 0).into(),
            IntegerAttribute::new(array_len.r#type(), 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let is_len_zero = entry
    let is_len_zero = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            arith::CmpiPredicate::Eq,
            arith::CmpiPredicate::Eq,
//            array_len,
            array_len,
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let null_ptr = entry
    let null_ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let block_realloc = helper.append_block(Block::new(&[]));
    let block_realloc = helper.append_block(Block::new(&[]));
//    let block_finish =
    let block_finish =
//        helper.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
        helper.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
//

//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_len_zero,
        is_len_zero,
//        block_finish,
        block_finish,
//        block_realloc,
        block_realloc,
//        &[null_ptr],
        &[null_ptr],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        // realloc
        // realloc
//        let dst_len_bytes: Value = {
        let dst_len_bytes: Value = {
//            let array_len = block_realloc
            let array_len = block_realloc
//                .append_operation(arith::extui(
                .append_operation(arith::extui(
//                    array_len,
                    array_len,
//                    IntegerType::new(context, 64).into(),
                    IntegerType::new(context, 64).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            block_realloc
            block_realloc
//                .append_operation(arith::muli(array_len, elem_stride, location))
                .append_operation(arith::muli(array_len, elem_stride, location))
//                .result(0)?
                .result(0)?
//                .into()
                .into()
//        };
        };
//

//        let dst_ptr = {
        let dst_ptr = {
//            let dst_ptr = null_ptr;
            let dst_ptr = null_ptr;
//

//            block_realloc
            block_realloc
//                .append_operation(ReallocBindingsMeta::realloc(
                .append_operation(ReallocBindingsMeta::realloc(
//                    context,
                    context,
//                    dst_ptr,
                    dst_ptr,
//                    dst_len_bytes,
                    dst_len_bytes,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into()
                .into()
//        };
        };
//

//        let src_ptr_offset = {
        let src_ptr_offset = {
//            let array_start = block_realloc
            let array_start = block_realloc
//                .append_operation(arith::extui(
                .append_operation(arith::extui(
//                    array_start,
                    array_start,
//                    IntegerType::new(context, 64).into(),
                    IntegerType::new(context, 64).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            block_realloc
            block_realloc
//                .append_operation(arith::muli(array_start, elem_stride, location))
                .append_operation(arith::muli(array_start, elem_stride, location))
//                .result(0)?
                .result(0)?
//                .into()
                .into()
//        };
        };
//        let src_ptr = block_realloc
        let src_ptr = block_realloc
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                src_ptr,
                src_ptr,
//                &[src_ptr_offset],
                &[src_ptr_offset],
//                IntegerType::new(context, 8).into(),
                IntegerType::new(context, 8).into(),
//                llvm::r#type::pointer(context, 0),
                llvm::r#type::pointer(context, 0),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        match elem_snapshot_take {
        match elem_snapshot_take {
//            Some(elem_snapshot_take) => {
            Some(elem_snapshot_take) => {
//                let value = block_realloc
                let value = block_realloc
//                    .append_operation(llvm::load(
                    .append_operation(llvm::load(
//                        context,
                        context,
//                        src_ptr,
                        src_ptr,
//                        elem_ty,
                        elem_ty,
//                        location,
                        location,
//                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            elem_layout.align() as i64,
                            elem_layout.align() as i64,
//                        ))),
                        ))),
//                    ))
                    ))
//                    .result(0)?
                    .result(0)?
//                    .into();
                    .into();
//

//                let (block_relloc, value) = elem_snapshot_take(
                let (block_relloc, value) = elem_snapshot_take(
//                    context,
                    context,
//                    registry,
                    registry,
//                    block_realloc,
                    block_realloc,
//                    location,
                    location,
//                    helper,
                    helper,
//                    metadata,
                    metadata,
//                    value,
                    value,
//                )?;
                )?;
//

//                block_relloc.append_operation(llvm::store(
                block_relloc.append_operation(llvm::store(
//                    context,
                    context,
//                    value,
                    value,
//                    dst_ptr,
                    dst_ptr,
//                    location,
                    location,
//                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                    LoadStoreOptions::new().align(Some(IntegerAttribute::new(
//                        IntegerType::new(context, 64).into(),
                        IntegerType::new(context, 64).into(),
//                        elem_layout.align() as i64,
                        elem_layout.align() as i64,
//                    ))),
                    ))),
//                ));
                ));
//                block_relloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
                block_relloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
//            }
            }
//            None => {
            None => {
//                block_realloc.append_operation(
                block_realloc.append_operation(
//                    ods::llvm::intr_memcpy(
                    ods::llvm::intr_memcpy(
//                        context,
                        context,
//                        dst_ptr,
                        dst_ptr,
//                        src_ptr,
                        src_ptr,
//                        dst_len_bytes,
                        dst_len_bytes,
//                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
//                        location,
                        location,
//                    )
                    )
//                    .into(),
                    .into(),
//                );
                );
//                block_realloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
                block_realloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
//            }
            }
//        }
        }
//    }
    }
//

//    let dst_value = block_finish
    let dst_value = block_finish
//        .append_operation(llvm::undef(array_ty, location))
        .append_operation(llvm::undef(array_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_ptr = block_finish.argument(0)?.into();
    let dst_ptr = block_finish.argument(0)?.into();
//

//    let k0 = block_finish
    let k0 = block_finish
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
            IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_value = block_finish
    let dst_value = block_finish
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            dst_value,
            dst_value,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            dst_ptr,
            dst_ptr,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_value = block_finish
    let dst_value = block_finish
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            dst_value,
            dst_value,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            k0,
            k0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_value = block_finish
    let dst_value = block_finish
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            dst_value,
            dst_value,
//            DenseI64ArrayAttribute::new(context, &[2]),
            DenseI64ArrayAttribute::new(context, &[2]),
//            array_len,
            array_len,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let dst_value = block_finish
    let dst_value = block_finish
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            dst_value,
            dst_value,
//            DenseI64ArrayAttribute::new(context, &[3]),
            DenseI64ArrayAttribute::new(context, &[3]),
//            array_len,
            array_len,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    Ok((block_finish, dst_value))
    Ok((block_finish, dst_value))
//}
}
