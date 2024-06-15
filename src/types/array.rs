//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! its capacity:
//!
//! | Index | Type           | Description              |
//! | ----- | -------------- | ------------------------ |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array end offset[^2].    |
//! |   2   | `i32`          | Allocated capacity[^2].  |
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::block_ext::BlockExt;
use crate::{
    error::Result,
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
        arith, cf,
        llvm::{self, r#type::pointer},
        ods,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type, Value,
        ValueLike,
    },
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

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    Ok(llvm::r#type::r#struct(
        context,
        &[ptr_ty, len_ty, len_ty, len_ty],
        false,
    ))
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

    let elem_snapshot_take = metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.ty));

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;
    let elem_stride = elem_layout.pad_to_align().size();
    let elem_ty = elem_ty.build(context, helper, registry, metadata, &info.ty)?;

    let src_ptr = entry.extract_value(
        context,
        location,
        src_value,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let array_start = entry.extract_value(
        context,
        location,
        src_value,
        IntegerType::new(context, 32).into(),
        1,
    )?;
    let array_end = entry.extract_value(
        context,
        location,
        src_value,
        IntegerType::new(context, 32).into(),
        2,
    )?;

    let elem_stride = entry.const_int(context, location, elem_stride, 64)?;

    let array_ty = registry.build_type(context, helper, registry, metadata, info.self_ty())?;

    let array_len: Value = entry.append_op_result(arith::subi(array_end, array_start, location))?;

    let k0 = entry.const_int_from_type(context, location, 0, array_len.r#type())?;
    let is_len_zero = entry.append_op_result(arith::cmpi(
        context,
        arith::CmpiPredicate::Eq,
        array_len,
        k0,
        location,
    ))?;

    let null_ptr = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    let block_realloc = helper.append_block(Block::new(&[]));
    let block_finish =
        helper.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    entry.append_operation(cf::cond_br(
        context,
        is_len_zero,
        block_finish,
        block_realloc,
        &[null_ptr],
        &[],
        location,
    ));

    {
        // realloc
        let dst_len_bytes: Value = {
            let array_len = block_realloc.append_op_result(arith::extui(
                array_len,
                IntegerType::new(context, 64).into(),
                location,
            ))?;

            block_realloc.append_op_result(arith::muli(array_len, elem_stride, location))?
        };

        let dst_ptr = {
            let dst_ptr = null_ptr;

            block_realloc.append_op_result(ReallocBindingsMeta::realloc(
                context,
                dst_ptr,
                dst_len_bytes,
                location,
            ))?
        };

        let src_ptr_offset = {
            let array_start = block_realloc.append_op_result(arith::extui(
                array_start,
                IntegerType::new(context, 64).into(),
                location,
            ))?;

            block_realloc.append_op_result(arith::muli(array_start, elem_stride, location))?
        };
        let src_ptr = block_realloc.append_op_result(llvm::get_element_ptr_dynamic(
            context,
            src_ptr,
            &[src_ptr_offset],
            IntegerType::new(context, 8).into(),
            llvm::r#type::pointer(context, 0),
            location,
        ))?;

        match elem_snapshot_take {
            Some(elem_snapshot_take) => {
                let value = block_realloc.load(
                    context,
                    location,
                    src_ptr,
                    elem_ty,
                    Some(elem_layout.align()),
                )?;

                let (block_relloc, value) = elem_snapshot_take(
                    context,
                    registry,
                    block_realloc,
                    location,
                    helper,
                    metadata,
                    value,
                )?;

                block_relloc.store(context, location, dst_ptr, value, Some(elem_layout.align()))?;
                block_relloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
            }
            None => {
                block_realloc.append_operation(
                    ods::llvm::intr_memcpy(
                        context,
                        dst_ptr,
                        src_ptr,
                        dst_len_bytes,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );
                block_realloc.append_operation(cf::br(block_finish, &[dst_ptr], location));
            }
        }
    }

    let dst_value = block_finish.append_op_result(llvm::undef(array_ty, location))?;
    let dst_ptr = block_finish.argument(0)?.into();

    let k0 = block_finish.const_int(context, location, 0, 32)?;
    let dst_value = block_finish.insert_value(context, location, dst_value, dst_ptr, 0)?;
    let dst_value = block_finish.insert_value(context, location, dst_value, k0, 1)?;
    let dst_value = block_finish.insert_value(context, location, dst_value, array_len, 2)?;
    let dst_value = block_finish.insert_value(context, location, dst_value, array_len, 3)?;

    Ok((block_finish, dst_value))
}
