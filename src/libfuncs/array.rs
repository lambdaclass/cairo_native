////! # Array libfuncs
//! # Array libfuncs
//

//// TODO: A future possible improvement would be to put the array behind a double pointer and a
// TODO: A future possible improvement would be to put the array behind a double pointer and a
////   reference counter, to avoid unnecessary clones.
//   reference counter, to avoid unnecessary clones.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::Result,
    error::Result,
//    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
//    types::TypeBuilder,
    types::TypeBuilder,
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        array::ArrayConcreteLibfunc,
        array::ArrayConcreteLibfunc,
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
//        ConcreteLibfunc,
        ConcreteLibfunc,
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
//        arith::{self, CmpiPredicate},
        arith::{self, CmpiPredicate},
//        cf,
        cf,
//        llvm::{self, r#type::pointer},
        llvm::{self, r#type::pointer},
//        ods,
        ods,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location, Value, ValueLike,
        Block, Location, Value, ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
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
//    selector: &ArrayConcreteLibfunc,
    selector: &ArrayConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        ArrayConcreteLibfunc::New(info) => {
        ArrayConcreteLibfunc::New(info) => {
//            build_new(context, registry, entry, location, helper, metadata, info)
            build_new(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::Append(info) => {
        ArrayConcreteLibfunc::Append(info) => {
//            build_append(context, registry, entry, location, helper, metadata, info)
            build_append(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::PopFront(info) => {
        ArrayConcreteLibfunc::PopFront(info) => {
//            build_pop_front(context, registry, entry, location, helper, metadata, info)
            build_pop_front(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::PopFrontConsume(info) => {
        ArrayConcreteLibfunc::PopFrontConsume(info) => {
//            build_pop_front_consume(context, registry, entry, location, helper, metadata, info)
            build_pop_front_consume(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::Get(info) => {
        ArrayConcreteLibfunc::Get(info) => {
//            build_get(context, registry, entry, location, helper, metadata, info)
            build_get(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::Slice(info) => {
        ArrayConcreteLibfunc::Slice(info) => {
//            build_slice(context, registry, entry, location, helper, metadata, info)
            build_slice(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::Len(info) => {
        ArrayConcreteLibfunc::Len(info) => {
//            build_len(context, registry, entry, location, helper, metadata, info)
            build_len(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::SnapshotPopFront(info) => {
        ArrayConcreteLibfunc::SnapshotPopFront(info) => {
//            build_snapshot_pop_front(context, registry, entry, location, helper, metadata, info)
            build_snapshot_pop_front(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::SnapshotPopBack(info) => {
        ArrayConcreteLibfunc::SnapshotPopBack(info) => {
//            build_snapshot_pop_back(context, registry, entry, location, helper, metadata, info)
            build_snapshot_pop_back(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ArrayConcreteLibfunc::SpanFromTuple(info) => {
        ArrayConcreteLibfunc::SpanFromTuple(info) => {
//            build_span_from_tuple(context, registry, entry, location, helper, metadata, info)
            build_span_from_tuple(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `array_new` libfunc.
/// Generate MLIR operations for the `array_new` libfunc.
//pub fn build_new<'ctx, 'this>(
pub fn build_new<'ctx, 'this>(
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let k0 = entry
    let k0 = entry
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
//

//    let value = entry
    let value = entry
//        .append_operation(llvm::undef(array_ty, location))
        .append_operation(llvm::undef(array_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let value = entry
    let value = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            value,
            value,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            ptr,
            ptr,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let value = entry
    let value = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            value,
            value,
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
//    let value = entry
    let value = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            value,
            value,
//            DenseI64ArrayAttribute::new(context, &[2]),
            DenseI64ArrayAttribute::new(context, &[2]),
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
//    let value = entry
    let value = entry
//        .append_operation(llvm::insert_value(
        .append_operation(llvm::insert_value(
//            context,
            context,
//            value,
            value,
//            DenseI64ArrayAttribute::new(context, &[3]),
            DenseI64ArrayAttribute::new(context, &[3]),
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

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_append` libfunc.
/// Generate MLIR operations for the `array_append` libfunc.
//pub fn build_append<'ctx, 'this>(
pub fn build_append<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    // Algorithm:
    // Algorithm:
//    //   - If array_end < capacity, then append.
    //   - If array_end < capacity, then append.
//    //   - If array_end == capacity:
    //   - If array_end == capacity:
//    //     - If array_start == 0: realloc, then append.
    //     - If array_start == 0: realloc, then append.
//    //     - If array_start != 0: memmove, then append.
    //     - If array_start != 0: memmove, then append.
//

//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[0].ty,
        &info.param_signatures()[0].ty,
//    )?;
    )?;
//

//    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//    let elem_stride = elem_layout.pad_to_align().size();
    let elem_stride = elem_layout.pad_to_align().size();
//

//    let k1 = entry.const_int(context, location, 1, 32)?;
    let k1 = entry.const_int(context, location, 1, 32)?;
//

//    let elem_stride = entry.const_int(context, location, elem_stride, 64)?;
    let elem_stride = entry.const_int(context, location, elem_stride, 64)?;
//

//    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;
//    let array_capacity =
    let array_capacity =
//        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 3)?;
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 3)?;
//

//    let has_tail_space = entry
    let has_tail_space = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Ult,
            CmpiPredicate::Ult,
//            array_end,
            array_end,
//            array_capacity,
            array_capacity,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let handle_block = helper.append_block(Block::new(&[]));
    let handle_block = helper.append_block(Block::new(&[]));
//    let memmove_block = helper.append_block(Block::new(&[]));
    let memmove_block = helper.append_block(Block::new(&[]));
//    let realloc_block = helper.append_block(Block::new(&[]));
    let realloc_block = helper.append_block(Block::new(&[]));
//    let append_block = helper.append_block(Block::new(&[(array_ty, location)]));
    let append_block = helper.append_block(Block::new(&[(array_ty, location)]));
//

//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        has_tail_space,
        has_tail_space,
//        append_block,
        append_block,
//        handle_block,
        handle_block,
//        &[entry.argument(0)?.into()],
        &[entry.argument(0)?.into()],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        let k0 = handle_block.const_int(context, location, 0, 32)?;
        let k0 = handle_block.const_int(context, location, 0, 32)?;
//        let array_start =
        let array_start =
//            handle_block.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
            handle_block.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
//

//        let has_head_space = handle_block
        let has_head_space = handle_block
//            .append_operation(arith::cmpi(
            .append_operation(arith::cmpi(
//                context,
                context,
//                CmpiPredicate::Ne,
                CmpiPredicate::Ne,
//                array_start,
                array_start,
//                k0,
                k0,
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        handle_block.append_operation(cf::cond_br(
        handle_block.append_operation(cf::cond_br(
//            context,
            context,
//            has_head_space,
            has_head_space,
//            memmove_block,
            memmove_block,
//            realloc_block,
            realloc_block,
//            &[],
            &[],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//    }
    }
//

//    {
    {
//        let array_start =
        let array_start =
//            memmove_block.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
            memmove_block.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
//

//        let start_offset = memmove_block
        let start_offset = memmove_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                array_start,
                array_start,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let start_offset = memmove_block
        let start_offset = memmove_block
//            .append_operation(arith::muli(start_offset, elem_stride, location))
            .append_operation(arith::muli(start_offset, elem_stride, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let dst_ptr =
        let dst_ptr =
//            memmove_block.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
            memmove_block.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
//        let src_ptr = memmove_block
        let src_ptr = memmove_block
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                dst_ptr,
                dst_ptr,
//                &[start_offset],
                &[start_offset],
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

//        let array_len = memmove_block
        let array_len = memmove_block
//            .append_operation(arith::subi(array_end, array_start, location))
            .append_operation(arith::subi(array_end, array_start, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let memmove_len = memmove_block
        let memmove_len = memmove_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                array_len,
                array_len,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let memmove_len = memmove_block
        let memmove_len = memmove_block
//            .append_operation(arith::muli(memmove_len, elem_stride, location))
            .append_operation(arith::muli(memmove_len, elem_stride, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        memmove_block.append_operation(
        memmove_block.append_operation(
//            ods::llvm::intr_memmove(
            ods::llvm::intr_memmove(
//                context,
                context,
//                dst_ptr,
                dst_ptr,
//                src_ptr,
                src_ptr,
//                memmove_len,
                memmove_len,
//                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
//                location,
                location,
//            )
            )
//            .into(),
            .into(),
//        );
        );
//

//        let k0 = memmove_block.const_int_from_type(context, location, 0, len_ty)?;
        let k0 = memmove_block.const_int_from_type(context, location, 0, len_ty)?;
//        let value =
        let value =
//            memmove_block.insert_value(context, location, entry.argument(0)?.into(), k0, 1)?;
            memmove_block.insert_value(context, location, entry.argument(0)?.into(), k0, 1)?;
//        let value = memmove_block.insert_value(context, location, value, array_len, 2)?;
        let value = memmove_block.insert_value(context, location, value, array_len, 2)?;
//

//        memmove_block.append_operation(cf::br(append_block, &[value], location));
        memmove_block.append_operation(cf::br(append_block, &[value], location));
//    }
    }
//

//    {
    {
//        let k8 = realloc_block.const_int(context, location, 8, 32)?;
        let k8 = realloc_block.const_int(context, location, 8, 32)?;
//        let k1024 = realloc_block.const_int(context, location, 1024, 32)?;
        let k1024 = realloc_block.const_int(context, location, 1024, 32)?;
//

//        // Array allocation growth formula:
        // Array allocation growth formula:
//        //   new_len = max(8, old_len + min(1024, 2 * old_len));
        //   new_len = max(8, old_len + min(1024, 2 * old_len));
//        let new_capacity = realloc_block
        let new_capacity = realloc_block
//            .append_operation(arith::shli(array_end, k1, location))
            .append_operation(arith::shli(array_end, k1, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let new_capacity = realloc_block
        let new_capacity = realloc_block
//            .append_operation(arith::minui(new_capacity, k1024, location))
            .append_operation(arith::minui(new_capacity, k1024, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let new_capacity = realloc_block
        let new_capacity = realloc_block
//            .append_operation(arith::addi(new_capacity, array_end, location))
            .append_operation(arith::addi(new_capacity, array_end, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let new_capacity = realloc_block
        let new_capacity = realloc_block
//            .append_operation(arith::maxui(new_capacity, k8, location))
            .append_operation(arith::maxui(new_capacity, k8, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let realloc_size = {
        let realloc_size = {
//            let new_capacity = realloc_block
            let new_capacity = realloc_block
//                .append_operation(arith::extui(
                .append_operation(arith::extui(
//                    new_capacity,
                    new_capacity,
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
//            realloc_block
            realloc_block
//                .append_operation(arith::muli(new_capacity, elem_stride, location))
                .append_operation(arith::muli(new_capacity, elem_stride, location))
//                .result(0)?
                .result(0)?
//                .into()
                .into()
//        };
        };
//

//        let ptr =
        let ptr =
//            realloc_block.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
            realloc_block.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
//        let ptr = realloc_block
        let ptr = realloc_block
//            .append_operation(ReallocBindingsMeta::realloc(
            .append_operation(ReallocBindingsMeta::realloc(
//                context,
                context,
//                ptr,
                ptr,
//                realloc_size,
                realloc_size,
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        // No need to memmove, guaranteed by the fact that if we needed to memmove we'd have gone
        // No need to memmove, guaranteed by the fact that if we needed to memmove we'd have gone
//        // through the memmove block instead of reallocating.
        // through the memmove block instead of reallocating.
//

//        let value =
        let value =
//            realloc_block.insert_value(context, location, entry.argument(0)?.into(), ptr, 0)?;
            realloc_block.insert_value(context, location, entry.argument(0)?.into(), ptr, 0)?;
//        let value = realloc_block.insert_value(context, location, value, new_capacity, 3)?;
        let value = realloc_block.insert_value(context, location, value, new_capacity, 3)?;
//

//        realloc_block.append_operation(cf::br(append_block, &[value], location));
        realloc_block.append_operation(cf::br(append_block, &[value], location));
//    }
    }
//

//    {
    {
//        let ptr = append_block.extract_value(
        let ptr = append_block.extract_value(
//            context,
            context,
//            location,
            location,
//            append_block.argument(0)?.into(),
            append_block.argument(0)?.into(),
//            ptr_ty,
            ptr_ty,
//            0,
            0,
//        )?;
        )?;
//        let array_end = append_block.extract_value(
        let array_end = append_block.extract_value(
//            context,
            context,
//            location,
            location,
//            append_block.argument(0)?.into(),
            append_block.argument(0)?.into(),
//            len_ty,
            len_ty,
//            2,
            2,
//        )?;
        )?;
//

//        let offset = append_block
        let offset = append_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                array_end,
                array_end,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let offset = append_block
        let offset = append_block
//            .append_operation(arith::muli(offset, elem_stride, location))
            .append_operation(arith::muli(offset, elem_stride, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let ptr = append_block
        let ptr = append_block
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                ptr,
                ptr,
//                &[offset],
                &[offset],
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

//        append_block.store(
        append_block.store(
//            context,
            context,
//            location,
            location,
//            ptr,
            ptr,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            Some(elem_layout.align()),
            Some(elem_layout.align()),
//        );
        );
//

//        let array_len = append_block
        let array_len = append_block
//            .append_operation(arith::addi(array_end, k1, location))
            .append_operation(arith::addi(array_end, k1, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = append_block.insert_value(
        let value = append_block.insert_value(
//            context,
            context,
//            location,
            location,
//            append_block.argument(0)?.into(),
            append_block.argument(0)?.into(),
//            array_len,
            array_len,
//            2,
            2,
//        )?;
        )?;
//

//        append_block.append_operation(helper.br(0, &[value], location));
        append_block.append_operation(helper.br(0, &[value], location));
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_len` libfunc.
/// Generate MLIR operations for the `array_len` libfunc.
//pub fn build_len<'ctx, 'this>(
pub fn build_len<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[0].ty,
        &info.param_signatures()[0].ty,
//    )?;
    )?;
//

//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//    let array_value = entry.argument(0)?.into();
    let array_value = entry.argument(0)?.into();
//

//    let array_start = entry.extract_value(context, location, array_value, len_ty, 1)?;
    let array_start = entry.extract_value(context, location, array_value, len_ty, 1)?;
//    let array_end = entry.extract_value(context, location, array_value, len_ty, 2)?;
    let array_end = entry.extract_value(context, location, array_value, len_ty, 2)?;
//

//    let array_len = entry
    let array_len = entry
//        .append_operation(arith::subi(array_end, array_start, location))
        .append_operation(arith::subi(array_end, array_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(0, &[array_len], location));
    entry.append_operation(helper.br(0, &[array_len], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_get` libfunc.
/// Generate MLIR operations for the `array_get` libfunc.
//pub fn build_get<'ctx, 'this>(
pub fn build_get<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//    let elem_stride = elem_layout.pad_to_align().size();
    let elem_stride = elem_layout.pad_to_align().size();
//

//    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let value = entry.argument(1)?.into();
    let value = entry.argument(1)?.into();
//    let index = entry.argument(2)?.into();
    let index = entry.argument(2)?.into();
//

//    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
//    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
//

//    let array_len = entry
    let array_len = entry
//        .append_operation(arith::subi(array_end, array_start, location))
        .append_operation(arith::subi(array_end, array_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let is_valid = entry
    let is_valid = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Ult,
            CmpiPredicate::Ult,
//            index,
            index,
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

//    let valid_block = helper.append_block(Block::new(&[]));
    let valid_block = helper.append_block(Block::new(&[]));
//    let error_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_valid,
        is_valid,
//        valid_block,
        valid_block,
//        error_block,
        error_block,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
//

//        let index = {
        let index = {
//            let array_start = valid_block
            let array_start = valid_block
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
//            let index = valid_block
            let index = valid_block
//                .append_operation(arith::extui(
                .append_operation(arith::extui(
//                    index,
                    index,
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
//            valid_block
            valid_block
//                .append_operation(arith::addi(array_start, index, location))
                .append_operation(arith::addi(array_start, index, location))
//                .result(0)?
                .result(0)?
//                .into()
                .into()
//        };
        };
//

//        let elem_stride = valid_block.const_int(context, location, elem_stride, 64)?;
        let elem_stride = valid_block.const_int(context, location, elem_stride, 64)?;
//        let elem_offset = valid_block
        let elem_offset = valid_block
//            .append_operation(arith::muli(elem_stride, index, location))
            .append_operation(arith::muli(elem_stride, index, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let elem_ptr = valid_block
        let elem_ptr = valid_block
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                ptr,
                ptr,
//                &[elem_offset],
                &[elem_offset],
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

//        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
//

//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ReallocBindingsMeta::realloc(
            .append_operation(ReallocBindingsMeta::realloc(
//                context, target_ptr, elem_size, location,
                context, target_ptr, elem_size, location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        assert_nonnull(
        assert_nonnull(
//            context,
            context,
//            valid_block,
            valid_block,
//            location,
            location,
//            target_ptr,
            target_ptr,
//            "realloc returned nullptr",
            "realloc returned nullptr",
//        )?;
        )?;
//

//        // TODO: Support clone-only types (those that are not copy).
        // TODO: Support clone-only types (those that are not copy).
//        valid_block.memcpy(context, location, elem_ptr, target_ptr, elem_size);
        valid_block.memcpy(context, location, elem_ptr, target_ptr, elem_size);
//

//        valid_block.append_operation(helper.br(0, &[range_check, target_ptr], location));
        valid_block.append_operation(helper.br(0, &[range_check, target_ptr], location));
//    }
    }
//

//    error_block.append_operation(helper.br(1, &[range_check], location));
    error_block.append_operation(helper.br(1, &[range_check], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_pop_front` libfunc.
/// Generate MLIR operations for the `array_pop_front` libfunc.
//pub fn build_pop_front<'ctx, 'this>(
pub fn build_pop_front<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[0].ty,
        &info.param_signatures()[0].ty,
//    )?;
    )?;
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//

//    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let value = entry.argument(0)?.into();
    let value = entry.argument(0)?.into();
//

//    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
//    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
//

//    let is_empty = entry
    let is_empty = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            array_start,
            array_start,
//            array_end,
            array_end,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let valid_block = helper.append_block(Block::new(&[]));
    let valid_block = helper.append_block(Block::new(&[]));
//    let empty_block = helper.append_block(Block::new(&[]));
    let empty_block = helper.append_block(Block::new(&[]));
//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_empty,
        is_empty,
//        empty_block,
        empty_block,
//        valid_block,
        valid_block,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
//

//        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
//        let elem_offset = valid_block
        let elem_offset = valid_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                array_start,
                array_start,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let elem_offset = valid_block
        let elem_offset = valid_block
//            .append_operation(arith::muli(elem_offset, elem_size, location))
            .append_operation(arith::muli(elem_offset, elem_size, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let ptr = valid_block
        let ptr = valid_block
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                ptr,
                ptr,
//                &[elem_offset],
                &[elem_offset],
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

//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ReallocBindingsMeta::realloc(
            .append_operation(ReallocBindingsMeta::realloc(
//                context, target_ptr, elem_size, location,
                context, target_ptr, elem_size, location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        assert_nonnull(
        assert_nonnull(
//            context,
            context,
//            valid_block,
            valid_block,
//            location,
            location,
//            target_ptr,
            target_ptr,
//            "realloc returned nullptr",
            "realloc returned nullptr",
//        )?;
        )?;
//

//        valid_block.memcpy(context, location, ptr, target_ptr, elem_size);
        valid_block.memcpy(context, location, ptr, target_ptr, elem_size);
//

//        let k1 = valid_block.const_int(context, location, 1, 32)?;
        let k1 = valid_block.const_int(context, location, 1, 32)?;
//        let new_start = valid_block
        let new_start = valid_block
//            .append_operation(arith::addi(array_start, k1, location))
            .append_operation(arith::addi(array_start, k1, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = valid_block.insert_value(context, location, value, new_start, 1)?;
        let value = valid_block.insert_value(context, location, value, new_start, 1)?;
//

//        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
//    }
    }
//

//    empty_block.append_operation(helper.br(1, &[value], location));
    empty_block.append_operation(helper.br(1, &[value], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_pop_front_consume` libfunc.
/// Generate MLIR operations for the `array_pop_front_consume` libfunc.
//pub fn build_pop_front_consume<'ctx, 'this>(
pub fn build_pop_front_consume<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    // Equivalent to `array_pop_front_consume` for our purposes.
    // Equivalent to `array_pop_front_consume` for our purposes.
//    build_pop_front(context, registry, entry, location, helper, metadata, info)
    build_pop_front(context, registry, entry, location, helper, metadata, info)
//}
}
//

///// Generate MLIR operations for the `array_snapshot_pop_front` libfunc.
/// Generate MLIR operations for the `array_snapshot_pop_front` libfunc.
//pub fn build_snapshot_pop_front<'ctx, 'this>(
pub fn build_snapshot_pop_front<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    build_pop_front(context, registry, entry, location, helper, metadata, info)
    build_pop_front(context, registry, entry, location, helper, metadata, info)
//}
}
//

///// Generate MLIR operations for the `array_snapshot_pop_back` libfunc.
/// Generate MLIR operations for the `array_snapshot_pop_back` libfunc.
//pub fn build_snapshot_pop_back<'ctx, 'this>(
pub fn build_snapshot_pop_back<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[0].ty,
        &info.param_signatures()[0].ty,
//    )?;
    )?;
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//

//    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let value = entry.argument(0)?.into();
    let value = entry.argument(0)?.into();
//

//    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
    let array_start = entry.extract_value(context, location, value, len_ty, 1)?;
//    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;
//    let is_empty = entry
    let is_empty = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            array_start,
            array_start,
//            array_end,
            array_end,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let valid_block = helper.append_block(Block::new(&[]));
    let valid_block = helper.append_block(Block::new(&[]));
//    let empty_block = helper.append_block(Block::new(&[]));
    let empty_block = helper.append_block(Block::new(&[]));
//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_empty,
        is_empty,
//        empty_block,
        empty_block,
//        valid_block,
        valid_block,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        let k1 = valid_block.const_int(context, location, 1, 32)?;
        let k1 = valid_block.const_int(context, location, 1, 32)?;
//        let new_end = valid_block
        let new_end = valid_block
//            .append_operation(arith::subi(array_end, k1, location))
            .append_operation(arith::subi(array_end, k1, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
        let ptr = valid_block.extract_value(context, location, value, ptr_ty, 0)?;
//

//        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
        let elem_size = valid_block.const_int(context, location, elem_layout.size(), 64)?;
//        let elem_offset = valid_block
        let elem_offset = valid_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                new_end,
                new_end,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let elem_offset = valid_block
        let elem_offset = valid_block
//            .append_operation(arith::muli(elem_offset, elem_size, location))
            .append_operation(arith::muli(elem_offset, elem_size, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let ptr = valid_block
        let ptr = valid_block
//            .append_operation(llvm::get_element_ptr_dynamic(
            .append_operation(llvm::get_element_ptr_dynamic(
//                context,
                context,
//                ptr,
                ptr,
//                &[elem_offset],
                &[elem_offset],
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

//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
            .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let target_ptr = valid_block
        let target_ptr = valid_block
//            .append_operation(ReallocBindingsMeta::realloc(
            .append_operation(ReallocBindingsMeta::realloc(
//                context, target_ptr, elem_size, location,
                context, target_ptr, elem_size, location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        assert_nonnull(
        assert_nonnull(
//            context,
            context,
//            valid_block,
            valid_block,
//            location,
            location,
//            target_ptr,
            target_ptr,
//            "realloc returned nullptr",
            "realloc returned nullptr",
//        )?;
        )?;
//

//        valid_block.memcpy(context, location, ptr, target_ptr, elem_size);
        valid_block.memcpy(context, location, ptr, target_ptr, elem_size);
//

//        let value = valid_block.insert_value(context, location, value, new_end, 2)?;
        let value = valid_block.insert_value(context, location, value, new_end, 2)?;
//

//        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
//    }
    }
//

//    empty_block.append_operation(helper.br(1, &[value], location));
    empty_block.append_operation(helper.br(1, &[value], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `array_slice` libfunc.
/// Generate MLIR operations for the `array_slice` libfunc.
//pub fn build_slice<'ctx, 'this>(
pub fn build_slice<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//

//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let elem_ty = registry.get_type(&info.ty)?;
    let elem_ty = registry.get_type(&info.ty)?;
//    let elem_layout = elem_ty.layout(registry)?;
    let elem_layout = elem_ty.layout(registry)?;
//

//    let slice_since = entry.argument(2)?.into();
    let slice_since = entry.argument(2)?.into();
//    let slice_length = entry.argument(3)?.into();
    let slice_length = entry.argument(3)?.into();
//

//    let slice_until = entry
    let slice_until = entry
//        .append_operation(arith::addi(slice_since, slice_length, location))
        .append_operation(arith::addi(slice_since, slice_length, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let array_start =
    let array_start =
//        entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 1)?;
        entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 1)?;
//    let array_end = entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 2)?;
    let array_end = entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 2)?;
//

//    let slice_since = entry
    let slice_since = entry
//        .append_operation(arith::addi(slice_since, array_start, location))
        .append_operation(arith::addi(slice_since, array_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let slice_until = entry
    let slice_until = entry
//        .append_operation(arith::addi(slice_until, array_start, location))
        .append_operation(arith::addi(slice_until, array_start, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let lhs_bound = entry
    let lhs_bound = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Uge,
            CmpiPredicate::Uge,
//            slice_since,
            slice_since,
//            array_start,
            array_start,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let rhs_bound = entry
    let rhs_bound = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Ule,
            CmpiPredicate::Ule,
//            slice_until,
            slice_until,
//            array_end,
            array_end,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let is_fully_contained = entry
    let is_fully_contained = entry
//        .append_operation(arith::andi(lhs_bound, rhs_bound, location))
        .append_operation(arith::andi(lhs_bound, rhs_bound, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let slice_block = helper.append_block(Block::new(&[]));
    let slice_block = helper.append_block(Block::new(&[]));
//    let error_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        is_fully_contained,
        is_fully_contained,
//        slice_block,
        slice_block,
//        error_block,
        error_block,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    {
    {
//        let elem_size =
        let elem_size =
//            slice_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;
            slice_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;
//        let dst_size = slice_block
        let dst_size = slice_block
//            .append_operation(arith::extui(
            .append_operation(arith::extui(
//                slice_length,
                slice_length,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let dst_size = slice_block.append_op_result(arith::muli(dst_size, elem_size, location))?;
        let dst_size = slice_block.append_op_result(arith::muli(dst_size, elem_size, location))?;
//

//        let dst_ptr = slice_block.append_op_result(
        let dst_ptr = slice_block.append_op_result(
//            ods::llvm::mlir_zero(context, pointer(context, 0), location).into(),
            ods::llvm::mlir_zero(context, pointer(context, 0), location).into(),
//        )?;
        )?;
//        let dst_ptr = slice_block.append_op_result(ReallocBindingsMeta::realloc(
        let dst_ptr = slice_block.append_op_result(ReallocBindingsMeta::realloc(
//            context, dst_ptr, dst_size, location,
            context, dst_ptr, dst_size, location,
//        ))?;
        ))?;
//

//        // TODO: Find out if we need to clone stuff using the snapshot clone meta.
        // TODO: Find out if we need to clone stuff using the snapshot clone meta.
//        let src_offset = {
        let src_offset = {
//            let slice_since = slice_block.append_op_result(arith::extui(
            let slice_since = slice_block.append_op_result(arith::extui(
//                slice_since,
                slice_since,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))?;
            ))?;
//

//            slice_block.append_op_result(arith::muli(slice_since, elem_size, location))?
            slice_block.append_op_result(arith::muli(slice_since, elem_size, location))?
//        };
        };
//

//        let src_ptr = slice_block.extract_value(
        let src_ptr = slice_block.extract_value(
//            context,
            context,
//            location,
            location,
//            entry.argument(1)?.into(),
            entry.argument(1)?.into(),
//            pointer(context, 0),
            pointer(context, 0),
//            0,
            0,
//        )?;
        )?;
//        let src_ptr = slice_block.append_op_result(llvm::get_element_ptr_dynamic(
        let src_ptr = slice_block.append_op_result(llvm::get_element_ptr_dynamic(
//            context,
            context,
//            src_ptr,
            src_ptr,
//            &[src_offset],
            &[src_offset],
//            IntegerType::new(context, 8).into(),
            IntegerType::new(context, 8).into(),
//            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
//            location,
            location,
//        ))?;
        ))?;
//

//        slice_block.memcpy(context, location, src_ptr, dst_ptr, dst_size);
        slice_block.memcpy(context, location, src_ptr, dst_ptr, dst_size);
//

//        let k0 = slice_block.const_int_from_type(context, location, 0, len_ty)?;
        let k0 = slice_block.const_int_from_type(context, location, 0, len_ty)?;
//

//        let value = slice_block.append_op_result(llvm::undef(array_ty, location))?;
        let value = slice_block.append_op_result(llvm::undef(array_ty, location))?;
//        let value = slice_block.insert_values(
        let value = slice_block.insert_values(
//            context,
            context,
//            location,
            location,
//            value,
            value,
//            &[dst_ptr, k0, slice_length, slice_length],
            &[dst_ptr, k0, slice_length, slice_length],
//        )?;
        )?;
//

//        slice_block.append_operation(helper.br(0, &[range_check, value], location));
        slice_block.append_operation(helper.br(0, &[range_check, value], location));
//    }
    }
//

//    error_block.append_operation(helper.br(1, &[range_check], location));
    error_block.append_operation(helper.br(1, &[range_check], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `span_from_tuple` libfunc.
/// Generate MLIR operations for the `span_from_tuple` libfunc.
//pub fn build_span_from_tuple<'ctx, 'this>(
pub fn build_span_from_tuple<'ctx, 'this>(
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
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    // tuple to array span (t,t,t) -> &[t,t,t]
    // tuple to array span (t,t,t) -> &[t,t,t]
//

//    if metadata.get::<ReallocBindingsMeta>().is_none() {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
//        metadata.insert(ReallocBindingsMeta::new(context, helper));
        metadata.insert(ReallocBindingsMeta::new(context, helper));
//    }
    }
//

//    let struct_type_info = registry.get_type(&info.ty)?;
    let struct_type_info = registry.get_type(&info.ty)?;
//

//    let struct_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
    let struct_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
//

//    let container: Value = {
    let container: Value = {
//        // load box
        // load box
//        entry.load(
        entry.load(
//            context,
            context,
//            location,
            location,
//            entry.argument(0)?.into(),
            entry.argument(0)?.into(),
//            struct_ty,
            struct_ty,
//            Some(struct_type_info.layout(registry)?.align()),
            Some(struct_type_info.layout(registry)?.align()),
//        )?
        )?
//    };
    };
//

//    let fields = struct_type_info.fields().expect("should have fields");
    let fields = struct_type_info.fields().expect("should have fields");
//    let (field_ty, field_layout) =
    let (field_ty, field_layout) =
//        registry.build_type_with_layout(context, helper, registry, metadata, &fields[0])?;
        registry.build_type_with_layout(context, helper, registry, metadata, &fields[0])?;
//    let field_stride = field_layout.pad_to_align().size();
    let field_stride = field_layout.pad_to_align().size();
//

//    let array_ty = registry.build_type(
    let array_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);
//

//    let array_len_value = entry.const_int_from_type(context, location, fields.len(), len_ty)?;
    let array_len_value = entry.const_int_from_type(context, location, fields.len(), len_ty)?;
//

//    let array_container = entry.append_op_result(llvm::undef(array_ty, location))?;
    let array_container = entry.append_op_result(llvm::undef(array_ty, location))?;
//

//    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
//

//    let array_container = entry.insert_value(context, location, array_container, k0, 1)?;
    let array_container = entry.insert_value(context, location, array_container, k0, 1)?;
//    let array_container =
    let array_container =
//        entry.insert_value(context, location, array_container, array_len_value, 2)?;
        entry.insert_value(context, location, array_container, array_len_value, 2)?;
//    let array_container =
    let array_container =
//        entry.insert_value(context, location, array_container, array_len_value, 3)?;
        entry.insert_value(context, location, array_container, array_len_value, 3)?;
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let field_size: Value = entry.const_int(context, location, field_stride, 64)?;
    let field_size: Value = entry.const_int(context, location, field_stride, 64)?;
//    let array_len_value_i64 = entry
    let array_len_value_i64 = entry
//        .append_operation(arith::extui(array_len_value, field_size.r#type(), location))
        .append_operation(arith::extui(array_len_value, field_size.r#type(), location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    let total_size = entry
    let total_size = entry
//        .append_operation(arith::muli(field_size, array_len_value_i64, location))
        .append_operation(arith::muli(field_size, array_len_value_i64, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let ptr = entry
    let ptr = entry
//        .append_operation(ReallocBindingsMeta::realloc(
        .append_operation(ReallocBindingsMeta::realloc(
//            context, ptr, total_size, location,
            context, ptr, total_size, location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    for (i, _) in fields.iter().enumerate() {
    for (i, _) in fields.iter().enumerate() {
//        let value: Value = entry.extract_value(context, location, container, field_ty, i)?;
        let value: Value = entry.extract_value(context, location, container, field_ty, i)?;
//

//        let target_ptr = entry
        let target_ptr = entry
//            .append_operation(llvm::get_element_ptr(
            .append_operation(llvm::get_element_ptr(
//                context,
                context,
//                ptr,
                ptr,
//                DenseI32ArrayAttribute::new(context, &[i as i32]),
                DenseI32ArrayAttribute::new(context, &[i as i32]),
//                field_ty,
                field_ty,
//                pointer(context, 0),
                pointer(context, 0),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        entry.store(context, location, target_ptr, value, None);
        entry.store(context, location, target_ptr, value, None);
//    }
    }
//

//    let array_container = entry.insert_value(context, location, array_container, ptr, 0)?;
    let array_container = entry.insert_value(context, location, array_container, ptr, 0)?;
//

//    entry.append_operation(helper.br(0, &[array_container], location));
    entry.append_operation(helper.br(0, &[array_container], location));
//

//    Ok(())
    Ok(())
//}
}
//

//fn assert_nonnull<'ctx, 'this>(
fn assert_nonnull<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    ptr: Value<'ctx, 'this>,
    ptr: Value<'ctx, 'this>,
//    msg: &str,
    msg: &str,
//) -> Result<()> {
) -> Result<()> {
//    let k0 = entry.const_int(context, location, 0, 64)?;
    let k0 = entry.const_int(context, location, 0, 64)?;
//    let ptr_value = entry.append_op_result(
    let ptr_value = entry.append_op_result(
//        ods::llvm::ptrtoint(context, IntegerType::new(context, 64).into(), ptr, location).into(),
        ods::llvm::ptrtoint(context, IntegerType::new(context, 64).into(), ptr, location).into(),
//    )?;
    )?;
//

//    let ptr_is_not_null = entry
    let ptr_is_not_null = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Ne,
            CmpiPredicate::Ne,
//            ptr_value,
            ptr_value,
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

//    entry.append_operation(cf::assert(context, ptr_is_not_null, msg, location));
    entry.append_operation(cf::assert(context, ptr_is_not_null, msg, location));
//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program},
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use pretty_assertions_sorted::assert_eq;
    use pretty_assertions_sorted::assert_eq;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    #[test]
    #[test]
//    fn run_roundtrip() {
    fn run_roundtrip() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test(x: Array<u32>) -> Array<u32> {
            fn run_test(x: Array<u32>) -> Array<u32> {
//                x
                x
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[[1u32, 2u32].into()]).return_value;
        let result = run_program(&program, "run_test", &[[1u32, 2u32].into()]).return_value;
//

//        assert_eq!(result, JitValue::from([1u32, 2u32]));
        assert_eq!(result, JitValue::from([1u32, 2u32]));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_append() {
    fn run_append() {
//        let program = load_cairo! {
        let program = load_cairo! {
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers
                numbers
//            }
            }
//        };
        };
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, [4u32].into());
        assert_eq!(result, [4u32].into());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_len() {
    fn run_len() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.len()
                numbers.len()
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, 3u32.into());
        assert_eq!(result, 3u32.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_get() {
    fn run_get() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> (u32, u32, u32, u32) {
            fn run_test() -> (u32, u32, u32, u32) {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(1_u32);
                numbers.append(1_u32);
//                (
                (
//                    *numbers.at(0),
                    *numbers.at(0),
//                    *numbers.at(1),
                    *numbers.at(1),
//                    *numbers.at(2),
                    *numbers.at(2),
//                    *numbers.at(3),
                    *numbers.at(3),
//                )
                )
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    4u32.into(),
                    4u32.into(),
//                    3u32.into(),
                    3u32.into(),
//                    2u32.into(),
                    2u32.into(),
//                    1u32.into()
                    1u32.into()
//                ))
                ))
//            )
            )
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_get_big() {
    fn run_get_big() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> (u32, u32, u32, u32) {
            fn run_test() -> (u32, u32, u32, u32) {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(17_u32);
                numbers.append(17_u32);
//                numbers.append(17_u32);
                numbers.append(17_u32);
//                numbers.append(18_u32);
                numbers.append(18_u32);
//                numbers.append(19_u32);
                numbers.append(19_u32);
//                numbers.append(20_u32);
                numbers.append(20_u32);
//                numbers.append(21_u32);
                numbers.append(21_u32);
//                numbers.append(22_u32);
                numbers.append(22_u32);
//                numbers.append(23_u32);
                numbers.append(23_u32);
//                (
                (
//                    *numbers.at(20),
                    *numbers.at(20),
//                    *numbers.at(21),
                    *numbers.at(21),
//                    *numbers.at(22),
                    *numbers.at(22),
//                    *numbers.at(23),
                    *numbers.at(23),
//                )
                )
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(jit_struct!(
                jit_struct!(jit_struct!(
//                    20u32.into(),
                    20u32.into(),
//                    21u32.into(),
                    21u32.into(),
//                    22u32.into(),
                    22u32.into(),
//                    23u32.into()
                    23u32.into()
//                ))
                ))
//            )
            )
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_pop_front() {
    fn run_pop_front() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                let _ = numbers.pop_front();
                let _ = numbers.pop_front();
//                numbers.append(1_u32);
                numbers.append(1_u32);
//                *numbers.at(0)
                *numbers.at(0)
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_pop_front_result() {
    fn run_pop_front_result() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Option<u32> {
            fn run_test() -> Option<u32> {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                numbers.pop_front()
                numbers.pop_front()
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_enum!(0, 4u32.into()));
        assert_eq!(result, jit_enum!(0, 4u32.into()));
//

//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Option<u32> {
            fn run_test() -> Option<u32> {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.pop_front()
                numbers.pop_front()
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_enum!(1, jit_struct!()));
        assert_eq!(result, jit_enum!(1, jit_struct!()));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_pop_front_consume() {
    fn run_pop_front_consume() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                match numbers.pop_front_consume() {
                match numbers.pop_front_consume() {
//                    Option::Some((_, x)) => x,
                    Option::Some((_, x)) => x,
//                    Option::None(()) => 0_u32,
                    Option::None(()) => 0_u32,
//                }
                }
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, 4u32.into());
        assert_eq!(result, 4u32.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn run_pop_back() {
    fn run_pop_back() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
            fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(4_u32);
                numbers.append(4_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                numbers.append(1_u32);
                numbers.append(1_u32);
//                let mut numbers = numbers.span();
                let mut numbers = numbers.span();
//                (
                (
//                    numbers.pop_back(),
                    numbers.pop_back(),
//                    numbers.pop_back(),
                    numbers.pop_back(),
//                    numbers.pop_back(),
                    numbers.pop_back(),
//                    numbers.pop_back(),
                    numbers.pop_back(),
//                )
                )
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            jit_struct!(
            jit_struct!(
//                jit_enum!(0, 1u32.into()),
                jit_enum!(0, 1u32.into()),
//                jit_enum!(0, 3u32.into()),
                jit_enum!(0, 3u32.into()),
//                jit_enum!(0, 4u32.into()),
                jit_enum!(0, 4u32.into()),
//                jit_enum!(
                jit_enum!(
//                    1,
                    1,
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: Vec::new(),
                        fields: Vec::new(),
//                        debug_name: None,
                        debug_name: None,
//                    }
                    }
//                ),
                ),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_slice() {
    fn run_slice() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::Array;
            use array::Array;
//            use array::ArrayTrait;
            use array::ArrayTrait;
//            use array::SpanTrait;
            use array::SpanTrait;
//            use option::OptionTrait;
            use option::OptionTrait;
//            use box::BoxTrait;
            use box::BoxTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data: Array<u32> = ArrayTrait::new();
                let mut data: Array<u32> = ArrayTrait::new();
//                data.append(1_u32);
                data.append(1_u32);
//                data.append(2_u32);
                data.append(2_u32);
//                data.append(3_u32);
                data.append(3_u32);
//                data.append(4_u32);
                data.append(4_u32);
//                let sp = data.span();
                let sp = data.span();
//                let slice = sp.slice(1, 2);
                let slice = sp.slice(1, 2);
//                data.append(5_u32);
                data.append(5_u32);
//                data.append(5_u32);
                data.append(5_u32);
//                data.append(5_u32);
                data.append(5_u32);
//                data.append(5_u32);
                data.append(5_u32);
//                data.append(5_u32);
                data.append(5_u32);
//                data.append(5_u32);
                data.append(5_u32);
//                *slice.get(1).unwrap().unbox()
                *slice.get(1).unwrap().unbox()
//            }
            }
//

//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
//    }
    }
//

//    #[test]
    #[test]
//    fn run_slice_fail() {
    fn run_slice_fail() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::Array;
            use array::Array;
//            use array::ArrayTrait;
            use array::ArrayTrait;
//            use array::SpanTrait;
            use array::SpanTrait;
//            use option::OptionTrait;
            use option::OptionTrait;
//            use box::BoxTrait;
            use box::BoxTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data: Array<u32> = ArrayTrait::new();
                let mut data: Array<u32> = ArrayTrait::new();
//                data.append(1_u32);
                data.append(1_u32);
//                data.append(2_u32);
                data.append(2_u32);
//                data.append(3_u32);
                data.append(3_u32);
//                data.append(4_u32);
                data.append(4_u32);
//                let sp = data.span();
                let sp = data.span();
//                let slice = sp.slice(1, 4); // oob
                let slice = sp.slice(1, 4); // oob
//                //data.append(5_u32);
                //data.append(5_u32);
//                *slice.get(0).unwrap().unbox()
                *slice.get(0).unwrap().unbox()
//            }
            }
//

//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            jit_panic!(JitValue::felt_str(
            jit_panic!(JitValue::felt_str(
//                "1637570914057682275393755530660268060279989363"
                "1637570914057682275393755530660268060279989363"
//            ))
            ))
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_span_from_tuple() {
    fn run_span_from_tuple() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            mod felt252_span_from_tuple {
            mod felt252_span_from_tuple {
//                pub extern fn span_from_tuple<T>(struct_like: Box<@T>) -> @Array<felt252> nopanic;
                pub extern fn span_from_tuple<T>(struct_like: Box<@T>) -> @Array<felt252> nopanic;
//            }
            }
//

//            fn run_test() -> Array<felt252> {
            fn run_test() -> Array<felt252> {
//                let span = felt252_span_from_tuple::span_from_tuple(BoxTrait::new(@(10, 20, 30)));
                let span = felt252_span_from_tuple::span_from_tuple(BoxTrait::new(@(10, 20, 30)));
//                span.clone()
                span.clone()
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            jit_enum!(
            jit_enum!(
//                0,
                0,
//                jit_struct!(JitValue::from([
                jit_struct!(JitValue::from([
//                    JitValue::Felt252(Felt::from(10)),
                    JitValue::Felt252(Felt::from(10)),
//                    Felt::from(20).into(),
                    Felt::from(20).into(),
//                    Felt::from(30).into()
                    Felt::from(30).into()
//                ]))
                ]))
//            )
            )
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn run_span_from_multi_tuple() {
    fn run_span_from_multi_tuple() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            mod tuple_span_from_tuple {
            mod tuple_span_from_tuple {
//                pub extern fn span_from_tuple<T>(
                pub extern fn span_from_tuple<T>(
//                    struct_like: Box<@T>
                    struct_like: Box<@T>
//                ) -> @Array<(felt252, felt252, felt252)> nopanic;
                ) -> @Array<(felt252, felt252, felt252)> nopanic;
//            }
            }
//

//            fn run_test() {
            fn run_test() {
//                let multi_tuple = ((10, 20, 30), (40, 50, 60), (70, 80, 90));
                let multi_tuple = ((10, 20, 30), (40, 50, 60), (70, 80, 90));
//                let span = tuple_span_from_tuple::span_from_tuple(BoxTrait::new(@multi_tuple));
                let span = tuple_span_from_tuple::span_from_tuple(BoxTrait::new(@multi_tuple));
//                assert!(*span[0] == (10, 20, 30));
                assert!(*span[0] == (10, 20, 30));
//                assert!(*span[1] == (40, 50, 60));
                assert!(*span[1] == (40, 50, 60));
//                assert!(*span[2] == (70, 80, 90));
                assert!(*span[2] == (70, 80, 90));
//            }
            }
//        );
        );
//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_enum!(0, jit_struct!(jit_struct!())));
        assert_eq!(result, jit_enum!(0, jit_struct!(jit_struct!())));
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append1() {
    fn seq_append1() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::from([1u32]),
            JitValue::from([1u32]),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2() {
    fn seq_append2() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::from([1u32, 2u32]),
            JitValue::from([1u32, 2u32]),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2_popf1() {
    fn seq_append2_popf1() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                let _ = data.pop_front();
                let _ = data.pop_front();
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::from([2u32]),
            JitValue::from([2u32]),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2_popb1() {
    fn seq_append2_popb1() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Span<u32> {
            fn run_test() -> Span<u32> {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                let mut data = data.span();
                let mut data = data.span();
//                let _ = data.pop_back();
                let _ = data.pop_back();
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            jit_struct!([1u32].into())
            jit_struct!([1u32].into())
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append1_popf1_append1() {
    fn seq_append1_popf1_append1() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                let _ = data.pop_front();
                let _ = data.pop_front();
//                data.append(2);
                data.append(2);
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::from([2u32]),
            JitValue::from([2u32]),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append1_first() {
    fn seq_append1_first() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                *data.at(0)
                *data.at(0)
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::from(1u32)],
                    fields: vec![JitValue::from(1u32)],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2_first() {
    fn seq_append2_first() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                *data.at(0)
                *data.at(0)
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::from(1u32)],
                    fields: vec![JitValue::from(1u32)],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2_popf1_first() {
    fn seq_append2_popf1_first() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                let _ = data.pop_front();
                let _ = data.pop_front();
//                *data.at(0)
                *data.at(0)
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::from(2u32)],
                    fields: vec![JitValue::from(2u32)],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append2_popb1_last() {
    fn seq_append2_popb1_last() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                data.append(2);
                data.append(2);
//                let mut data_span = data.span();
                let mut data_span = data.span();
//                let _ = data_span.pop_back();
                let _ = data_span.pop_back();
//                let last = data_span.len() - 1;
                let last = data_span.len() - 1;
//                *data_span.at(last)
                *data_span.at(last)
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::from(1u32)],
                    fields: vec![JitValue::from(1u32)],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            }
            }
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn seq_append1_popf1_append1_first() {
    fn seq_append1_popf1_append1_first() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> u32 {
            fn run_test() -> u32 {
//                let mut data = ArrayTrait::new();
                let mut data = ArrayTrait::new();
//                data.append(1);
                data.append(1);
//                let _ = data.pop_front();
                let _ = data.pop_front();
//                data.append(2);
                data.append(2);
//                *data.at(0)
                *data.at(0)
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::from(2u32)],
                    fields: vec![JitValue::from(2u32)],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn array_clone() {
    fn array_clone() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> Array<u32> {
            fn run_test() -> Array<u32> {
//                let x = ArrayTrait::new();
                let x = ArrayTrait::new();
//                x.clone()
                x.clone()
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Enum {
            JitValue::Enum {
//                tag: 0,
                tag: 0,
//                value: Box::new(JitValue::Struct {
                value: Box::new(JitValue::Struct {
//                    fields: vec![JitValue::Array(vec![])],
                    fields: vec![JitValue::Array(vec![])],
//                    debug_name: None,
                    debug_name: None,
//                }),
                }),
//                debug_name: None,
                debug_name: None,
//            },
            },
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn array_pop_back_state() {
    fn array_pop_back_state() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use array::ArrayTrait;
            use array::ArrayTrait;
//

//            fn run_test() -> Span<u32> {
            fn run_test() -> Span<u32> {
//                let mut numbers = ArrayTrait::new();
                let mut numbers = ArrayTrait::new();
//                numbers.append(1_u32);
                numbers.append(1_u32);
//                numbers.append(2_u32);
                numbers.append(2_u32);
//                numbers.append(3_u32);
                numbers.append(3_u32);
//                let mut numbers = numbers.span();
                let mut numbers = numbers.span();
//                let _ = numbers.pop_back();
                let _ = numbers.pop_back();
//                numbers
                numbers
//            }
            }
//        );
        );
//

//        let result = run_program(&program, "run_test", &[]).return_value;
        let result = run_program(&program, "run_test", &[]).return_value;
//

//        assert_eq!(result, jit_struct!([1u32, 2u32].into()));
        assert_eq!(result, jit_struct!([1u32, 2u32].into()));
//    }
    }
//

//    #[test]
    #[test]
//    fn array_empty_span() {
    fn array_empty_span() {
//        // Tests snapshot_take on a empty array.
        // Tests snapshot_take on a empty array.
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> Span<u32> {
            fn run_test() -> Span<u32> {
//                let x = ArrayTrait::new();
                let x = ArrayTrait::new();
//                x.span()
                x.span()
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            jit_struct!(JitValue::Array(vec![])),
            jit_struct!(JitValue::Array(vec![])),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn array_span_modify_span() {
    fn array_span_modify_span() {
//        // Tests pop_back on a span.
        // Tests pop_back on a span.
//        let program = load_cairo!(
        let program = load_cairo!(
//            use core::array::SpanTrait;
            use core::array::SpanTrait;
//            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
//                let x = self.pop_back();
                let x = self.pop_back();
//                x
                x
//            }
            }
//

//            fn run_test() -> Option<@u64> {
            fn run_test() -> Option<@u64> {
//                let mut data = array![2].span();
                let mut data = array![2].span();
//                let x = pop_elem(data);
                let x = pop_elem(data);
//                x
                x
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            jit_enum!(0, 2u64.into()),
            jit_enum!(0, 2u64.into()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn array_span_check_array() {
    fn array_span_check_array() {
//        // Tests pop back on a span not modifying the original array.
        // Tests pop back on a span not modifying the original array.
//        let program = load_cairo!(
        let program = load_cairo!(
//            use core::array::SpanTrait;
            use core::array::SpanTrait;
//            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
//                let x = self.pop_back();
                let x = self.pop_back();
//                x
                x
//            }
            }
//

//            fn run_test() -> Array<u64> {
            fn run_test() -> Array<u64> {
//                let mut data = array![1, 2];
                let mut data = array![1, 2];
//                let _x = pop_elem(data.span());
                let _x = pop_elem(data.span());
//                data
                data
//            }
            }
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            run_program(&program, "run_test", &[]).return_value,
            run_program(&program, "run_test", &[]).return_value,
//            JitValue::Array(vec![1u64.into(), 2u64.into()]),
            JitValue::Array(vec![1u64.into(), 2u64.into()]),
//        );
        );
//    }
    }
//}
}
