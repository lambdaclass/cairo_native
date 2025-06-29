//! # Array libfuncs

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Error, Result, SierraAssertError},
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    native_assert,
    types::array::calc_data_prefix_offset,
    utils::{BlockExt, GepIndex, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        array::{ArrayConcreteLibfunc, ConcreteMultiPopLibfunc},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm, ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Block, BlockLike, Location, Region, Value,
    },
    Context,
};
use std::alloc::Layout;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &ArrayConcreteLibfunc,
) -> Result<()> {
    match selector {
        ArrayConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SpanFromTuple(info) => {
            build_span_from_tuple(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::TupleFromSpan(info) => {
            build_tuple_from_span(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Append(info) => {
            build_append(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::PopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::PopFrontConsume(info) => build_pop::<true, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Slice(info) => {
            build_slice(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Len(info) => {
            build_len(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SnapshotPopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::SnapshotPopBack(info) => build_pop::<false, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::SnapshotMultiPopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Multi(info),
        ),
        ArrayConcreteLibfunc::SnapshotMultiPopBack(info) => build_pop::<false, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Multi(info),
        ),
    }
}

/// Builds a new array with no initial capacity.
///
/// # Cairo Signature
///
/// ```cairo
/// extern fn array_new<T>() -> Array<T> nopanic;
/// ```
pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;

    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let value = entry.insert_values(context, location, value, &[nullptr, k0, k0, k0])?;

    helper.br(entry, 0, &[value], location)
}

/// Buils a span (a cairo native array) from a boxed tuple of same-type elements.
///
/// Note: The `&info.ty` field has the entire `[T; N]` tuple. It is not the `T` in `Array<T>`.
///
/// # Cairo Signature
///
/// ```cairo
/// extern fn span_from_tuple<T, impl Info: FixedSizedArrayInfo<T>>(
///     struct_like: Box<@T>
/// ) -> @Array<Info::Element> nopanic;
/// ```
pub fn build_span_from_tuple<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let tuple_len = {
        let CoreTypeConcrete::Struct(info) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        info.members.len()
    };

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();
    let (_, tuple_layout) = registry.build_type_with_layout(context, helper, metadata, &info.ty)?;

    let array_len_bytes = tuple_layout.pad_to_align().size();
    let array_len_bytes_with_offset = entry.const_int(
        context,
        location,
        array_len_bytes + calc_data_prefix_offset(tuple_layout),
        64,
    )?;
    let array_len_bytes = entry.const_int(context, location, array_len_bytes, 64)?;
    let array_len = entry.const_int_from_type(context, location, tuple_len, len_ty)?;

    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
    let k1 = entry.const_int_from_type(context, location, 1, len_ty)?;

    // Allocate space for the array.
    let allocation_ptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let allocation_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context,
        allocation_ptr,
        array_len_bytes_with_offset,
        location,
    )?)?;

    // Write the array data prefix.
    let data_prefix = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[len_ty, len_ty], false),
        location,
    ))?;
    let data_prefix = entry.insert_values(context, location, data_prefix, &[k1, array_len])?;
    entry.store(context, location, allocation_ptr, data_prefix)?;
    let array_ptr = entry.gep(
        context,
        location,
        allocation_ptr,
        &[GepIndex::Const(calc_data_prefix_offset(tuple_layout) as i32)],
        IntegerType::new(context, 8).into(),
    )?;

    // Move the data into the array and free the original tuple. Since the tuple and the array are
    // represented the same way, a simple memcpy is enough.
    entry.memcpy(
        context,
        location,
        entry.argument(0)?.into(),
        array_ptr,
        array_len_bytes,
    );
    entry.append_operation(ReallocBindingsMeta::free(
        context,
        entry.argument(0)?.into(),
        location,
    )?);

    // Build the array representation.
    let k8 = entry.const_int(context, location, 8, 64)?;
    let array_ptr_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let array_ptr_ptr: Value<'ctx, '_> = entry.append_op_result(ReallocBindingsMeta::realloc(
        context,
        array_ptr_ptr,
        k8,
        location,
    )?)?;
    entry.store(context, location, array_ptr_ptr, array_ptr)?;

    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let value = entry.insert_values(
        context,
        location,
        value,
        &[array_ptr_ptr, k0, array_len, array_len],
    )?;

    helper.br(entry, 0, &[value], location)
}

/// Buils a tuple (struct) from an span (a cairo native array)
///
/// Note: The `&info.ty` field has the entire `[T; N]` tuple. It is not the `T` in `Array<T>`.
/// The tuple size `N` must match the span length.
///
/// # Cairo Signature
///
/// ```cairo
/// fn tuple_from_span<T, impl Info: FixedSizedArrayInfo<T>>(span: @Array<Info::Element>  -> Option<@Box<T>> nopanic;
/// ```
pub fn build_tuple_from_span<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let elem_id = {
        let CoreTypeConcrete::Snapshot(info) =
            registry.get_type(&info.signature.param_signatures[0].ty)?
        else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };
        let CoreTypeConcrete::Array(info) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        &info.ty
    };
    let tuple_len_const = {
        let CoreTypeConcrete::Struct(param) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        param.members.len()
    };

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();
    let (_, elem_layout) = registry.build_type_with_layout(context, helper, metadata, elem_id)?;
    let (tuple_ty, tuple_layout) =
        registry.build_type_with_layout(context, helper, metadata, &info.ty)?;

    let array_ptr_ptr =
        entry.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
    let array_start =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let tuple_len = entry.const_int_from_type(context, location, tuple_len_const, len_ty)?;
    let len_matches = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_len,
        tuple_len,
        location,
    ))?;

    // Ensure the tuple's length matches the array's.
    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        len_matches,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    // Ensure the type's clone and drop implementations are registered.
    registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    // Branch for when the lengths match:
    {
        let value_size = valid_block.const_int(context, location, tuple_layout.size(), 64)?;

        let value = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let value = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context, value, value_size, location,
        )?)?;

        let array_start_offset = valid_block.append_op_result(arith::extui(
            array_start,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let array_start_offset = valid_block.append_op_result(arith::muli(
            array_start_offset,
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?,
            location,
        ))?;

        let array_ptr = valid_block.load(context, location, array_ptr_ptr, ptr_ty)?;
        let array_data_start_ptr = valid_block.gep(
            context,
            location,
            array_ptr,
            &[GepIndex::Value(array_start_offset)],
            IntegerType::new(context, 8).into(),
        )?;

        // Check if the array is shared.
        let is_shared = is_shared(context, valid_block, location, array_ptr_ptr, elem_layout)?;
        valid_block.append_operation(scf::r#if(
            is_shared,
            &[],
            {
                // When the array is shared, we should clone the entire data.
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                match metadata.get::<DupOverridesMeta>() {
                    Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(&info.ty) => {
                        let src_ptr = array_data_start_ptr;
                        let dst_ptr = value;

                        let value = block.load(context, location, src_ptr, tuple_ty)?;

                        // Invoke the tuple's clone mechanism, which will take care of copying or
                        // cloning each item in the array.
                        let values = dup_overrides_meta
                            .invoke_override(context, &block, location, &info.ty, value)?;
                        block.store(context, location, src_ptr, values.0)?;
                        block.store(context, location, dst_ptr, values.1)?;
                    }
                    _ => block.memcpy(context, location, array_data_start_ptr, value, value_size),
                }

                // Drop the original array (by decreasing its reference counter).
                metadata
                    .get::<DropOverridesMeta>()
                    .to_native_assert_error("array always has a drop implementation")?
                    .invoke_override(
                        context,
                        &block,
                        location,
                        &info.signature.param_signatures[0].ty,
                        entry.argument(0)?.into(),
                    )?;

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            {
                // When the array is not shared, we can just move the data to the new tuple and free
                // the array.

                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.memcpy(context, location, array_data_start_ptr, value, value_size);

                // NOTE: If the target tuple has no elements, and the array is not shared, then we
                // would attempt to free 0xfffffffffffffff0. This is not possible and disallowed by
                // the Cairo compiler.

                // TODO: Drop elements before array_start and between array_end and max length.
                let data_ptr = block.gep(
                    context,
                    location,
                    array_ptr,
                    &[GepIndex::Const(
                        -(calc_data_prefix_offset(elem_layout) as i32),
                    )],
                    IntegerType::new(context, 8).into(),
                )?;
                block.append_operation(ReallocBindingsMeta::free(context, data_ptr, location)?);
                block.append_operation(ReallocBindingsMeta::free(
                    context,
                    array_ptr_ptr,
                    location,
                )?);

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            location,
        ));

        helper.br(valid_block, 0, &[value], location)?;
    }

    {
        // When there's a length mismatch, just consume (drop) the array.
        metadata
            .get::<DropOverridesMeta>()
            .ok_or(Error::MissingMetadata)?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[0].ty,
                entry.argument(0)?.into(),
            )?;

        helper.br(error_block, 1, &[], location)
    }
}

/// Generate MLIR operations for the `array_append` libfunc.
pub fn build_append<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let self_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let (_, elem_layout) = registry.build_type_with_layout(context, helper, metadata, &info.ty)?;
    let elem_stride = entry.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

    fn compute_next_capacity<'ctx, 'this>(
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        elem_stride: Value<'ctx, 'this>,
        array_capacity: Value<'ctx, 'this>,
    ) -> Result<(Value<'ctx, 'this>, Value<'ctx, 'this>)> {
        let len_ty = IntegerType::new(context, 32).into();

        let k1 = block.const_int_from_type(context, location, 1, len_ty)?;
        let k8 = block.const_int_from_type(context, location, 8, len_ty)?;
        let k1024 = block.const_int_from_type(context, location, 1024, len_ty)?;

        let realloc_len = block.append_op_result(arith::shli(array_capacity, k1, location))?;
        let realloc_len = block.append_op_result(arith::minui(realloc_len, k1024, location))?;
        let realloc_len =
            block.append_op_result(arith::addi(realloc_len, array_capacity, location))?;
        let realloc_len = block.append_op_result(arith::maxui(realloc_len, k8, location))?;

        let realloc_size = block.append_op_result(arith::extui(
            realloc_len,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let realloc_size =
            block.append_op_result(arith::muli(realloc_size, elem_stride, location))?;

        Result::Ok((realloc_len, realloc_size))
    }

    let data_prefix_size = calc_data_prefix_offset(elem_layout);

    let array_capacity =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 3)?;
    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
    let is_empty = entry.cmpi(context, CmpiPredicate::Eq, array_capacity, k0, location)?;
    let array_obj = entry.append_op_result(scf::r#if(
        is_empty,
        &[self_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let (array_capacity, realloc_len) =
                compute_next_capacity(context, &block, location, elem_stride, array_capacity)?;

            let data_prefix_size_value =
                block.const_int(context, location, data_prefix_size, 64)?;
            let realloc_len = block.addi(realloc_len, data_prefix_size_value, location)?;

            let null_ptr = block.append_op_result(llvm::zero(ptr_ty, location))?;
            let array_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                context,
                null_ptr,
                realloc_len,
                location,
            )?)?;

            let k1 = block.const_int_from_type(context, location, 1, len_ty)?;
            block.store(context, location, array_ptr, k1)?;
            let max_len_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(size_of::<u32>() as i32)],
                IntegerType::new(context, 8).into(),
            )?;
            block.store(context, location, max_len_ptr, k0)?;

            let array_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(data_prefix_size as i32)],
                IntegerType::new(context, 8).into(),
            )?;

            let k8 = block.const_int(context, location, 8, 64)?;
            let array_ptr_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                context, null_ptr, k8, location,
            )?)?;
            block.store(context, location, array_ptr_ptr, array_ptr)?;

            let array_obj = entry.argument(0)?.into();
            let array_obj = block.insert_value(context, location, array_obj, array_ptr_ptr, 0)?;
            let array_obj = block.insert_value(context, location, array_obj, array_capacity, 3)?;
            block.append_operation(scf::r#yield(&[array_obj], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_end =
                block.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;
            let has_space = block.cmpi(
                context,
                CmpiPredicate::Ult,
                array_end,
                array_capacity,
                location,
            )?;
            let array_obj = block.append_op_result(scf::r#if(
                has_space,
                &[self_ty],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    block.append_operation(scf::r#yield(&[entry.argument(0)?.into()], location));
                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let (array_capacity, realloc_len) = compute_next_capacity(
                        context,
                        &block,
                        location,
                        elem_stride,
                        array_capacity,
                    )?;

                    let data_prefix_size_value =
                        block.const_int(context, location, data_prefix_size, 64)?;
                    let realloc_len = block.addi(realloc_len, data_prefix_size_value, location)?;

                    let array_ptr_ptr = block.extract_value(
                        context,
                        location,
                        entry.argument(0)?.into(),
                        ptr_ty,
                        0,
                    )?;
                    let array_ptr = block.load(context, location, array_ptr_ptr, ptr_ty)?;
                    let array_ptr = block.gep(
                        context,
                        location,
                        array_ptr,
                        &[GepIndex::Const(-(data_prefix_size as i32))],
                        IntegerType::new(context, 8).into(),
                    )?;

                    let array_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                        context,
                        array_ptr,
                        realloc_len,
                        location,
                    )?)?;
                    let array_ptr = block.gep(
                        context,
                        location,
                        array_ptr,
                        &[GepIndex::Const(data_prefix_size as i32)],
                        IntegerType::new(context, 8).into(),
                    )?;

                    block.store(context, location, array_ptr_ptr, array_ptr)?;

                    let array_obj = block.insert_value(
                        context,
                        location,
                        entry.argument(0)?.into(),
                        array_capacity,
                        3,
                    )?;
                    block.append_operation(scf::r#yield(&[array_obj], location));
                    region
                },
                location,
            ))?;

            block.append_operation(scf::r#yield(&[array_obj], location));
            region
        },
        location,
    ))?;

    let array_ptr_ptr = entry.extract_value(context, location, array_obj, ptr_ty, 0)?;
    let array_ptr = entry.load(context, location, array_ptr_ptr, ptr_ty)?;

    // Insert the value.
    let target_offset = entry.extract_value(context, location, array_obj, len_ty, 2)?;
    let target_offset = entry.extui(
        target_offset,
        IntegerType::new(context, 64).into(),
        location,
    )?;
    let target_offset = entry.muli(target_offset, elem_stride, location)?;
    let target_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Value(target_offset)],
        IntegerType::new(context, 8).into(),
    )?;

    entry.store(context, location, target_ptr, entry.argument(1)?.into())?;

    // Update array.
    let k1 = entry.const_int_from_type(context, location, 1, len_ty)?;
    let array_end = entry.extract_value(context, location, array_obj, len_ty, 2)?;
    let array_end = entry.addi(array_end, k1, location)?;
    let array_obj = entry.insert_value(context, location, array_obj, array_end, 2)?;

    // Update max length.
    let max_len_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Const(
            -((crate::types::array::calc_data_prefix_offset(elem_layout) - size_of::<u32>())
                as i32),
        )],
        IntegerType::new(context, 8).into(),
    )?;
    entry.store(context, location, max_len_ptr, array_end)?;

    helper.br(entry, 0, &[array_obj], location)
}

#[derive(Clone, Copy)]
enum PopInfo<'a> {
    Single(&'a SignatureAndTypeConcreteLibfunc),
    Multi(&'a ConcreteMultiPopLibfunc),
}

/// Generate MLIR operations for the `array_pop_*` libfuncs.
///
/// Template arguments:
///   - Consume: Whether to consume or not the array on failure.
///   - Reverse: False for front-popping, true for back-popping.
///
/// The `info` argument contains how many items to pop.
fn build_pop<'ctx, 'this, const CONSUME: bool, const REVERSE: bool>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: PopInfo,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let (self_ty, elem_ty, array_obj, extract_len, branch_values) = match info {
        PopInfo::Single(info) => (
            &info.signature.param_signatures[0].ty,
            &info.ty,
            entry.argument(0)?.into(),
            1,
            Vec::new(),
        ),
        PopInfo::Multi(ConcreteMultiPopLibfunc {
            popped_ty,
            signature,
        }) => {
            let range_check = super::increment_builtin_counter(
                context,
                entry,
                location,
                entry.argument(0)?.into(),
            )?;

            let CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType { ty, .. }) =
                registry.get_type(&signature.param_signatures[1].ty)?
            else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };

            let CoreTypeConcrete::Array(InfoAndTypeConcreteType { ty, .. }) =
                registry.get_type(ty)?
            else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };

            let CoreTypeConcrete::Struct(info) = registry.get_type(popped_ty)? else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };
            native_assert!(
                info.members.iter().all(|member_ty| member_ty == ty),
                "output struct type should match the array's type"
            );

            (
                &signature.param_signatures[1].ty,
                ty,
                entry.argument(1)?.into(),
                info.members.len(),
                vec![range_check],
            )
        }
    };
    let (elem_type, elem_layout) =
        registry.build_type_with_layout(context, helper, metadata, elem_ty)?;

    let array_start = entry.extract_value(context, location, array_obj, len_ty, 1)?;
    let array_end = entry.extract_value(context, location, array_obj, len_ty, 2)?;

    let extract_len_value = entry.const_int_from_type(context, location, extract_len, len_ty)?;
    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let has_enough_data = entry.cmpi(
        context,
        CmpiPredicate::Ule,
        extract_len_value,
        array_len,
        location,
    )?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        has_enough_data,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        let mut branch_values = branch_values.clone();

        let array_ptr_ptr = valid_block.extract_value(context, location, array_obj, ptr_ty, 0)?;
        let array_ptr = valid_block.load(context, location, array_ptr_ptr, ptr_ty)?;

        let elem_stride =
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

        // Extract pointer and update bounds.
        let (array_obj, source_ptr) = if REVERSE {
            let array_end = valid_block.append_op_result(arith::subi(
                array_end,
                extract_len_value,
                location,
            ))?;
            let array_obj = valid_block.insert_value(context, location, array_obj, array_end, 2)?;

            // Compute data offset (elem_stride * array_end) and GEP.
            let data_offset =
                valid_block.extui(array_end, IntegerType::new(context, 64).into(), location)?;
            let data_offset = valid_block.muli(elem_stride, data_offset, location)?;
            let data_ptr = valid_block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Value(data_offset)],
                IntegerType::new(context, 8).into(),
            )?;

            (array_obj, data_ptr)
        } else {
            // Compute data offset (elem_stride * array_end) and GEP.
            let data_offset =
                valid_block.extui(array_start, IntegerType::new(context, 64).into(), location)?;
            let data_offset = valid_block.muli(elem_stride, data_offset, location)?;
            let data_ptr = valid_block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Value(data_offset)],
                IntegerType::new(context, 8).into(),
            )?;

            let array_start = valid_block.append_op_result(arith::addi(
                array_start,
                extract_len_value,
                location,
            ))?;
            let array_obj =
                valid_block.insert_value(context, location, array_obj, array_start, 1)?;

            (array_obj, data_ptr)
        };

        // Allocate output pointer.
        let target_size = valid_block.const_int(
            context,
            location,
            elem_layout.pad_to_align().size() * extract_len,
            64,
        )?;
        let target_ptr = valid_block
            .append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
        let target_ptr = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context,
            target_ptr,
            target_size,
            location,
        )?)?;

        // Clone popped items.
        match metadata.get::<DupOverridesMeta>() {
            Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(elem_ty) => {
                for i in 0..extract_len {
                    let source_ptr = valid_block.gep(
                        context,
                        location,
                        source_ptr,
                        &[GepIndex::Const(
                            (elem_layout.pad_to_align().size() * i) as i32,
                        )],
                        IntegerType::new(context, 8).into(),
                    )?;
                    let target_ptr = valid_block.gep(
                        context,
                        location,
                        target_ptr,
                        &[GepIndex::Const(
                            (elem_layout.pad_to_align().size() * i) as i32,
                        )],
                        IntegerType::new(context, 8).into(),
                    )?;

                    let value = valid_block.load(context, location, source_ptr, elem_type)?;
                    let values = dup_overrides_meta.invoke_override(
                        context,
                        valid_block,
                        location,
                        elem_ty,
                        value,
                    )?;
                    valid_block.store(context, location, source_ptr, values.0)?;
                    valid_block.store(context, location, target_ptr, values.1)?;
                }
            }
            _ => valid_block.memcpy(context, location, source_ptr, target_ptr, target_size),
        }

        branch_values.push(array_obj);
        branch_values.push(target_ptr);
        helper.br(valid_block, 0, &branch_values, location)?;
    }

    {
        let mut branch_values = branch_values.clone();

        if CONSUME {
            metadata
                .get::<DropOverridesMeta>()
                .to_native_assert_error("drop overrides meta should exist")?
                .invoke_override(context, error_block, location, self_ty, array_obj)?;
        } else {
            branch_values.push(array_obj);
        }

        helper.br(error_block, 1, &branch_values, location)?;
    }

    Ok(())
}

/// Generate MLIR operations for the `array_get` libfunc.
pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    // Build the type so that the drop impl invocation works properly.
    registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let array_start =
        entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let is_valid = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ult,
        entry.argument(2)?.into(),
        array_len,
        location,
    ))?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_valid,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        let (elem_ty, elem_layout) =
            registry.build_type_with_layout(context, helper, metadata, &info.ty)?;
        let elem_stride =
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

        // Compute data pointer.
        let source_offset = valid_block.addi(array_start, entry.argument(2)?.into(), location)?;
        let source_offset = valid_block.extui(
            source_offset,
            IntegerType::new(context, 64).into(),
            location,
        )?;
        let source_offset = valid_block.muli(source_offset, elem_stride, location)?;
        let source_ptr =
            valid_block.extract_value(context, location, entry.argument(1)?.into(), ptr_ty, 0)?;
        let source_ptr = valid_block.load(context, location, source_ptr, ptr_ty)?;
        let source_ptr = valid_block.gep(
            context,
            location,
            source_ptr,
            &[GepIndex::Value(source_offset)],
            IntegerType::new(context, 8).into(),
        )?;

        // Allocate output pointer.
        let target_ptr = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let target_ptr = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context,
            target_ptr,
            elem_stride,
            location,
        )?)?;

        // Clone the output data.
        match metadata.get::<DupOverridesMeta>() {
            Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(&info.ty) => {
                let value = valid_block.load(context, location, source_ptr, elem_ty)?;
                let values = dup_overrides_meta.invoke_override(
                    context,
                    valid_block,
                    location,
                    &info.ty,
                    value,
                )?;
                valid_block.store(context, location, source_ptr, values.0)?;
                valid_block.store(context, location, target_ptr, values.1)?;
            }
            _ => valid_block.memcpy(context, location, source_ptr, target_ptr, elem_stride),
        }

        // Drop the input array.
        metadata
            .get::<DropOverridesMeta>()
            .to_native_assert_error("drop overrides metadata should be available")?
            .invoke_override(
                context,
                valid_block,
                location,
                &info.signature.param_signatures[1].ty,
                entry.argument(1)?.into(),
            )?;

        helper.br(valid_block, 0, &[range_check, target_ptr], location)?;
    }

    {
        metadata
            .get::<DropOverridesMeta>()
            .to_native_assert_error("drop overrides meta should exist")?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[1].ty,
                entry.argument(1)?.into(),
            )?;

        helper.br(error_block, 1, &[range_check], location)?;
    }

    Ok(())
}

/// Generate MLIR operations for the `array_slice` libfunc.
pub fn build_slice<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let len_ty = IntegerType::new(context, 32).into();

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let array_obj = entry.argument(1)?.into();
    let array_start = entry.extract_value(context, location, array_obj, len_ty, 1)?;
    let array_end = entry.extract_value(context, location, array_obj, len_ty, 2)?;
    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;

    let slice_start = entry.argument(2)?.into();
    let slice_len = entry.argument(3)?.into();
    let slice_end = entry.append_op_result(arith::addi(slice_start, slice_len, location))?;

    let slice_lhs_bound = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        slice_start,
        array_len,
        location,
    ))?;
    let slice_rhs_bound = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        slice_end,
        array_len,
        location,
    ))?;
    let slice_bounds =
        entry.append_op_result(arith::andi(slice_lhs_bound, slice_rhs_bound, location))?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        slice_bounds,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        let array_start = valid_block.addi(array_start, slice_start, location)?;
        let array_end = valid_block.addi(array_start, slice_len, location)?;

        let array_obj = valid_block.insert_value(context, location, array_obj, array_start, 1)?;
        let array_obj = valid_block.insert_value(context, location, array_obj, array_end, 2)?;

        helper.br(valid_block, 0, &[range_check, array_obj], location)?;
    }

    {
        metadata
            .get::<DropOverridesMeta>()
            .ok_or(Error::MissingMetadata)?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[1].ty,
                array_obj,
            )?;

        helper.br(error_block, 1, &[range_check], location)?;
    }

    Ok(())
}

/// Generate MLIR operations for the `array_len` libfunc.
pub fn build_len<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let len_ty = IntegerType::new(context, 32).into();

    let array_start =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;

    metadata
        .get::<DropOverridesMeta>()
        .to_native_assert_error("drop overrides meta should exist")?
        .invoke_override(
            context,
            entry,
            location,
            &info.signature.param_signatures[0].ty,
            entry.argument(0)?.into(),
        )?;

    helper.br(entry, 0, &[array_len], location)
}

fn is_shared<'ctx, 'this>(
    context: &'ctx Context,
    block: &'this Block<'ctx>,
    location: Location<'ctx>,
    array_ptr_ptr: Value<'ctx, 'this>,
    elem_layout: Layout,
) -> Result<Value<'ctx, 'this>> {
    let null_ptr =
        block.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let ptr_is_null = block.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            array_ptr_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let is_shared = block.append_op_result(scf::r#if(
        ptr_is_null,
        &[IntegerType::new(context, 1).into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let k0 = block.const_int(context, location, 0, 1)?;

            block.append_operation(scf::r#yield(&[k0], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_ptr = block.load(
                context,
                location,
                array_ptr_ptr,
                llvm::r#type::pointer(context, 0),
            )?;
            let array_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(
                    -(calc_data_prefix_offset(elem_layout) as i32),
                )],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                array_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            #[cfg(debug_assertions)]
            {
                let k0 = block.const_int(context, location, 0, 32)?;
                let is_nonzero = block.append_op_result(arith::cmpi(
                    context,
                    CmpiPredicate::Ne,
                    ref_count,
                    k0,
                    location,
                ))?;

                block.append_operation(cf::assert(
                    context,
                    is_nonzero,
                    "ref_count must not be zero",
                    location,
                ));
            }

            let k1 = block.const_int(context, location, 1, 32)?;
            let is_shared = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                ref_count,
                k1,
                location,
            ))?;

            block.append_operation(scf::r#yield(&[is_shared], location));
            region
        },
        location,
    ))?;

    Ok(is_shared)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{
            felt252_str,
            test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program},
        },
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;
    use starknet_types_core::felt::Felt;

    #[test]
    fn run_roundtrip() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test(x: Array<u32>) -> Array<u32> {
                x
            }
        );
        let result = run_program(&program, "run_test", &[[1u32, 2u32].into()]).return_value;

        assert_eq!(result, Value::from([1u32, 2u32]));
    }

    #[test]
    fn run_append() {
        let program = load_cairo! {
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, [4u32].into());
    }

    #[test]
    fn run_len() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.len()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, 3u32.into());
    }

    #[test]
    fn run_get() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (u32, u32, u32, u32) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.append(1_u32);
                (
                    *numbers.at(0),
                    *numbers.at(1),
                    *numbers.at(2),
                    *numbers.at(3),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    4u32.into(),
                    3u32.into(),
                    2u32.into(),
                    1u32.into()
                ))
            )
        );
    }

    #[test]
    fn run_get_big() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (u32, u32, u32, u32) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(17_u32);
                numbers.append(17_u32);
                numbers.append(18_u32);
                numbers.append(19_u32);
                numbers.append(20_u32);
                numbers.append(21_u32);
                numbers.append(22_u32);
                numbers.append(23_u32);
                (
                    *numbers.at(20),
                    *numbers.at(21),
                    *numbers.at(22),
                    *numbers.at(23),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    20u32.into(),
                    21u32.into(),
                    22u32.into(),
                    23u32.into()
                ))
            )
        );
    }

    #[test]
    fn run_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                let _ = numbers.pop_front();
                numbers.append(1_u32);
                *numbers.at(0)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_pop_front_result() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Option<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.pop_front()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, 4u32.into()));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Option<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.pop_front()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(1, jit_struct!()));
    }

    #[test]
    fn run_pop_front_consume() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                match numbers.pop_front_consume() {
                    Option::Some((_, x)) => x,
                    Option::None(()) => 0_u32,
                }
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, 4u32.into());
    }

    #[test]
    fn run_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(1_u32);
                let mut numbers = numbers.span();
                (
                    numbers.pop_back(),
                    numbers.pop_back(),
                    numbers.pop_back(),
                    numbers.pop_back(),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_struct!(
                jit_enum!(0, 1u32.into()),
                jit_enum!(0, 3u32.into()),
                jit_enum!(0, 4u32.into()),
                jit_enum!(
                    1,
                    Value::Struct {
                        fields: Vec::new(),
                        debug_name: None,
                    }
                ),
            ),
        );
    }

    #[test]
    fn run_slice() {
        let program = load_cairo!(
            use array::Array;
            use array::ArrayTrait;
            use array::SpanTrait;
            use option::OptionTrait;
            use box::BoxTrait;

            fn run_test() -> u32 {
                let mut data: Array<u32> = ArrayTrait::new(); // Alloca (freed).
                data.append(1_u32);
                data.append(2_u32);
                data.append(3_u32);
                data.append(4_u32);
                let sp = data.span(); // Alloca (leaked).
                let slice = sp.slice(1, 2);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32); // Realloc (freed).
                data.append(5_u32);
                *slice.get(1).unwrap().unbox()
            }

        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_slice_fail() {
        let program = load_cairo!(
            use array::Array;
            use array::ArrayTrait;
            use array::SpanTrait;
            use option::OptionTrait;
            use box::BoxTrait;

            fn run_test() -> u32 {
                let mut data: Array<u32> = ArrayTrait::new();
                data.append(1_u32);
                data.append(2_u32);
                data.append(3_u32);
                data.append(4_u32);
                let sp = data.span();
                let slice = sp.slice(1, 4); // oob
                //data.append(5_u32);
                *slice.get(0).unwrap().unbox()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_panic!(felt252_str(
                "1637570914057682275393755530660268060279989363"
            ))
        );
    }

    #[test]
    fn run_slice_empty_array() {
        let program = load_cairo!(
            fn run_test() -> Span<felt252> {
                let x: Span<felt252> = array![].span();
                x.slice(0, 0)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Struct {
                        fields: vec![Value::Array(vec![])],
                        debug_name: None,
                    }],
                    debug_name: None,
                }),
                debug_name: None
            },
        );
    }

    #[test]
    fn run_span_from_tuple() {
        let program = load_cairo!(
            mod felt252_span_from_tuple {
                pub extern fn span_from_tuple<T>(struct_like: Box<@T>) -> @Array<felt252> nopanic;
            }

            fn run_test() -> Array<felt252> {
                let span = felt252_span_from_tuple::span_from_tuple(BoxTrait::new(@(10, 20, 30)));
                span.clone()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(Value::from([
                    Value::Felt252(Felt::from(10)),
                    Felt::from(20).into(),
                    Felt::from(30).into()
                ]))
            )
        );
    }

    #[test]
    fn run_span_from_multi_tuple() {
        let program = load_cairo!(
            mod tuple_span_from_tuple {
                pub extern fn span_from_tuple<T>(
                    struct_like: Box<@T>
                ) -> @Array<(felt252, felt252, felt252)> nopanic;
            }

            fn run_test() {
                let multi_tuple = ((10, 20, 30), (40, 50, 60), (70, 80, 90));
                let span = tuple_span_from_tuple::span_from_tuple(BoxTrait::new(@multi_tuple));
                assert!(*span[0] == (10, 20, 30));
                assert!(*span[1] == (40, 50, 60));
                assert!(*span[2] == (70, 80, 90));
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(jit_struct!())));
    }

    #[test]
    fn seq_append1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([1u32]),
        );
    }

    #[test]
    fn seq_append2() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([1u32, 2u32]),
        );
    }

    #[test]
    fn seq_append2_popf1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let _ = data.pop_front();
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append2_popb1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let mut data = data.span();
                let _ = data.pop_back();
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_struct!([1u32].into())
        );
    }

    #[test]
    fn seq_append1_popf1_append1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                let _ = data.pop_front();
                data.append(2);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_popf1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let _ = data.pop_front();
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(2u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_popb1_last() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let mut data_span = data.span();
                let _ = data_span.pop_back();
                let last = data_span.len() - 1;
                *data_span.at(last)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            }
        );
    }

    #[test]
    fn seq_append1_popf1_append1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                let _ = data.pop_front();
                data.append(2);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(2u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn array_clone() {
        let program = load_cairo!(
            fn run_test() -> Array<u32> {
                let x = ArrayTrait::new();
                x.clone()
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Array(vec![])],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn array_pop_back_state() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(1_u32);
                numbers.append(2_u32);
                numbers.append(3_u32);
                let mut numbers = numbers.span();
                let _ = numbers.pop_back();
                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_struct!([1u32, 2u32].into()));
    }

    #[test]
    fn array_empty_span() {
        // Tests snapshot_take on a empty array.
        let program = load_cairo!(
            fn run_test() -> Span<u32> {
                let x = ArrayTrait::new();
                x.span()
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_struct!(Value::Array(vec![])),
        );
    }

    #[test]
    fn array_span_modify_span() {
        // Tests pop_back on a span.
        let program = load_cairo!(
            use core::array::SpanTrait;
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
                let x = self.pop_back();
                x
            }

            fn run_test() -> Option<@u64> {
                let mut data = array![2].span();
                let x = pop_elem(data);
                x
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_enum!(0, 2u64.into()),
        );
    }

    #[test]
    fn array_span_check_array() {
        // Tests pop back on a span not modifying the original array.
        let program = load_cairo!(
            use core::array::SpanTrait;
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
                let x = self.pop_back();
                x
            }

            fn run_test() -> Array<u64> {
                let mut data = array![1, 2];
                let _x = pop_elem(data.span());
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Array(vec![1u64.into(), 2u64.into()]),
        );
    }

    #[test]
    fn tuple_from_span() {
        let program = load_cairo! {
            use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};

            fn run_test(x: Array<felt252>) -> [felt252; 3] {
                (*tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x).unwrap()).unbox()
            }
        };

        assert_eq!(
            run_program(
                &program,
                "run_test",
                &[Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ])],
            )
            .return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Struct {
                        fields: vec![
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into()),
                        ],
                        debug_name: None
                    }],
                    debug_name: None
                }),
                debug_name: None
            }
        );
    }

    #[test]
    fn tuple_from_span_failed() {
        let program = load_cairo! {
            use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};

            fn run_test(x: Array<felt252>) -> Option<@Box<[core::felt252; 3]>> {
                tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x)
            }
        };

        assert_eq!(
            run_program(
                &program,
                "run_test",
                &[Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                ])],
            )
            .return_value,
            jit_enum!(1, jit_struct!())
        );
    }

    #[test]
    fn snapshot_multi_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped = numbers.multi_pop_front::<3>().unwrap();

                (numbers, popped)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(4.into()),
                            Value::Felt252(5.into()),
                            Value::Felt252(6.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into())
                        ),
                    )
                )
            )
        );
    }

    #[test]
    fn snapshot_failed_multi_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<felt252> {
                let mut numbers = array![1, 2].span();

                // should fail (return none)
                assert!(numbers.multi_pop_front::<3>().is_none());

                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Span of original array
                    jit_struct!(Value::Array(vec![
                        Value::Felt252(1.into()),
                        Value::Felt252(2.into()),
                    ]),)
                )
            )
        );
    }

    #[test]
    fn snapshot_multi_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped = numbers.multi_pop_back::<3>().unwrap();

                (numbers, popped)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(
                            Value::Felt252(4.into()),
                            Value::Felt252(5.into()),
                            Value::Felt252(6.into())
                        ),
                    )
                )
            )
        );
    }

    #[test]
    fn snapshot_failed_multi_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<felt252> {
                let mut numbers = array![1, 2].span();

                // should fail (return none)
                assert!(numbers.multi_pop_back::<3>().is_none());

                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Span of original array
                    jit_struct!(Value::Array(vec![
                        Value::Felt252(1.into()),
                        Value::Felt252(2.into()),
                    ]),)
                )
            )
        );
    }

    #[test]
    fn snapshot_multi_pop_back_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 2]>, @Box<[felt252; 2]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped_front = numbers.multi_pop_front::<2>().unwrap();
                let popped_back = numbers.multi_pop_back::<2>().unwrap();

                (numbers, popped_front, popped_back)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(3.into()),
                            Value::Felt252(4.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(Value::Felt252(1.into()), Value::Felt252(2.into()),),
                        // Box of fixed array
                        jit_struct!(Value::Felt252(5.into()), Value::Felt252(6.into())),
                    )
                )
            )
        );
    }

    /// Test to ensure that the returned element in `array_get` does NOT get dropped.
    #[test]
    fn array_get_avoid_dropping_element() {
        let program = load_cairo! {
            use core::{array::{array_append, array_at, array_new}, box::{into_box, unbox}};

            fn run_test() -> @Box<felt252> {
                let mut x: Array<Box<felt252>> = array_new();
                array_append(ref x, into_box(42));

                unbox(array_at(@x, 0))
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(Value::Felt252(42.into()))));
    }

    #[test]
    fn array_snapshot_pop_front_clone_offset() {
        let program = load_cairo! {
            fn run_test() -> Span<felt252> {
                let data = array![7, 3, 4, 193827];
                let mut data = data.span();

                assert(*data.pop_front().unwrap() == 7, 0);
                let data2 = data.clone();

                assert(*data.pop_front().unwrap() == 3, 1);

                drop(data2);
                data
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(Value::Array(vec![
                    Value::Felt252(4.into()),
                    Value::Felt252(193827.into()),
                ])))
            ),
        );
    }

    #[test]
    fn array_snapshot_pop_back_clone_offset() {
        let program = load_cairo! {
            fn run_test() -> Span<felt252> {
                let data = array![7, 3, 4, 193827];
                let mut data = data.span();

                assert(*data.pop_front().unwrap() == 7, 0);
                let data2 = data.clone();

                assert(*data.pop_back().unwrap() == 193827, 1);

                drop(data2);
                data
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(Value::Array(vec![
                    Value::Felt252(3.into()),
                    Value::Felt252(4.into()),
                ])))
            ),
        );
    }
}
