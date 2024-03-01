//! # Array libfuncs

// TODO: A future possible improvement would be to put the array behind a double pointer and a
//   reference counter, to avoid unnecessary clones.

use super::LibfuncHelper;
use crate::{
    error::libfuncs::Result,
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        array::ArrayConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
        llvm::{self, r#type::opaque_pointer, LoadStoreOptions},
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Value, ValueLike,
    },
    Context,
};

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
        ArrayConcreteLibfunc::Append(info) => {
            build_append(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::PopFront(info) => {
            build_pop_front(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::PopFrontConsume(info) => {
            build_pop_front_consume(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Slice(info) => {
            build_slice(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Len(info) => {
            build_len(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SnapshotPopFront(info) => {
            build_snapshot_pop_front(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SnapshotPopBack(info) => {
            build_snapshot_pop_back(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SpanFromTuple(info) => {
            build_span_from_tuple(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `array_new` libfunc.
pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let ptr = entry
        .append_operation(llvm::nullptr(
            crate::ffi::get_struct_field_type_at(&array_ty, 0),
            location,
        ))
        .result(0)?
        .into();
    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let value = entry
        .append_operation(llvm::undef(array_ty, location))
        .result(0)?
        .into();
    let value = entry
        .append_operation(llvm::insert_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[0]),
            ptr,
            location,
        ))
        .result(0)?
        .into();
    let value = entry
        .append_operation(llvm::insert_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[1]),
            k0,
            location,
        ))
        .result(0)?
        .into();
    let value = entry
        .append_operation(llvm::insert_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[2]),
            k0,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
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
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;
    let elem_stride = elem_layout.pad_to_align().size();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let elem_stride = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(elem_stride as i64, IntegerType::new(context, 64).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let len = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let cap = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[2]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();

    let should_realloc = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Uge, len, cap, location))
        .result(0)?
        .into();

    let realloc_block = helper.append_block(Block::new(&[]));
    let insert_block = helper.append_block(Block::new(&[(array_ty, location)]));
    entry.append_operation(cf::cond_br(
        context,
        should_realloc,
        realloc_block,
        insert_block,
        &[],
        &[entry.argument(0)?.into()],
        location,
    ));

    {
        let k8 = realloc_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(8, IntegerType::new(context, 32).into()).into(),
                location,
            ))
            .result(0)?
            .into();
        let k1024 = realloc_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(1024, IntegerType::new(context, 32).into()).into(),
                location,
            ))
            .result(0)?
            .into();

        // Array allocation growth formula:
        //   new_len = max(8, old_len + min(1024, 2 * old_len));
        let new_len = realloc_block
            .append_operation(arith::shli(len, k1, location))
            .result(0)?
            .into();
        let new_len = realloc_block
            .append_operation(arith::minui(new_len, k1024, location))
            .result(0)?
            .into();
        let new_len = realloc_block
            .append_operation(arith::addi(new_len, len, location))
            .result(0)?
            .into();
        let new_len = realloc_block
            .append_operation(arith::maxui(new_len, k8, location))
            .result(0)?
            .into();

        let new_len_extended = realloc_block
            .append_operation(arith::extui(
                new_len,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let new_size = realloc_block
            .append_operation(arith::muli(elem_stride, new_len_extended, location))
            .result(0)?
            .into();

        let ptr = realloc_block
            .append_operation(llvm::extract_value(
                context,
                entry.argument(0)?.into(),
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();
        let ptr = realloc_block
            .append_operation(ReallocBindingsMeta::realloc(
                context, ptr, new_size, location,
            ))
            .result(0)?
            .into();
        // TODO: Assert that `ptr != nullptr`.

        let value = realloc_block
            .append_operation(llvm::insert_value(
                context,
                entry.argument(0)?.into(),
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr,
                location,
            ))
            .result(0)?
            .into();
        let value = realloc_block
            .append_operation(llvm::insert_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[2]),
                new_len,
                location,
            ))
            .result(0)?
            .into();

        realloc_block.append_operation(cf::br(insert_block, &[value], location));
    }

    {
        let value = insert_block.argument(0)?.into();

        let new_len = insert_block
            .append_operation(arith::addi(len, k1, location))
            .result(0)?
            .into();
        let value = insert_block
            .append_operation(llvm::insert_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[1]),
                new_len,
                location,
            ))
            .result(0)?
            .into();

        let ptr = insert_block
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();

        let insert_offset = insert_block
            .append_operation(arith::extui(
                len,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let insert_offset = insert_block
            .append_operation(arith::muli(elem_stride, insert_offset, location))
            .result(0)?
            .into();
        let elem_ptr = insert_block
            .append_operation(llvm::get_element_ptr_dynamic(
                context,
                ptr,
                &[insert_offset],
                IntegerType::new(context, 8).into(),
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        insert_block.append_operation(llvm::store(
            context,
            entry.argument(1)?.into(),
            elem_ptr,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                elem_layout.align() as i64,
                IntegerType::new(context, 64).into(),
            ))),
        ));

        insert_block.append_operation(helper.br(0, &[value], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `array_append` libfunc.
pub fn build_len<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let len = entry
        .append_operation(llvm::extract_value(
            context,
            entry.argument(0)?.into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[len], location));
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
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;
    let elem_stride = elem_layout.pad_to_align().size();

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let value = entry.argument(1)?.into();
    let index = entry.argument(2)?.into();

    let len = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let is_valid = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ult,
            index,
            len,
            location,
        ))
        .result(0)?
        .into();

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
        let ptr = valid_block
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();

        let elem_stride = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(elem_stride as i64, IntegerType::new(context, 32).into())
                    .into(),
                location,
            ))
            .result(0)?
            .into();
        let elem_offset = valid_block
            .append_operation(arith::muli(elem_stride, index, location))
            .result(0)?
            .into();

        let elem_ptr = valid_block
            .append_operation(llvm::get_element_ptr_dynamic(
                context,
                ptr,
                &[elem_offset],
                IntegerType::new(context, 8).into(),
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        let elem_size = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_layout.size() as i64,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        let target_ptr = valid_block
            .append_operation(llvm::nullptr(
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();
        let target_ptr = valid_block
            .append_operation(ReallocBindingsMeta::realloc(
                context, target_ptr, elem_size, location,
            ))
            .result(0)?
            .into();
        // TODO: Assert that `target_ptr != nullptr`.

        let is_volatile = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                Location::unknown(context),
            ))
            .result(0)?
            .into();

        // TODO: Support clone-only types (those that are not copy).
        valid_block.append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.memcpy.inline"),
            &[target_ptr, elem_ptr, elem_size, is_volatile],
            &[],
            location,
        ));

        valid_block.append_operation(helper.br(0, &[range_check, target_ptr], location));
    }

    error_block.append_operation(helper.br(1, &[range_check], location));
    Ok(())
}

/// Generate MLIR operations for the `array_pop_front` libfunc.
pub fn build_pop_front<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let value = entry.argument(0)?.into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let len = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let is_empty = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Eq, len, k0, location))
        .result(0)?
        .into();

    let valid_block = helper.append_block(Block::new(&[]));
    let empty_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_empty,
        empty_block,
        valid_block,
        &[],
        &[],
        location,
    ));

    {
        let ptr = valid_block
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();

        let elem_size = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_layout.size() as i64,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        let target_ptr = valid_block
            .append_operation(llvm::nullptr(
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();
        let target_ptr = valid_block
            .append_operation(ReallocBindingsMeta::realloc(
                context, target_ptr, elem_size, location,
            ))
            .result(0)?
            .into();
        // TODO: Assert that `target_ptr != nullptr`.

        let is_volatile = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                Location::unknown(context),
            ))
            .result(0)?
            .into();

        // TODO: Support clone-only types (those that are not copy).
        valid_block.append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.memcpy.inline"),
            &[target_ptr, ptr, elem_size, is_volatile],
            &[],
            location,
        ));

        let k1 = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(1, IntegerType::new(context, 32).into()).into(),
                location,
            ))
            .result(0)?
            .into();
        let new_len = valid_block
            .append_operation(arith::subi(len, k1, location))
            .result(0)?
            .into();
        let value = valid_block
            .append_operation(llvm::insert_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[1]),
                new_len,
                location,
            ))
            .result(0)?
            .into();

        let elem_stride = elem_layout.pad_to_align().size();
        let source_ptr = valid_block
            .append_operation(llvm::get_element_ptr(
                context,
                ptr,
                DenseI32ArrayAttribute::new(context, &[elem_stride as i32]),
                IntegerType::new(context, 8).into(),
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        let elem_stride = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(elem_stride as i64, IntegerType::new(context, 32).into())
                    .into(),
                location,
            ))
            .result(0)?
            .into();
        let new_size = valid_block
            .append_operation(arith::muli(elem_stride, new_len, location))
            .result(0)?
            .into();

        let is_volatile = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                Location::unknown(context),
            ))
            .result(0)?
            .into();
        valid_block.append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.memmove"),
            &[ptr, source_ptr, new_size, is_volatile],
            &[],
            location,
        ));

        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
    }

    empty_block.append_operation(helper.br(1, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `array_pop_front_consume` libfunc.
pub fn build_pop_front_consume<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Equivalent to `array_pop_front_consume` for our purposes.
    build_pop_front(context, registry, entry, location, helper, metadata, info)
}

/// Generate MLIR operations for the `array_snapshot_pop_front` libfunc.
pub fn build_snapshot_pop_front<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let value = entry.argument(0)?.into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let offset = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let length = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[2]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let slice_len = entry
        .append_operation(arith::subi(length, offset, location))
        .result(0)?
        .into();
    let is_empty = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            slice_len,
            k0,
            location,
        ))
        .result(0)?
        .into();

    let valid_block = helper.append_block(Block::new(&[]));
    let empty_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_empty,
        empty_block,
        valid_block,
        &[],
        &[],
        location,
    ));

    {
        let ptr = valid_block
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();

        let elem_size = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_layout.size() as i64,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        let elem_offset = valid_block
            .append_operation(arith::extui(
                offset,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let elem_offset = valid_block
            .append_operation(arith::muli(elem_size, elem_offset, location))
            .result(0)?
            .into();
        let ptr = valid_block
            .append_operation(llvm::get_element_ptr_dynamic(
                context,
                ptr,
                &[elem_offset],
                IntegerType::new(context, 8).into(),
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        let k1 = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(1, IntegerType::new(context, 32).into()).into(),
                location,
            ))
            .result(0)?
            .into();
        let offset = valid_block
            .append_operation(arith::addi(offset, k1, location))
            .result(0)?
            .into();
        let value = valid_block
            .append_operation(llvm::insert_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[1]),
                offset,
                location,
            ))
            .result(0)?
            .into();

        let target_ptr = valid_block
            .append_operation(llvm::nullptr(
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();
        let target_ptr = valid_block
            .append_operation(ReallocBindingsMeta::realloc(
                context, target_ptr, elem_size, location,
            ))
            .result(0)?
            .into();
        // TODO: Assert that `target_ptr != nullptr`.

        let is_volatile = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                Location::unknown(context),
            ))
            .result(0)?
            .into();

        // TODO: Support clone-only types (those that are not copy).
        valid_block.append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.memcpy.inline"),
            &[target_ptr, ptr, elem_size, is_volatile],
            &[],
            location,
        ));

        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
    }

    empty_block.append_operation(helper.br(1, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `array_snapshot_pop_back` libfunc.
pub fn build_snapshot_pop_back<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_layout = elem_ty.layout(registry)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let value = entry.argument(0)?.into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let offset = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[1]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let length = entry
        .append_operation(llvm::extract_value(
            context,
            value,
            DenseI64ArrayAttribute::new(context, &[2]),
            len_ty,
            location,
        ))
        .result(0)?
        .into();
    let slice_len = entry
        .append_operation(arith::subi(length, offset, location))
        .result(0)?
        .into();
    let is_empty = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            slice_len,
            k0,
            location,
        ))
        .result(0)?
        .into();

    let valid_block = helper.append_block(Block::new(&[]));
    let empty_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_empty,
        empty_block,
        valid_block,
        &[],
        &[],
        location,
    ));

    {
        let ptr = valid_block
            .append_operation(llvm::extract_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[0]),
                ptr_ty,
                location,
            ))
            .result(0)?
            .into();

        let elem_size = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_layout.size() as i64,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        let k1 = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(1, IntegerType::new(context, 32).into()).into(),
                location,
            ))
            .result(0)?
            .into();
        let length = valid_block
            .append_operation(arith::subi(length, k1, location))
            .result(0)?
            .into();
        let value = valid_block
            .append_operation(llvm::insert_value(
                context,
                value,
                DenseI64ArrayAttribute::new(context, &[2]),
                length,
                location,
            ))
            .result(0)?
            .into();

        let elem_offset = valid_block
            .append_operation(arith::extui(
                length,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let elem_offset = valid_block
            .append_operation(arith::muli(elem_size, elem_offset, location))
            .result(0)?
            .into();
        let ptr = valid_block
            .append_operation(llvm::get_element_ptr_dynamic(
                context,
                ptr,
                &[elem_offset],
                IntegerType::new(context, 8).into(),
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        let target_ptr = valid_block
            .append_operation(llvm::nullptr(
                llvm::r#type::opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();
        let target_ptr = valid_block
            .append_operation(ReallocBindingsMeta::realloc(
                context, target_ptr, elem_size, location,
            ))
            .result(0)?
            .into();
        // TODO: Assert that `target_ptr != nullptr`.

        let is_volatile = valid_block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                Location::unknown(context),
            ))
            .result(0)?
            .into();

        // TODO: Support clone-only types (those that are not copy).
        valid_block.append_operation(llvm::call_intrinsic(
            context,
            StringAttribute::new(context, "llvm.memcpy.inline"),
            &[target_ptr, ptr, elem_size, is_volatile],
            &[],
            location,
        ));

        valid_block.append_operation(helper.br(0, &[value, target_ptr], location));
    }

    empty_block.append_operation(helper.br(1, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `array_slice` libfunc.
pub fn build_slice<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let array_val = entry.argument(1)?.into();
    let index_val = entry.argument(2)?.into();
    let length_val = entry.argument(3)?.into();

    let op = entry.append_operation(arith::addi(index_val, length_val, location));
    let end_val = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let len: Value = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        end_val,
        len,
        location,
    ));
    let is_inbounds = op.result(0)?.into();

    let block_not_oob = helper.append_block(Block::new(&[]));
    let block_oob = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_inbounds,
        block_not_oob,
        block_oob,
        &[],
        &[],
        location,
    ));

    block_oob.append_operation(helper.br(1, &[range_check], location));

    let op = block_not_oob.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[0]),
        ptr_ty,
        location,
    ));
    let array_ptr = op.result(0)?.into();

    let op = block_not_oob.append_operation(
        OperationBuilder::new("llvm.getelementptr", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "rawConstantIndices"),
                    DenseI32ArrayAttribute::new(context, &[i32::MIN]).into(),
                ),
                (
                    Identifier::new(context, "elem_type"),
                    TypeAttribute::new(elem_ty).into(),
                ),
            ])
            .add_operands(&[array_ptr, index_val])
            .add_results(&[opaque_pointer(context)])
            .build()?,
    );
    let elem_ptr = op.result(0)?.into();

    let stride = elem_layout.pad_to_align().size();

    let op = block_not_oob.append_operation(arith::constant(
        context,
        IntegerAttribute::new(stride as i64, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let stride_val = op.result(0)?.into();

    let op = block_not_oob.append_operation(arith::extui(
        length_val,
        IntegerType::new(context, 64).into(),
        location,
    ));
    let length_val_64 = op.result(0)?.into();

    let op = block_not_oob.append_operation(arith::muli(stride_val, length_val_64, location));

    let bytes_val = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::nullptr(opaque_pointer(context), location));

    let nullptr = op.result(0)?.into();

    let op = block_not_oob.append_operation(ReallocBindingsMeta::realloc(
        context, nullptr, bytes_val, location,
    ));

    let new_ptr = op.result(0)?.into();

    let op = block_not_oob.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
        location,
    ));
    let is_volatile = op.result(0)?.into();

    block_not_oob.append_operation(llvm::call_intrinsic(
        context,
        StringAttribute::new(context, "llvm.memcpy"),
        &[new_ptr, elem_ptr, bytes_val, is_volatile],
        &[],
        location,
    ));

    let op = block_not_oob.append_operation(llvm::undef(array_ty, location));
    let new_array_value = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::insert_value(
        context,
        new_array_value,
        DenseI64ArrayAttribute::new(context, &[0]),
        new_ptr,
        location,
    ));
    let new_array_value = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::insert_value(
        context,
        new_array_value,
        DenseI64ArrayAttribute::new(context, &[1]),
        length_val,
        location,
    ));
    let new_array_value = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::insert_value(
        context,
        new_array_value,
        DenseI64ArrayAttribute::new(context, &[2]),
        length_val,
        location,
    ));
    let new_array_value = op.result(0)?.into();

    block_not_oob.append_operation(helper.br(0, &[range_check, new_array_value], location));

    Ok(())
}

/// Generate MLIR operations for the `span_from_tuple` libfunc.
pub fn build_span_from_tuple<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // tuple to array span (t,t,t) -> &[t,t,t]

    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let struct_type_info = registry.get_type(&info.ty)?;

    let struct_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;

    let container: Value = {
        // load box
        entry
            .append_operation(llvm::load(
                context,
                entry.argument(0)?.into(),
                struct_ty,
                location,
                LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                    struct_type_info.layout(registry)?.align() as i64,
                    IntegerType::new(context, 64).into(),
                ))),
            ))
            .result(0)?
            .into()
    };

    let fields = struct_type_info.fields().expect("should have fields");
    let (field_ty, field_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &fields[0])?;
    let field_stride = field_layout.pad_to_align().size();

    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let array_len_value = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(fields.len().try_into().unwrap(), len_ty).into(),
            location,
        ))
        .result(0)?
        .into();

    let array_container = entry
        .append_operation(llvm::undef(array_ty, location))
        .result(0)?
        .into();

    // set len
    let array_container = entry
        .append_operation(llvm::insert_value(
            context,
            array_container,
            DenseI64ArrayAttribute::new(context, &[1]),
            array_len_value,
            location,
        ))
        .result(0)?
        .into();
    // set capacity
    let array_container = entry
        .append_operation(llvm::insert_value(
            context,
            array_container,
            DenseI64ArrayAttribute::new(context, &[2]),
            array_len_value,
            location,
        ))
        .result(0)?
        .into();

    let opaque_ptr_ty = opaque_pointer(context);

    let ptr = entry
        .append_operation(llvm::nullptr(opaque_ptr_ty, location))
        .result(0)?
        .into();

    let field_size: Value = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                field_stride.try_into().unwrap(),
                IntegerType::new(context, 64).into(),
            )
            .into(),
            location,
        ))
        .result(0)?
        .into();
    let array_len_value_i64 = entry
        .append_operation(arith::extui(array_len_value, field_size.r#type(), location))
        .result(0)?
        .into();
    let total_size = entry
        .append_operation(arith::muli(field_size, array_len_value_i64, location))
        .result(0)?
        .into();

    let ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(
            context, ptr, total_size, location,
        ))
        .result(0)?
        .into();

    for (i, _) in fields.iter().enumerate() {
        let value: Value = entry
            .append_operation(llvm::extract_value(
                context,
                container,
                DenseI64ArrayAttribute::new(context, &[i.try_into()?]),
                field_ty,
                location,
            ))
            .result(0)?
            .into();

        let target_ptr = entry
            .append_operation(llvm::get_element_ptr(
                context,
                ptr,
                DenseI32ArrayAttribute::new(context, &[i as i32]),
                field_ty,
                opaque_pointer(context),
                location,
            ))
            .result(0)?
            .into();

        entry.append_operation(llvm::store(
            context,
            value,
            target_ptr,
            location,
            LoadStoreOptions::default(),
        ));
    }

    let array_container = entry
        .append_operation(llvm::insert_value(
            context,
            array_container,
            DenseI64ArrayAttribute::new(context, &[0]),
            ptr,
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[array_container], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program},
        values::JitValue,
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

        assert_eq!(result, JitValue::from([1u32, 2u32]));
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
    fn run_slice() {
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
                let slice = sp.slice(1, 2);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
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
            jit_panic!(JitValue::felt_str(
                "1637570914057682275393755530660268060279989363"
            ))
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
                jit_struct!(JitValue::from([
                    JitValue::Felt252(Felt::from(10)),
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
}
