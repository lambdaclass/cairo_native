//! # Array libfuncs

// TODO: A future possible improvement would be to put the array behind a double pointer and a
//   reference counter, to avoid unnecessary clones.

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        array::ArrayConcreteLibfunc,
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
        llvm::{self, r#type::opaque_pointer, LoadStoreOptions},
        scf,
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Region, Value, ValueLike,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &ArrayConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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
    }
}

/// Generate MLIR operations for the `array_new` libfunc.
pub fn build_new<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let op0 = entry.append_operation(
        OperationBuilder::new("llvm.mlir.null", location)
            .add_results(&[crate::ffi::get_struct_field_type_at(&array_ty, 0)])
            .build(),
    );
    let op1 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, IntegerType::new(context, 32).into()).into(),
        location,
    ));

    let op2 = entry.append_operation(llvm::undef(array_ty, location));
    let op3 = entry.append_operation(llvm::insert_value(
        context,
        op2.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        op0.result(0)?.into(),
        location,
    ));
    let op4 = entry.append_operation(llvm::insert_value(
        context,
        op3.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        op1.result(0)?.into(),
        location,
    ));
    let op5 = entry.append_operation(llvm::insert_value(
        context,
        op4.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[2]),
        op1.result(0)?.into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op5.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `array_append` libfunc.
pub fn build_append<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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
    let opaque_ptr_ty = llvm::r#type::opaque_pointer(context);

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;
    let elem_stride = elem_layout.pad_to_align().size();

    let op_ptr = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        ptr_ty,
        location,
    ));
    let op_length = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let op_capacity = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[2]),
        len_ty,
        location,
    ));

    let op_has_cap = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Uge,
        op_length.result(0)?.into(),
        op_capacity.result(0)?.into(),
        location,
    ));
    let op4 = entry.append_operation(scf::r#if(
        op_has_cap.result(0)?.into(),
        &[array_ty, ptr_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let op4 = block.append_operation(arith::constant(
                context,
                IntegerAttribute::new(8, IntegerType::new(context, 32).into()).into(),
                location,
            ));
            let op5 = block.append_operation(arith::addi(
                op_capacity.result(0)?.into(),
                op_capacity.result(0)?.into(),
                location,
            ));
            let op6 = block.append_operation(arith::maxui(
                op4.result(0)?.into(),
                op5.result(0)?.into(),
                location,
            ));

            let op7 = block.append_operation(arith::extui(
                op6.result(0)?.into(),
                IntegerType::new(context, 64).into(),
                location,
            ));
            let op8 = block.append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_stride.try_into()?,
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ));
            let op9 = block.append_operation(arith::muli(
                op7.result(0)?.into(),
                op8.result(0)?.into(),
                location,
            ));

            let op10 = block.append_operation(
                OperationBuilder::new("llvm.bitcast", location)
                    .add_operands(&[op_ptr.result(0)?.into()])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build(),
            );
            let op11 = block.append_operation(ReallocBindingsMeta::realloc(
                context,
                op10.result(0)?.into(),
                op9.result(0)?.into(),
                location,
            ));
            let op12 = block.append_operation(
                OperationBuilder::new("llvm.bitcast", location)
                    .add_operands(&[op11.result(0)?.into()])
                    .add_results(&[ptr_ty])
                    .build(),
            );

            let op13 = block.append_operation(llvm::insert_value(
                context,
                entry.argument(0)?.into(),
                DenseI64ArrayAttribute::new(context, &[0]),
                op12.result(0)?.into(),
                location,
            ));
            let op14 = block.append_operation(llvm::insert_value(
                context,
                op13.result(0)?.into(),
                DenseI64ArrayAttribute::new(context, &[2]),
                op6.result(0)?.into(),
                location,
            ));

            block.append_operation(scf::r#yield(
                &[op14.result(0)?.into(), op12.result(0)?.into()],
                location,
            ));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(
                &[entry.argument(0)?.into(), op_ptr.result(0)?.into()],
                location,
            ));

            region
        },
        location,
    ));

    let op5 = entry.append_operation(
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
            .add_operands(&[op4.result(1)?.into()])
            .add_operands(&[op_length.result(0)?.into()])
            .add_results(&[opaque_ptr_ty])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        entry.argument(1)?.into(),
        op5.result(0)?.into(),
        location,
        LoadStoreOptions::default(),
    ));

    let op6 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, len_ty).into(),
        location,
    ));
    let op7 = entry.append_operation(arith::addi(
        op_length.result(0)?.into(),
        op6.result(0)?.into(),
        location,
    ));

    let op8 = entry.append_operation(llvm::insert_value(
        context,
        op4.result(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        op7.result(0)?.into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op8.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `array_append` libfunc.
pub fn build_len<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let op = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0)?.into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let len = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[len], location));

    Ok(())
}

/// Generate MLIR operations for the `array_get` libfunc.
pub fn build_get<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let range_check = entry.argument(0)?.into();
    let array_val = entry.argument(1)?.into();
    let index_val = entry.argument(2)?.into();

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
        CmpiPredicate::Uge,
        index_val,
        len,
        location,
    ));
    let is_oob = op.result(0)?.into();

    let block_not_oob = helper.append_block(Block::new(&[]));
    let block_oob = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_oob,
        block_oob,
        block_not_oob,
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
            .build(),
    );
    let elem_ptr = op.result(0)?.into();

    // we need to allocate the elem ptr into another malloc because the array can resize and change ptr.
    let op = block_not_oob.append_operation(llvm::nullptr(opaque_pointer(context), location));
    let nullptr = op.result(0)?.into();

    let op = block_not_oob.append_operation(arith::constant(
        context,
        IntegerAttribute::new(
            elem_layout.pad_to_align().size().try_into()?,
            IntegerType::new(context, 64).into(),
        )
        .into(),
        location,
    ));
    let value_len = op.result(0)?.into();

    let op = block_not_oob.append_operation(ReallocBindingsMeta::realloc(
        context, nullptr, value_len, location,
    ));

    let new_elem_ptr = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::load(
        context,
        elem_ptr,
        elem_ty,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            elem_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let elem_value = op.result(0)?.into();

    block_not_oob.append_operation(llvm::store(
        context,
        elem_value,
        new_elem_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            elem_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    block_not_oob.append_operation(helper.br(0, &[range_check, new_elem_ptr], location));

    Ok(())
}

/// Generate MLIR operations for the `array_pop_front` libfunc.
pub fn build_pop_front<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;

    let elem_stride = registry
        .get_type(&info.ty)?
        .layout(registry)?
        .pad_to_align()
        .size();

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let array_val = entry.argument(0)?.into();

    // get len
    let op = entry.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let len: Value = op.result(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, len.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0)?.into();

    // check if array is empty
    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        len,
        const_0,
        location,
    ));
    let is_empty = op.result(0)?.into();

    let block_not_empty = helper.append_block(Block::new(&[]));
    let block_empty = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_empty,
        block_empty,
        block_not_empty,
        &[],
        &[],
        location,
    ));

    // empty branch
    block_empty.append_operation(helper.br(1, &[array_val], location));

    // non empty branch

    // get ptr
    let op = block_not_empty.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[0]),
        ptr_ty,
        location,
    ));
    let array_ptr = op.result(0)?.into();

    // get the first elem
    let op = block_not_empty.append_operation(
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
            .add_operands(&[array_ptr, const_0])
            .add_results(&[opaque_pointer(context)])
            .build(),
    );
    let elem_ptr = op.result(0)?.into();

    // we need to allocate the elem ptr into another malloc because the array can resize and change ptr.
    let op = block_not_empty.append_operation(llvm::nullptr(opaque_pointer(context), location));
    let nullptr = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::constant(
        context,
        IntegerAttribute::new(
            elem_layout.pad_to_align().size().try_into()?,
            IntegerType::new(context, 64).into(),
        )
        .into(),
        location,
    ));
    let value_len = op.result(0)?.into();

    let op = block_not_empty.append_operation(ReallocBindingsMeta::realloc(
        context, nullptr, value_len, location,
    ));

    let new_elem_ptr = op.result(0)?.into();

    let op = block_not_empty.append_operation(llvm::load(
        context,
        elem_ptr,
        elem_ty,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            elem_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let elem_value = op.result(0)?.into();

    block_not_empty.append_operation(llvm::store(
        context,
        elem_value,
        new_elem_ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            elem_layout.align() as i64,
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let op = block_not_empty.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, len.r#type()).into(),
        location,
    ));
    let const_1 = op.result(0)?.into();

    let op = block_not_empty.append_operation(
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
            .add_operands(&[array_ptr, const_1])
            .add_results(&[opaque_pointer(context)])
            .build(),
    );
    let array_ptr_src = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::subi(len, const_1, location));
    let len_min_1_i32 = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::extui(
        len_min_1_i32,
        IntegerType::new(context, 64).into(),
        location,
    ));
    let len_min_1 = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::constant(
        context,
        IntegerAttribute::new(
            elem_stride.try_into()?,
            IntegerType::new(context, 64).into(),
        )
        .into(),
        location,
    ));
    let elem_stride_val = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::muli(len_min_1, elem_stride_val, location));
    let elems_size = op.result(0)?.into();

    let op = block_not_empty.append_operation(
        OperationBuilder::new("llvm.bitcast", location)
            .add_operands(&[array_ptr])
            .add_results(&[llvm::r#type::opaque_pointer(context)])
            .build(),
    );
    let array_opaque_ptr = op.result(0)?.into();

    let array_ptr_src_opaque = array_ptr_src;

    block_not_empty.append_operation(
        OperationBuilder::new("llvm.intr.memmove", location)
            .add_attributes(&[(
                Identifier::new(context, "isVolatile"),
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
            )])
            .add_operands(&[array_opaque_ptr, array_ptr_src_opaque, elems_size])
            .build(),
    );

    let op = block_not_empty.append_operation(llvm::insert_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[1]),
        len_min_1_i32,
        location,
    ));
    let array_val = op.result(0)?.into();

    block_not_empty.append_operation(helper.br(0, &[array_val, new_elem_ptr], location));

    Ok(())
}

/// Generate MLIR operations for the `array_pop_front_consume` libfunc.
pub fn build_pop_front_consume<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // same signature at sierra level
    build_pop_front(context, registry, entry, location, helper, metadata, info)
}

/// Generate MLIR operations for the `array_snapshot_pop_front` libfunc.
pub fn build_snapshot_pop_front<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // same signature at sierra level
    // TODO: is this really equal?
    build_pop_front(context, registry, entry, location, helper, metadata, info)
}

/// Generate MLIR operations for the `array_snapshot_pop_back` libfunc.
pub fn build_snapshot_pop_back<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let array_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;

    let (elem_ty, elem_layout) = registry.build_type_with_layout(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let array_val = entry.argument(0)?.into();

    // get len
    let op = entry.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let len: Value = op.result(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, len.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0)?.into();

    // check if array is empty
    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        len,
        const_0,
        location,
    ));
    let is_empty = op.result(0)?.into();

    let block_not_empty = helper.append_block(Block::new(&[]));
    let block_empty = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_empty,
        block_empty,
        block_not_empty,
        &[],
        &[],
        location,
    ));

    // empty branch
    block_empty.append_operation(helper.br(1, &[array_val], location));

    // non empty branch

    // get ptr
    let op = block_not_empty.append_operation(llvm::extract_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[0]),
        ptr_ty,
        location,
    ));
    let array_ptr = op.result(0)?.into();

    // get the last elem
    let op = block_not_empty.append_operation(
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
            .add_operands(&[array_ptr, len])
            .add_results(&[opaque_pointer(context)])
            .build(),
    );
    let elem_ptr = op.result(0)?.into();

    let op = block_not_empty.append_operation(llvm::load(
        context,
        elem_ptr,
        elem_ty,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            elem_layout.align().try_into()?,
            IntegerType::new(context, 64).into(),
        ))),
    ));
    let elem_value = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, len.r#type()).into(),
        location,
    ));
    let const_1 = op.result(0)?.into();

    let op = block_not_empty.append_operation(arith::subi(len, const_1, location));
    let len_min_1_i32 = op.result(0)?.into();

    let op = block_not_empty.append_operation(llvm::insert_value(
        context,
        array_val,
        DenseI64ArrayAttribute::new(context, &[1]),
        len_min_1_i32,
        location,
    ));
    let array_val = op.result(0)?.into();

    block_not_empty.append_operation(helper.br(0, &[array_val, elem_value], location));

    Ok(())
}

/// Generate MLIR operations for the `array_slice` libfunc.
pub fn build_slice<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
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

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, registry, metadata, &info.ty)?;

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let range_check = entry.argument(0)?.into();
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
            .build(),
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
        StringAttribute::new(context, "llvm.memcpy.inline"),
        &[new_ptr, elem_ptr, bytes_val, is_volatile],
        &[],
        location,
    ));

    let op = block_not_oob.append_operation(llvm::undef(array_ty, location));
    let new_array_value = op.result(0)?.into();

    let op = block_not_oob.append_operation(llvm::bitcast(new_ptr, ptr_ty, location));
    let new_ptr = op.result(0)?.into();

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

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};
    use serde_json::json;

    #[test]
    fn run_roundtrip() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test(x: Array<u32>) -> Array<u32> {
                x
            }
        );
        let result = run_program(&program, "run_test", json!([[1, 2]]));

        assert_eq!(result, json!([[1, 2]]));
    }

    #[test]
    fn run_append() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[4]]));
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
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([3]));
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
        let result = run_program(&program, "run_test", json!([null]));

        assert_eq!(result, json!([null, [0, [[4, 3, 2, 1]]]]));
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
        let result = run_program(&program, "run_test", json!([null]));

        assert_eq!(result, json!([null, [0, [[20, 21, 22, 23]]]]));
    }

    #[test]
    fn run_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.pop_front();
                numbers.append(1_u32);
                *numbers.at(0)
            }
        );
        let result = run_program(&program, "run_test", json!([null]));

        assert_eq!(result, json!([null, [0, [3]]]));
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
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[0, 4]]));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Option<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.pop_front()
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([[1, []]]));
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
                    Option::Some((arr, x)) => x,
                    Option::None(()) => 0_u32,
                }
            }
        );
        let result = run_program(&program, "run_test", json!([]));

        assert_eq!(result, json!([4]));
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
        let result = run_program(&program, "run_test", json!([()]));

        assert_eq!(result, json!([null, [0, [3]]]));
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
        let result = run_program(&program, "run_test", json!([()]));

        assert_eq!(
            result,
            json!([
                null,
                [
                    1,
                    [
                        [],
                        [[1970168947, 1713398383, 1970544751, 1702371439, 4812388, 0, 0, 0]]
                    ]
                ]
            ])
        );
    }
}
