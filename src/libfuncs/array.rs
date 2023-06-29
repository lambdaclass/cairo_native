//! # Array libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
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
        func,
        llvm::{self, LoadStoreOptions},
        scf,
    },
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location, Region,
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
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        ArrayConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Append(info) => {
            build_append(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::PopFront(_) => todo!(),
        ArrayConcreteLibfunc::PopFrontConsume(_) => todo!(),
        ArrayConcreteLibfunc::Get(_) => todo!(),
        ArrayConcreteLibfunc::Slice(_) => todo!(),
        ArrayConcreteLibfunc::Len(info) =>  {
            build_len(context, registry, entry, location, helper, metadata, info)
        },
        ArrayConcreteLibfunc::SnapshotPopFront(_) => todo!(),
        ArrayConcreteLibfunc::SnapshotPopBack(_) => todo!(),
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
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let array_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

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
        op2.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        op0.result(0).unwrap().into(),
        location,
    ));
    let op4 = entry.append_operation(llvm::insert_value(
        context,
        op3.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        op1.result(0).unwrap().into(),
        location,
    ));
    let op5 = entry.append_operation(llvm::insert_value(
        context,
        op4.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[2]),
        op1.result(0).unwrap().into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op5.result(0).unwrap().into()], location));

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
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let array_ty = registry
        .get_type(&info.param_signatures()[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let elem_stride = registry
        .get_type(&info.ty)
        .unwrap()
        .layout(registry)
        .pad_to_align()
        .size();

    let op0 = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        ptr_ty,
        location,
    ));
    let op1 = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let op2 = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[2]),
        len_ty,
        location,
    ));

    let op3 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Uge,
        op1.result(0).unwrap().into(),
        op2.result(0).unwrap().into(),
        location,
    ));
    let op4 = entry.append_operation(scf::r#if(
        op3.result(0).unwrap().into(),
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
                op2.result(0).unwrap().into(),
                op2.result(0).unwrap().into(),
                location,
            ));
            let op6 = block.append_operation(arith::maxui(
                op4.result(0).unwrap().into(),
                op5.result(0).unwrap().into(),
                location,
            ));

            let op7 = block.append_operation(arith::extui(
                op6.result(0).unwrap().into(),
                IntegerType::new(context, 64).into(),
                location,
            ));
            let op8 = block.append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    elem_stride.try_into().unwrap(),
                    IntegerType::new(context, 64).into(),
                )
                .into(),
                location,
            ));
            let op9 = block.append_operation(arith::muli(
                op7.result(0).unwrap().into(),
                op8.result(0).unwrap().into(),
                location,
            ));

            let op10 = block.append_operation(
                OperationBuilder::new("llvm.bitcast", location)
                    .add_operands(&[op0.result(0).unwrap().into()])
                    .add_results(&[llvm::r#type::opaque_pointer(context)])
                    .build(),
            );
            let op11 = block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(context, "realloc"),
                &[
                    op10.result(0).unwrap().into(),
                    op9.result(0).unwrap().into(),
                ],
                &[llvm::r#type::opaque_pointer(context)],
                location,
            ));
            let op12 = block.append_operation(
                OperationBuilder::new("llvm.bitcast", location)
                    .add_operands(&[op11.result(0).unwrap().into()])
                    .add_results(&[ptr_ty])
                    .build(),
            );

            let op13 = block.append_operation(llvm::insert_value(
                context,
                entry.argument(0).unwrap().into(),
                DenseI64ArrayAttribute::new(context, &[0]),
                op12.result(0).unwrap().into(),
                location,
            ));
            let op14 = block.append_operation(llvm::insert_value(
                context,
                op13.result(0).unwrap().into(),
                DenseI64ArrayAttribute::new(context, &[2]),
                op6.result(0).unwrap().into(),
                location,
            ));

            block.append_operation(scf::r#yield(
                &[
                    op14.result(0).unwrap().into(),
                    op12.result(0).unwrap().into(),
                ],
                location,
            ));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(
                &[
                    entry.argument(0).unwrap().into(),
                    op0.result(0).unwrap().into(),
                ],
                location,
            ));

            region
        },
        location,
    ));

    let op5 = entry.append_operation(
        OperationBuilder::new("llvm.getelementptr", location)
            .add_attributes(&[(
                Identifier::new(context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(context, &[i32::MIN]).into(),
            )])
            .add_operands(&[op4.result(1).unwrap().into()])
            .add_operands(&[op1.result(0).unwrap().into()])
            .add_results(&[ptr_ty])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        entry.argument(1).unwrap().into(),
        op5.result(0).unwrap().into(),
        location,
        LoadStoreOptions::default(),
    ));

    let op6 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, len_ty).into(),
        location,
    ));
    let op7 = entry.append_operation(arith::addi(
        op1.result(0).unwrap().into(),
        op6.result(0).unwrap().into(),
        location,
    ));

    let op8 = entry.append_operation(llvm::insert_value(
        context,
        op4.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        op7.result(0).unwrap().into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op8.result(0).unwrap().into()], location));

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
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let array_ty = registry
        .get_type(&info.param_signatures()[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let len_ty = crate::ffi::get_struct_field_type_at(&array_ty, 1);

    let op = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        len_ty,
        location,
    ));
    let len = op.result(0).unwrap().into();

    entry.append_operation(helper.br(0, &[len], location));

    Ok(())
}
