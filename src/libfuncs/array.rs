use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    array::ArrayConcreteLibfunc,
    lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
    GenericLibfunc, GenericType,
};
use melior::{
    dialect::{arith, cf, index, llvm, memref, scf},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::{IntegerType, MemRefType},
        Block, Region, Type, Value, ValueLike,
    },
};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &ArrayConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        ArrayConcreteLibfunc::New(info) => build_new(context, info),
        ArrayConcreteLibfunc::Append(info) => build_append(context, info),
        ArrayConcreteLibfunc::PopFront(_) => todo!(),
        ArrayConcreteLibfunc::PopFrontConsume(_) => todo!(),
        ArrayConcreteLibfunc::Get(_) => todo!(),
        ArrayConcreteLibfunc::Slice(_) => todo!(),
        ArrayConcreteLibfunc::Len(_) => todo!(),
        ArrayConcreteLibfunc::SnapshotPopFront(_) => todo!(),
        ArrayConcreteLibfunc::SnapshotPopBack(_) => todo!(),
    }
}

pub fn build_new<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let struct_ty = context
        .registry()
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(*context)
        .unwrap();
    let memref_ty: MemRefType = crate::ffi::get_struct_field_type_at(&struct_ty, 0)
        .try_into()
        .unwrap();
    let elem_ty = crate::ffi::get_memref_element_type(&memref_ty);

    let op0 = context.entry().append_operation(arith::constant(
        context.context(),
        IntegerAttribute::new(0, Type::index(context.context())).into(),
        context.location(),
    ));
    let op1 = context.entry().append_operation(memref::alloc(
        context.context(),
        memref_ty,
        &[op0.result(0).unwrap().into()],
        &[],
        Some(IntegerAttribute::new(
            crate::ffi::get_abi_alignment(context.module(), &elem_ty) as _,
            IntegerType::new(context.context(), 64).into(),
        )),
        context.location(),
    ));

    let op2 = context
        .entry()
        .append_operation(llvm::undef(struct_ty, context.location()));

    let op3 = context.entry().append_operation(llvm::insert_value(
        context.context(),
        op2.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[0]),
        op1.result(0).unwrap().into(),
        context.location(),
    ));

    let op4 = context.entry().append_operation(arith::constant(
        context.context(),
        IntegerAttribute::new(0, IntegerType::unsigned(context.context(), 32).into()).into(),
        context.location(),
    ));
    let op5 = context.entry().append_operation(llvm::insert_value(
        context.context(),
        op3.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[1]),
        op4.result(0).unwrap().into(),
        context.location(),
    ));

    context
        .entry()
        .append_operation(context.br(0, &[op5.result(0).unwrap().into()]));

    Ok(())
}

pub fn build_append<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let memref_ty =
        crate::ffi::get_struct_field_type_at(&context.entry().argument(0).unwrap().r#type(), 0)
            .try_into()
            .unwrap();
    let elem_ty = crate::ffi::get_memref_element_type(&memref_ty);

    let op0 = context.entry().append_operation(llvm::extract_value(
        context.context(),
        context.entry().argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[0]),
        elem_ty,
        context.location(),
    ));
    let op1 = context.entry().append_operation(llvm::extract_value(
        context.context(),
        context.entry().argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[1]),
        IntegerType::unsigned(context.context(), 32).into(),
        context.location(),
    ));

    let ptr: Value = op0.result(0).unwrap().into();
    let len: Value = op1.result(0).unwrap().into();

    let op2 = context.entry().append_operation(index::constant(
        context.context(),
        IntegerAttribute::new(0, IntegerType::new(context.context(), 64).into()),
        context.location(),
    ));
    let op3 = context.entry().append_operation(memref::dim(
        ptr,
        op2.result(0).unwrap().into(),
        context.location(),
    ));

    let op4 = context.entry().append_operation(index::castu(
        len,
        Type::index(context.context()),
        context.location(),
    ));
    let op5 = context.entry().append_operation(index::cmp(
        context.context(),
        arith::CmpiPredicate::Uge,
        op4.result(0).unwrap().into(),
        op3.result(0).unwrap().into(),
        context.location(),
    ));
    let op6 = context.entry().append_operation(scf::r#if(
        op5.result(0).unwrap().into(),
        &[memref_ty.into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let op6 = block.append_operation(index::constant(
                context.context(),
                IntegerAttribute::new(1, Type::index(context.context())),
                context.location(),
            ));
            let op7 = block.append_operation(index::shl(
                op4.result(0).unwrap().into(),
                op6.result(0).unwrap().into(),
                context.location(),
            ));

            let op8 = block.append_operation(index::constant(
                context.context(),
                IntegerAttribute::new(8, Type::index(context.context())),
                context.location(),
            ));
            let op9 = block.append_operation(index::maxu(
                op7.result(0).unwrap().into(),
                op8.result(0).unwrap().into(),
                context.location(),
            ));

            let op10 = block.append_operation(memref::realloc(
                context.context(),
                ptr,
                Some(op9.result(0).unwrap().into()),
                memref_ty,
                Some(IntegerAttribute::new(
                    crate::ffi::get_abi_alignment(context.module(), &elem_ty) as _,
                    Type::index(context.context()),
                )),
                context.location(),
            ));
            block.append_operation(scf::r#yield(
                &[op10.result(0).unwrap().into()],
                context.location(),
            ));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[ptr], context.location()));

            region
        },
        context.location(),
    ));

    let ptr: Value = op6.result(0).unwrap().into();

    context.entry().append_operation(memref::store(
        context.entry().argument(1).unwrap().into(),
        ptr,
        &[op4.result(0).unwrap().into()],
        context.location(),
    ));

    let op7 = context.entry().append_operation(index::constant(
        context.context(),
        IntegerAttribute::new(1, Type::index(context.context())),
        context.location,
    ));
    let op8 = context.entry().append_operation(index::add(
        op4.result(0).unwrap().into(),
        op7.result(0).unwrap().into(),
        context.location(),
    ));

    let op9 = context.entry().append_operation(index::constant(
        context.context(),
        IntegerAttribute::new(u32::MAX as _, Type::index(context.context())),
        context.location(),
    ));
    let op10 = context.entry().append_operation(index::cmp(
        context.context(),
        arith::CmpiPredicate::Uge,
        op8.result(0).unwrap().into(),
        op9.result(0).unwrap().into(),
        context.location(),
    ));
    context.entry().append_operation(cf::assert(
        context.context(),
        op10.result(0).unwrap().into(),
        "Array size overflow (max. 4294967295).",
        context.location(),
    ));

    let op11 = context.entry().append_operation(index::castu(
        op8.result(0).unwrap().into(),
        IntegerType::unsigned(context.context(), 32).into(),
        context.location(),
    ));

    let op12 = context.entry().append_operation(llvm::insert_value(
        context.context(),
        context.entry().argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[0]),
        ptr,
        context.location(),
    ));
    let op13 = context.entry().append_operation(llvm::insert_value(
        context.context(),
        op12.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context.context(), &[1]),
        op11.result(0).unwrap().into(),
        context.location(),
    ));

    context
        .entry()
        .append_operation(context.br(0, &[op13.result(0).unwrap().into()]));

    Ok(())
}
