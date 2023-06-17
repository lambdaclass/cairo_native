use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    lib_func::SignatureOnlyConcreteLibfunc, structure::StructConcreteLibfunc, GenericLibfunc,
    GenericType,
};
use melior::{dialect::llvm, ir::attribute::DenseI64ArrayAttribute};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &StructConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        StructConcreteLibfunc::Construct(info) => build_construct(context, info),
        StructConcreteLibfunc::Deconstruct(_) => todo!(),
        StructConcreteLibfunc::SnapshotDeconstruct(_) => todo!(),
    }
}

pub fn build_construct<TType, TLibfunc>(
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

    let mut acc = context
        .entry()
        .append_operation(llvm::undef(struct_ty, context.location()));
    for i in 0..info.signature.param_signatures.len() {
        acc = context.entry().append_operation(llvm::insert_value(
            context.context(),
            acc.result(0).unwrap().into(),
            DenseI64ArrayAttribute::new(context.context(), &[i as _]),
            context.entry().argument(i).unwrap().into(),
            context.location(),
        ));
    }

    context
        .entry()
        .append_operation(context.br(0, &[acc.result(0).unwrap().into()]));

    Ok(())
}
