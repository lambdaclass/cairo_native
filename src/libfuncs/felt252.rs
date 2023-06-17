use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    felt252::{Felt252Concrete, Felt252ConstConcreteLibfunc},
    GenericLibfunc, GenericType,
};
use melior::{dialect::arith, ir::Attribute};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &Felt252Concrete,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        Felt252Concrete::BinaryOperation(_) => todo!(),
        Felt252Concrete::Const(info) => build_const(context, info),
        Felt252Concrete::IsZero(_) => todo!(),
    }
}

pub fn build_const<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let value = &info.c;
    let value_ty = context
        .registry()
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(*context)
        .unwrap();

    let op0 = context.entry().append_operation(arith::constant(
        context.context(),
        Attribute::parse(context.context(), &format!("{value} : {value_ty}")).unwrap(),
        context.location(),
    ));
    context
        .entry()
        .append_operation(context.br(0, &[op0.result(0).unwrap().into()]));

    Ok(())
}
