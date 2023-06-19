use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    int::unsigned::{Uint32Concrete, Uint32Traits, UintConcrete, UintConstConcreteLibfunc},
    lib_func::SignatureOnlyConcreteLibfunc,
    GenericLibfunc, GenericType,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{Attribute, Value},
};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &Uint32Concrete,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        UintConcrete::Const(info) => build_const(context, info),
        UintConcrete::Operation(_) => todo!(),
        UintConcrete::SquareRoot(_) => todo!(),
        UintConcrete::Equal(info) => build_equal(context, info),
        UintConcrete::ToFelt252(_) => todo!(),
        UintConcrete::FromFelt252(_) => todo!(),
        UintConcrete::IsZero(_) => todo!(),
        UintConcrete::Divmod(_) => todo!(),
        UintConcrete::WideMul(_) => todo!(),
    }
}

pub fn build_const<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    info: &UintConstConcreteLibfunc<Uint32Traits>,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let value = info.c;
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

pub fn build_equal<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let arg0: Value = context.entry().argument(0).unwrap().into();
    let arg1: Value = context.entry().argument(1).unwrap().into();

    let op0 = context.entry().append_operation(arith::cmpi(
        context.context(),
        CmpiPredicate::Eq,
        arg0,
        arg1,
        context.location(),
    ));

    context
        .entry()
        .append_operation(context.cond_br(op0.result(0).unwrap().into(), (0, 1), &[]));

    Ok(())
}
