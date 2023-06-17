use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    debug::DebugConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc,
    GenericType,
};
use melior::ir::operation::OperationBuilder;

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &DebugConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        DebugConcreteLibfunc::Print(info) => build_print(context, info),
    }
}

pub fn build_print<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    // TODO: Implement.
    context
        .entry
        .append_operation(OperationBuilder::new("llvm.unreachable", context.location()).build());

    // context
    //     .entry()
    //     .append_operation(context.br(0, &[op0.result(0).unwrap().into()]));

    Ok(())
}
