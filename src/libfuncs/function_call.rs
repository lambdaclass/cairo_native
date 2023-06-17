use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    function_call::FunctionCallConcreteLibfunc, GenericLibfunc, GenericType,
};
use melior::{dialect::func, ir::attribute::FlatSymbolRefAttribute};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    info: &FunctionCallConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let arguments = (0..context.entry().argument_count())
        .map(|i| context.entry().argument(i).unwrap().into())
        .collect::<Vec<_>>();
    let result_types = info.signature.branch_signatures[0]
        .vars
        .iter()
        .map(|x| {
            context
                .registry()
                .get_type(&x.ty)
                .unwrap()
                .build(*context)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let op0 = context.entry().append_operation(func::call(
        context.context(),
        FlatSymbolRefAttribute::new(context.context(), &info.function.id.to_string()),
        &arguments,
        &result_types,
        context.location(),
    ));

    context.entry().append_operation(
        context.br(
            0,
            &result_types
                .iter()
                .enumerate()
                .map(|(i, _)| op0.result(i).unwrap().into())
                .collect::<Vec<_>>(),
        ),
    );

    Ok(())
}
