use super::{LibfuncBuilder, LibfuncHelper};
use crate::types::TypeBuilder;
use cairo_lang_sierra::{
    extensions::{function_call::FunctionCallConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::func,
    ir::{attribute::FlatSymbolRefAttribute, Block, Location},
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &FunctionCallConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let arguments = (0..entry.argument_count())
        .map(|i| entry.argument(i).unwrap().into())
        .collect::<Vec<_>>();
    let result_types = info.signature.branch_signatures[0]
        .vars
        .iter()
        .map(|x| {
            registry
                .get_type(&x.ty)
                .unwrap()
                .build(context, registry)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let op0 = entry.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, &info.function.id.to_string()),
        &arguments,
        &result_types,
        location,
    ));

    entry.append_operation(
        helper.br(
            0,
            &result_types
                .iter()
                .enumerate()
                .map(|(i, _)| op0.result(i).unwrap().into())
                .collect::<Vec<_>>(),
            location,
        ),
    );

    Ok(())
}
