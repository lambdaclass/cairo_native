use super::TypeBuilder;
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{ir::Type, Context};
use std::{borrow::Cow, iter::once};

pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    info: &StructConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let type_asm = once(Cow::Borrowed("!llvm.struct<("))
        .chain(
            info.members
                .iter()
                .map(|x| {
                    Cow::Owned(
                        registry
                            .get_type(x)
                            .unwrap()
                            .build(context, registry)
                            .unwrap()
                            .to_string(),
                    )
                })
                .intersperse(Cow::Borrowed(", ")),
        )
        .chain(once(Cow::Borrowed(")>")))
        .collect::<String>();

    Ok(Type::parse(context, &type_asm).unwrap())
}
