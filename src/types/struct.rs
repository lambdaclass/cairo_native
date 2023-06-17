use super::{TypeBuilder, TypeBuilderContext};
use cairo_lang_sierra::extensions::{structure::StructConcreteType, GenericLibfunc, GenericType};
use melior::ir::Type;
use std::{borrow::Cow, iter::once};

pub fn build<'ctx, TType, TLibfunc>(
    context: TypeBuilderContext<'ctx, '_, TType, TLibfunc>,
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
                        context
                            .registry
                            .get_type(x)
                            .unwrap()
                            .build(context)
                            .unwrap()
                            .to_string(),
                    )
                })
                .intersperse(Cow::Borrowed(", ")),
        )
        .chain(once(Cow::Borrowed(")>")))
        .collect::<String>();

    Ok(Type::parse(context.context(), &type_asm).unwrap())
}
