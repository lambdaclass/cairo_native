use super::TypeBuilder;
use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};
use std::{borrow::Cow, iter::once};

pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
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
                            .build(context, module, registry, metadata)
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
