use super::{TypeBuilder, TypeBuilderContext};
use cairo_lang_sierra::extensions::{types::InfoOnlyConcreteType, GenericLibfunc, GenericType};
use melior::ir::{r#type::IntegerType, Type};

pub fn build<'ctx, TType, TLibfunc>(
    context: TypeBuilderContext<'ctx, '_, TType, TLibfunc>,
    _info: &InfoOnlyConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    Ok(IntegerType::new(context.context(), 32).into())
}
