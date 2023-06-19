use super::{TypeBuilder, TypeBuilderContext};
use cairo_lang_sierra::extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Type},
};

pub fn build<'ctx, TType, TLibfunc>(
    context: TypeBuilderContext<'ctx, '_, TType, TLibfunc>,
    info: &InfoAndTypeConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let item_ty = context
        .registry
        .get_type(&info.ty)
        .unwrap()
        .build(context)
        .unwrap();

    Ok(llvm::r#type::r#struct(
        context.context(),
        &[
            item_ty,
            IntegerType::new(context.context, 32).into(),
            IntegerType::new(context.context, 32).into(),
        ],
        false,
    ))
}
