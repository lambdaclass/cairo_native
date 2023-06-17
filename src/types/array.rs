use super::{TypeBuilder, TypeBuilderContext};
use cairo_lang_sierra::extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType};
use melior::ir::{r#type::MemRefType, Type};

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
    let array_ty = MemRefType::new(item_ty, &[i64::MIN as _], None, None);

    Ok(Type::parse(
        context.context(),
        &format!("!llvm.struct<({array_ty}, ui32)>"),
    )
    .unwrap())
}
