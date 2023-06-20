use super::TypeBuilder;
use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::{enm::EnumConcreteType, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: &EnumConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    Ok(llvm::r#type::opaque_pointer(context))
}

pub fn get_type_for_variants<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    variants: &[ConcreteTypeId],
) -> Result<(Type<'ctx>, Vec<(Type<'ctx>, Type<'ctx>)>), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let tag_ty: Type =
        IntegerType::new(context, variants.len().next_power_of_two().trailing_zeros()).into();

    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let payload_ty = registry
            .get_type(variant)
            .unwrap()
            .build(context, module, registry, metadata)
            .unwrap();

        output.push((
            llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false),
            payload_ty,
        ));
    }

    Ok((tag_ty, output))
}
