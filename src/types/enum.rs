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

type EnumVariantTypes<'ctx> = (Type<'ctx>, Vec<(Type<'ctx>, Type<'ctx>)>, usize);

pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &EnumConcreteType,
) -> Result<Type<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let (tag_ty, variant_tys, _) =
        get_type_for_variants(context, module, registry, metadata, &info.variants).unwrap();

    let tag_size = crate::ffi::get_size(module, &tag_ty);
    let payload_size = variant_tys
        .iter()
        .map(|(variant_ty, _)| crate::ffi::get_size(module, variant_ty))
        .max()
        .unwrap_or_default();

    Ok(llvm::r#type::pointer(
        llvm::r#type::r#struct(
            context,
            &[
                tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (payload_size - tag_size).try_into().unwrap(),
                ),
            ],
            false,
        ),
        0,
    ))
}

pub fn get_type_for_variants<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    variants: &[ConcreteTypeId],
) -> Result<EnumVariantTypes<'ctx>, std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let tag_ty: Type =
        IntegerType::new(context, variants.len().next_power_of_two().trailing_zeros()).into();

    let mut align = 0;
    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let payload_ty = registry
            .get_type(variant)
            .unwrap()
            .build(context, module, registry, metadata)
            .unwrap();

        let variant_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);
        align = align.max(crate::ffi::get_abi_alignment(module, &variant_ty));

        output.push((variant_ty, payload_ty));
    }

    Ok((tag_ty, output, align))
}
