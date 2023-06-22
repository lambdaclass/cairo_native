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

    let variant_ty = variant_tys
        .iter()
        .copied()
        .max_by_key(|ty| crate::ffi::get_preferred_alignment(module, ty).min(8))
        .unwrap_or(llvm::r#type::r#struct(context, &[], false));

    let total_len = variant_tys
        .iter()
        .map(|ty| {
            crate::ffi::get_size(
                module,
                &llvm::r#type::r#struct(context, &[tag_ty, *ty], false),
            )
        })
        .max()
        .unwrap_or(0);
    let padding_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        (total_len
            - crate::ffi::get_size(
                module,
                &llvm::r#type::r#struct(context, &[tag_ty, variant_ty], false),
            ))
        .try_into()
        .unwrap(),
    );

    Ok(llvm::r#type::r#struct(
        context,
        &[tag_ty, variant_ty, padding_ty],
        false,
    ))
}

pub fn get_type_for_variants<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    variants: &[ConcreteTypeId],
) -> Result<(Type<'ctx>, Vec<Type<'ctx>>, usize), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let tag_ty: Type =
        IntegerType::new(context, variants.len().next_power_of_two().trailing_zeros()).into();

    let mut align = crate::ffi::get_preferred_alignment(module, &tag_ty).min(8);
    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let payload_ty = registry
            .get_type(variant)
            .unwrap()
            .build(context, module, registry, metadata)
            .unwrap();

        align = align.max(crate::ffi::get_preferred_alignment(module, &payload_ty).min(8));

        output.push(payload_ty);
    }

    Ok((tag_ty, output, align))
}
