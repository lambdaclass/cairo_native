use super::TypeBuilder;
use crate::{metadata::MetadataStorage, utils::get_integer_layout};
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
use std::alloc::Layout;

pub type TypeLayout<'ctx> = (Type<'ctx>, Layout);

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
    let (_, (tag_ty, tag_layout), variant_tys) =
        get_type_for_variants(context, module, registry, metadata, &info.variants).unwrap();

    let (variant_ty, variant_layout) = variant_tys
        .iter()
        .copied()
        .max_by_key(|(_, layout)| layout.align())
        .unwrap_or((
            llvm::r#type::r#struct(context, &[], false),
            Layout::from_size_align(0, 1).unwrap(),
        ));

    let total_len = variant_tys
        .iter()
        .map(|(_, layout)| tag_layout.extend(*layout).unwrap().0.size())
        .max()
        .unwrap_or(0);
    let padding_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        (total_len - tag_layout.extend(variant_layout).unwrap().0.size())
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
) -> Result<(Layout, TypeLayout<'ctx>, Vec<TypeLayout<'ctx>>), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
{
    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
    let tag_layout = get_integer_layout(tag_bits);
    let tag_ty: Type = IntegerType::new(context, tag_bits).into();

    let mut layout = tag_layout;
    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let concrete_payload_ty = registry.get_type(variant).unwrap();

        let payload_ty = concrete_payload_ty
            .build(context, module, registry, metadata)
            .unwrap();
        let payload_layout = concrete_payload_ty.layout(registry);

        let full_layout = tag_layout.extend(payload_layout).unwrap().0;
        layout = Layout::from_size_align(
            layout.size().max(full_layout.size()),
            layout.align().max(full_layout.align()),
        )
        .unwrap();

        output.push((payload_ty, payload_layout));
    }

    Ok((layout, (tag_ty, tag_layout), output))
}
