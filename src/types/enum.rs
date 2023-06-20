use super::TypeBuilder;
use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::{enm::EnumConcreteType, GenericLibfunc, GenericType},
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
    let mut payload_align = 0;
    let mut payload_size = 0;

    for variant in &info.variants {
        let variant_ty = registry
            .get_type(variant)
            .unwrap()
            .build(context, module, registry, metadata)
            .unwrap();

        payload_align = payload_align.max(crate::ffi::get_abi_alignment(module, &variant_ty));
        payload_size = payload_size.max(crate::ffi::get_abi_alignment(module, &variant_ty));
    }

    let tag_ty = IntegerType::new(
        context,
        info.variants.len().next_power_of_two().trailing_zeros(),
    )
    .into();
    // TODO: Align the payload buffer to `payload_align`.
    let payload_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        payload_size.try_into().unwrap(),
    );

    Ok(llvm::r#type::r#struct(
        context,
        &[tag_ty, payload_ty],
        false,
    ))
}
