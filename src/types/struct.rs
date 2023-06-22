use super::TypeBuilder;
use crate::metadata::MetadataStorage;
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Module, Type},
    Context,
};

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
    let fields: Vec<_> = info
        .members
        .iter()
        .map(|field| {
            registry
                .get_type(field)
                .unwrap()
                .build(context, module, registry, metadata)
                .unwrap()
        })
        .collect();
    let struct_ty = llvm::r#type::r#struct(context, &fields, false);

    Ok(struct_ty)
}
