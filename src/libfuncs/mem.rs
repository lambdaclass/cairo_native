use super::{LibfuncBuilder, LibfuncBuilderContext};
use crate::types::TypeBuilder;
use cairo_lang_sierra::extensions::{
    lib_func::SignatureAndTypeConcreteLibfunc, mem::MemConcreteLibfunc, GenericLibfunc, GenericType,
};

pub fn build<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    selector: &MemConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        MemConcreteLibfunc::StoreTemp(info) => build_store_temp(context, info),
        MemConcreteLibfunc::StoreLocal(_) => todo!(),
        MemConcreteLibfunc::FinalizeLocals(_) => todo!(),
        MemConcreteLibfunc::AllocLocal(_) => todo!(),
        MemConcreteLibfunc::Rename(_) => todo!(),
    }
}

pub fn build_store_temp<TType, TLibfunc>(
    context: LibfuncBuilderContext<TType, TLibfunc>,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    context
        .entry()
        .append_operation(context.br(0, &[context.entry().argument(0).unwrap().into()]));

    Ok(())
}
