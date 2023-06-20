use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        felt252::{Felt252Concrete, Felt252ConstConcreteLibfunc},
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{Attribute, Block, Location},
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    selector: &Felt252Concrete,
    _metadata: &MetadataStorage,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        Felt252Concrete::BinaryOperation(_) => todo!(),
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, info)
        }
        Felt252Concrete::IsZero(_) => todo!(),
    }
}

pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let value = &info.c;
    let value_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(context, registry)
        .unwrap();

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0).unwrap().into()], location));

    Ok(())
}
