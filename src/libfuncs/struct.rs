use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        lib_func::SignatureOnlyConcreteLibfunc, structure::StructConcreteLibfunc, GenericLibfunc,
        GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{attribute::DenseI64ArrayAttribute, Block, Location},
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &StructConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        StructConcreteLibfunc::Construct(info) => {
            build_construct(context, registry, entry, location, helper, metadata, info)
        }
        StructConcreteLibfunc::Deconstruct(_) => todo!(),
        StructConcreteLibfunc::SnapshotDeconstruct(_) => todo!(),
    }
}

pub fn build_construct<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let struct_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(context, registry, metadata)
        .unwrap();

    let mut acc = entry.append_operation(llvm::undef(struct_ty, location));
    for i in 0..info.signature.param_signatures.len() {
        acc = entry.append_operation(llvm::insert_value(
            context,
            acc.result(0).unwrap().into(),
            DenseI64ArrayAttribute::new(context, &[i as _]),
            entry.argument(i).unwrap().into(),
            location,
        ));
    }

    entry.append_operation(helper.br(0, &[acc.result(0).unwrap().into()], location));

    Ok(())
}
