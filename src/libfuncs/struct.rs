//! # Struct-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        lib_func::SignatureOnlyConcreteLibfunc, structure::StructConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{attribute::DenseI64ArrayAttribute, Block, Location, Value},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
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
        StructConcreteLibfunc::Deconstruct(info) => {
            build_deconstruct(context, registry, entry, location, helper, metadata, info)
        }
        StructConcreteLibfunc::SnapshotDeconstruct(_) => todo!(),
    }
}

/// Generate MLIR operations for the `struct_construct` libfunc.
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
        .get_type(&info.branch_signatures()[0].vars[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let mut acc = entry.append_operation(llvm::undef(struct_ty, location));
    for i in 0..info.param_signatures().len() {
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

/// Generate MLIR operations for the `struct_deconstruct` libfunc.
pub fn build_deconstruct<'ctx, 'this, TType, TLibfunc>(
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
        .get_type(&info.param_signatures()[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let mut fields = Vec::<Value>::with_capacity(info.branch_signatures()[0].vars.len());
    for i in 0..info.branch_signatures()[0].vars.len() {
        fields.push(
            entry
                .append_operation(llvm::extract_value(
                    context,
                    entry.argument(0).unwrap().into(),
                    DenseI64ArrayAttribute::new(context, &[i.try_into().unwrap()]),
                    crate::ffi::get_struct_field_type_at(&struct_ty, i),
                    location,
                ))
                .result(0)
                .unwrap()
                .into(),
        );
    }

    entry.append_operation(helper.br(0, &fields, location));

    Ok(())
}
