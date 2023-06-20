use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        enm::EnumConcreteLibfunc, lib_func::SignatureOnlyConcreteLibfunc, ConcreteLibfunc,
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith, cf,
        llvm::{self, LoadStoreOptions},
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location,
    },
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &EnumConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        EnumConcreteLibfunc::Init(_) => todo!(),
        EnumConcreteLibfunc::Match(info) => {
            build_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::SnapshotMatch(_) => todo!(),
    }
}

pub fn build_match<'ctx, 'this, TType, TLibfunc>(
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
    let (tag_ty, variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.param_signatures()[0].ty)
            .unwrap()
            .variants()
            .unwrap(),
    )
    .unwrap();

    let op0 = entry.append_operation(llvm::load(
        context,
        entry.argument(0).unwrap().into(),
        variant_tys[0].0,
        location,
        LoadStoreOptions::default(),
    ));
    let op1 = entry.append_operation(llvm::extract_value(
        context,
        op0.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));

    let default_block = helper.append_block(&[]);
    let variant_blocks = variant_tys
        .iter()
        .map(|_| helper.append_block(&[]))
        .collect::<Vec<_>>();

    entry.append_operation(
        cf::switch(
            context,
            &(0..variant_tys.len())
                .map(i64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            op1.result(0).unwrap().into(),
            tag_ty,
            (default_block, &[]),
            &variant_blocks
                .iter()
                .map(|block| (*block, [].as_slice()))
                .collect::<Vec<_>>(),
            location,
        )
        .unwrap(),
    );

    {
        let op2 = default_block.append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
            location,
        ));

        default_block.append_operation(cf::assert(
            context,
            op2.result(0).unwrap().into(),
            "Invalid enum tag.",
            location,
        ));
    }

    for (i, (block, (variant_ty, payload_ty))) in
        variant_blocks.into_iter().zip(variant_tys).enumerate()
    {
        let op2 = block.append_operation(llvm::load(
            context,
            entry.argument(0).unwrap().into(),
            variant_ty,
            location,
            LoadStoreOptions::default(),
        ));
        let op3 = block.append_operation(llvm::extract_value(
            context,
            op2.result(0).unwrap().into(),
            DenseI64ArrayAttribute::new(context, &[0]),
            payload_ty,
            location,
        ));

        block.append_operation(helper.br(i, &[op3.result(0).unwrap().into()], location));
    }

    Ok(())
}
