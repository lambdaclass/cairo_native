use super::{LibfuncBuilder, LibfuncHelper};
use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        enm::{EnumConcreteLibfunc, EnumInitConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
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
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Identifier, Location,
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
        EnumConcreteLibfunc::Init(info) => {
            build_init(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::Match(info) => {
            build_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::SnapshotMatch(_) => todo!(),
    }
}

pub fn build_init<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &EnumInitConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let (tag_ty, variant_tys, align) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        registry
            .get_type(&info.branch_signatures()[0].vars[0].ty)
            .unwrap()
            .variants()
            .unwrap(),
    )
    .unwrap();

    let enum_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let padding_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        (crate::ffi::get_size(helper, &enum_ty)
            - crate::ffi::get_size(
                helper,
                &llvm::r#type::r#struct(context, &[tag_ty, variant_tys[info.index]], false),
            ))
        .try_into()
        .unwrap(),
    );

    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(info.index.try_into().unwrap(), tag_ty).into(),
        location,
    ));

    let concrete_enum_ty = llvm::r#type::r#struct(
        context,
        &[tag_ty, variant_tys[info.index], padding_ty],
        false,
    );

    let op1 = entry.append_operation(llvm::undef(concrete_enum_ty, location));
    let op2 = entry.append_operation(llvm::insert_value(
        context,
        op1.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        op0.result(0).unwrap().into(),
        location,
    ));
    let op3 = entry.append_operation(llvm::insert_value(
        context,
        op2.result(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[1]),
        entry.argument(0).unwrap().into(),
        location,
    ));

    let op4 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let op5 = entry.append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    align.try_into().unwrap(),
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[op4.result(0).unwrap().into()])
            .add_results(&[llvm::r#type::pointer(concrete_enum_ty, 0)])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        op3.result(0).unwrap().into(),
        op5.result(0).unwrap().into(),
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            align.try_into().unwrap(),
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let op6 = entry.append_operation(
        OperationBuilder::new("llvm.bitcast", location)
            .add_operands(&[op5.result(0).unwrap().into()])
            .add_results(&[llvm::r#type::pointer(enum_ty, 0)])
            .build(),
    );
    let op7 = entry.append_operation(llvm::load(
        context,
        op6.result(0).unwrap().into(),
        enum_ty,
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            align.try_into().unwrap(),
            IntegerType::new(context, 64).into(),
        ))),
    ));

    entry.append_operation(helper.br(0, &[op7.result(0).unwrap().into()], location));

    Ok(())
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
    let (tag_ty, variant_tys, align) = crate::types::r#enum::get_type_for_variants(
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

    let enum_ty = registry
        .get_type(&info.param_signatures()[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(1, IntegerType::new(context, 64).into()).into(),
        location,
    ));
    let op1 = entry.append_operation(
        OperationBuilder::new("llvm.alloca", location)
            .add_attributes(&[(
                Identifier::new(context, "alignment"),
                IntegerAttribute::new(
                    align.try_into().unwrap(),
                    IntegerType::new(context, 64).into(),
                )
                .into(),
            )])
            .add_operands(&[op0.result(0).unwrap().into()])
            .add_results(&[llvm::r#type::pointer(enum_ty, 0)])
            .build(),
    );
    entry.append_operation(llvm::store(
        context,
        entry.argument(0).unwrap().into(),
        op1.result(0).unwrap().into(),
        location,
        LoadStoreOptions::default().align(Some(IntegerAttribute::new(
            align.try_into().unwrap(),
            IntegerType::new(context, 64).into(),
        ))),
    ));

    let default_block = helper.append_block(&[]);
    let variant_blocks = variant_tys
        .iter()
        .map(|_| helper.append_block(&[]))
        .collect::<Vec<_>>();

    let op2 = entry.append_operation(llvm::extract_value(
        context,
        entry.argument(0).unwrap().into(),
        DenseI64ArrayAttribute::new(context, &[0]),
        tag_ty,
        location,
    ));
    entry.append_operation(
        cf::switch(
            context,
            &(0..variant_tys.len())
                .map(i64::try_from)
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            op2.result(0).unwrap().into(),
            tag_ty,
            (default_block, &[]),
            &variant_blocks
                .iter()
                .copied()
                .map(|block| (block, [].as_slice()))
                .collect::<Vec<_>>(),
            location,
        )
        .unwrap(),
    );

    {
        let op3 = default_block.append_operation(arith::constant(
            context,
            IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
            location,
        ));

        default_block.append_operation(cf::assert(
            context,
            op3.result(0).unwrap().into(),
            "Invalid enum tag.",
            location,
        ));
        default_block.append_operation(OperationBuilder::new("llvm.unreachable", location).build());
    }

    for (i, (block, payload_ty)) in variant_blocks.into_iter().zip(variant_tys).enumerate() {
        let padding_ty = llvm::r#type::array(
            IntegerType::new(context, 8).into(),
            (crate::ffi::get_size(helper, &enum_ty)
                - crate::ffi::get_size(
                    helper,
                    &llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false),
                ))
            .try_into()
            .unwrap(),
        );
        let concrete_enum_ty =
            llvm::r#type::r#struct(context, &[tag_ty, payload_ty, padding_ty], false);

        let op3 = block.append_operation(
            OperationBuilder::new("llvm.bitcast", location)
                .add_operands(&[op1.result(0).unwrap().into()])
                .add_results(&[llvm::r#type::pointer(concrete_enum_ty, 0)])
                .build(),
        );
        let op4 = block.append_operation(llvm::load(
            context,
            op3.result(0).unwrap().into(),
            concrete_enum_ty,
            location,
            LoadStoreOptions::default(),
        ));
        let op5 = block.append_operation(llvm::extract_value(
            context,
            op4.result(0).unwrap().into(),
            DenseI64ArrayAttribute::new(context, &[1]),
            payload_ty,
            location,
        ));

        block.append_operation(helper.br(i, &[op5.result(0).unwrap().into()], location));
    }

    Ok(())
}
