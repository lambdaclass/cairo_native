//! # `felt252`-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    types::{felt252::Felt252, TypeBuilder},
};
use cairo_lang_sierra::{
    extensions::{
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        scf,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, Region, Type,
    },
    Context,
};
use num_bigint::{Sign, ToBigInt};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Felt252Concrete::BinaryOperation(info) => {
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the following libfuncs:
///   - `felt252_add` and `felt252_add_const`.
///   - `felt252_sub` and `felt252_sub_const`.
///   - `felt252_mul` and `felt252_mul_const`.
///   - `felt252_div` and `felt252_div_const`.
pub fn build_binary_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252BinaryOperationConcrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let felt252_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let prime = metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime();

    let result = match info {
        Felt252BinaryOperationConcrete::WithVar(info) => match info.operator {
            Felt252BinaryOperator::Add => {
                let op0 = entry.append_operation(arith::addi(
                    entry.argument(0)?.into(),
                    entry.argument(1)?.into(),
                    location,
                ));

                let op1 = entry.append_operation(arith::constant(
                    context,
                    Attribute::parse(context, &format!("{prime} : {felt252_ty}")).unwrap(),
                    location,
                ));
                let op2 = entry.append_operation(arith::cmpi(
                    context,
                    CmpiPredicate::Uge,
                    op0.result(0)?.into(),
                    op1.result(0)?.into(),
                    location,
                ));
                let op3 = entry.append_operation(scf::r#if(
                    op2.result(0)?.into(),
                    &[felt252_ty],
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        let op3 = block.append_operation(arith::subi(
                            op0.result(0)?.into(),
                            op1.result(0)?.into(),
                            location,
                        ));

                        block.append_operation(scf::r#yield(&[op3.result(0)?.into()], location));

                        region
                    },
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        block.append_operation(scf::r#yield(&[op0.result(0)?.into()], location));

                        region
                    },
                    location,
                ));

                op3.result(0)?.into()
            }
            Felt252BinaryOperator::Sub => {
                let op0 = entry.append_operation(arith::subi(
                    entry.argument(0)?.into(),
                    entry.argument(1)?.into(),
                    location,
                ));

                let op1 = entry.append_operation(arith::cmpi(
                    context,
                    CmpiPredicate::Ult,
                    entry.argument(0)?.into(),
                    entry.argument(1)?.into(),
                    location,
                ));
                let op2 = entry.append_operation(scf::r#if(
                    op1.result(0)?.into(),
                    &[felt252_ty],
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        let op2 = block.append_operation(arith::constant(
                            context,
                            Attribute::parse(context, &format!("{prime} : {felt252_ty}")).unwrap(),
                            location,
                        ));
                        let op3 = block.append_operation(arith::addi(
                            op0.result(0)?.into(),
                            op2.result(0)?.into(),
                            location,
                        ));

                        block.append_operation(scf::r#yield(&[op3.result(0)?.into()], location));

                        region
                    },
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        block.append_operation(scf::r#yield(&[op0.result(0)?.into()], location));

                        region
                    },
                    location,
                ));

                op2.result(0)?.into()
            }
            Felt252BinaryOperator::Mul => {
                let double_felt252_ty: Type = IntegerType::new(context, 504).into();

                let op0 = entry.append_operation(arith::extui(
                    entry.argument(0)?.into(),
                    double_felt252_ty,
                    location,
                ));
                let op1 = entry.append_operation(arith::extui(
                    entry.argument(1)?.into(),
                    double_felt252_ty,
                    location,
                ));

                let op2 = entry.append_operation(arith::muli(
                    op0.result(0)?.into(),
                    op1.result(0)?.into(),
                    location,
                ));
                let op3 = entry.append_operation(arith::constant(
                    context,
                    Attribute::parse(context, &format!("{prime} : i504")).unwrap(),
                    location,
                ));
                let op4 = entry.append_operation(arith::remui(
                    op2.result(0)?.into(),
                    op3.result(0)?.into(),
                    location,
                ));
                let op5 = entry.append_operation(arith::trunci(
                    op4.result(0)?.into(),
                    felt252_ty,
                    location,
                ));

                op5.result(0)?.into()
            }
            Felt252BinaryOperator::Div => todo!(),
        },
        Felt252BinaryOperationConcrete::WithConst(_) => todo!(),
    };

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `felt252_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = match info.c.sign() {
        Sign::Minus => {
            let prime = metadata.get::<PrimeModuloMeta<Felt252>>().unwrap().prime();
            (&info.c + prime.to_bigint().unwrap()).to_biguint().unwrap()
        }
        _ => info.c.to_biguint().unwrap(),
    };
    let felt252_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {felt252_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `felt252_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let felt252_ty = registry
        .get_type(&info.param_signatures()[0].ty)?
        .build(context, helper, registry, metadata)?;

    let op0 = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, felt252_ty).into(),
        location,
    ));
    let op1 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        entry.argument(0)?.into(),
        op0.result(0)?.into(),
        location,
    ));

    entry.append_operation(helper.cond_br(
        op1.result(0)?.into(),
        [0, 1],
        [&[], &[entry.argument(0)?.into()]],
        location,
    ));

    Ok(())
}
