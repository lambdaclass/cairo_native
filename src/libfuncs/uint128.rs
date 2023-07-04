//! # `u128`-related libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::mlir_asm,
};
use cairo_lang_sierra::{
    extensions::{
        int::{
            unsigned::{UintConstConcreteLibfunc, UintOperationConcreteLibfunc},
            unsigned128::{Uint128Concrete, Uint128Traits},
            IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{
        attribute::DenseI64ArrayAttribute, operation::OperationBuilder, r#type::IntegerType,
        Attribute, Block, Location, Value, ValueLike,
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
    selector: &Uint128Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        Uint128Concrete::ByteReverse(info) => {
            build_byte_reverse(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Equal(_) => todo!(),
        Uint128Concrete::FromFelt252(_) => todo!(),
        Uint128Concrete::GuaranteeMul(_) => todo!(),
        Uint128Concrete::IsZero(_) => todo!(),
        Uint128Concrete::MulGuaranteeVerify(_) => todo!(),
        Uint128Concrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::SquareRoot(_) => todo!(),
        Uint128Concrete::ToFelt252(_) => todo!(),
    }
}

pub fn build_byte_reverse<'ctx, 'this, TType, TLibfunc>(
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
    let u128_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let arg0 = entry.argument(0)?.into();
    mlir_asm! { context, entry, location =>
        ; res = "llvm.intr.bswap"(arg0) : (u128_ty) -> u128_ty
    };

    entry.append_operation(helper.br(0, &[res], location));
    Ok(())
}

pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &UintConstConcreteLibfunc<Uint128Traits>,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = info.c;

    let u128_ty = registry
        .get_type(&info.branch_signatures()[0].vars[0].ty)?
        .build(context, helper, registry, metadata)?;

    let attr_c = Attribute::parse(context, &format!("{value} : {u128_ty}")).unwrap();

    mlir_asm! { context, entry, location =>
        ; k0 = "arith.constant"() { "value" = attr_c } : () -> u128_ty
    }

    entry.append_operation(helper.br(0, &[k0], location));
    Ok(())
}

pub fn build_divmod<'ctx, 'this, TType, TLibfunc>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let lhs: Value = entry.argument(0)?.into();
    let rhs: Value = entry.argument(1)?.into();

    let op = entry.append_operation(arith::divui(lhs, rhs, location));
    let result_div = op.result(0)?.into();

    let op = entry.append_operation(arith::remui(lhs, rhs, location));
    let result_rem = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[result_div, result_rem], location));
    Ok(())
}

pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &UintOperationConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op_name = match info.operator {
        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
    };

    let values_type = lhs.r#type();

    let result_type = llvm::r#type::r#struct(
        context,
        &[values_type, IntegerType::new(context, 1).into()],
        false,
    );

    let result: Value = entry
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[lhs, rhs])
                .add_results(&[result_type])
                .build(),
        )
        .result(0)?
        .into();

    let result = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[0]),
            values_type,
            location,
        ))
        .result(0)?
        .into();
    let overflow = entry
        .append_operation(llvm::extract_value(
            context,
            result,
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        overflow,
        [1, 0],
        [&[range_check, result], &[range_check, result]],
        location,
    ));
    Ok(())
}
