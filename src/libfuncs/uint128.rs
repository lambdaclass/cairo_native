//! # `u128`-related libfuncs

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, ErrorImpl, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{mlir_asm, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        int::{
            unsigned128::{Uint128Concrete, Uint128Traits},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm, scf,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute, StringAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Identifier, Location, Region, Value, ValueLike,
    },
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
        Uint128Concrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::GuaranteeMul(info) => {
            build_guarantee_mul(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::MulGuaranteeVerify(info) => {
            build_guarantee_verify(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Bitwise(info) => {
            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `u128_byte_reverse` libfunc.
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
    let u128_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let bswap_intrin_attr = StringAttribute::new(context, "llvm.bswap.i128").into();

    let arg1 = entry.argument(1)?.into();
    mlir_asm! { context, entry, location =>
        ; res = "llvm.call_intrinsic"(arg1) { "intrin" = bswap_intrin_attr } : (u128_ty) -> u128_ty
    };

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), res], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Uint128Traits>,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = info.c;

    let u128_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let attr_c = Attribute::parse(context, &format!("{value} : {u128_ty}"))
        .ok_or(ErrorImpl::ParseAttributeError)?;

    mlir_asm! { context, entry, location =>
        ; k0 = "arith.constant"() { "value" = attr_c } : () -> u128_ty
    }

    entry.append_operation(helper.br(0, &[k0], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_safe_divmod` libfunc.
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
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op = entry.append_operation(arith::divui(lhs, rhs, location));
    let result_div = op.result(0)?.into();

    let op = entry.append_operation(arith::remui(lhs, rhs, location));
    let result_rem = op.result(0)?.into();

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), result_div, result_rem],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u128_equal` libfunc.
pub fn build_equal<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let arg0: Value = entry.argument(0)?.into();
    let arg1: Value = entry.argument(1)?.into();

    let op0 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        arg1,
        location,
    ));

    entry.append_operation(helper.cond_br(
        context,
        op0.result(0)?.into(),
        [1, 0],
        [&[]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u128_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let arg1 = entry.argument(1)?.into();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, IntegerType::new(context, 252).into()).into(),
            location,
        ))
        .result(0)?
        .into();
    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(128, IntegerType::new(context, 252).into()).into(),
            location,
        ))
        .result(0)?
        .into();

    let min_wide_val = entry
        .append_operation(arith::shli(k1, k128, location))
        .result(0)?
        .into();
    let is_wide = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Uge,
            arg1,
            min_wide_val,
            location,
        ))
        .result(0)?
        .into();

    let lsb_bits = entry
        .append_operation(arith::trunci(
            arg1,
            IntegerType::new(context, 128).into(),
            location,
        ))
        .result(0)?
        .into();

    let msb_bits = entry
        .append_operation(arith::shrui(arg1, k128, location))
        .result(0)?
        .into();
    let msb_bits = entry
        .append_operation(arith::trunci(
            msb_bits,
            IntegerType::new(context, 128).into(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        is_wide,
        [1, 0],
        [
            &[entry.argument(0)?.into(), msb_bits, lsb_bits],
            &[entry.argument(0)?.into(), lsb_bits],
        ],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u128_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let arg0: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, arg0.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        const_0,
        location,
    ));
    let condition = op.result(0)?.into();

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_add` and `u128_sub` libfuncs.
pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &IntOperationConcreteLibfunc,
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

    let result_struct: Value = entry
        .append_operation(
            OperationBuilder::new(op_name, location)
                .add_operands(&[lhs, rhs])
                .add_results(&[result_type])
                .build()?,
        )
        .result(0)?
        .into();

    let result = entry
        .append_operation(llvm::extract_value(
            context,
            result_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            values_type,
            location,
        ))
        .result(0)?
        .into();
    let overflow = entry
        .append_operation(llvm::extract_value(
            context,
            result_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            IntegerType::new(context, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        overflow,
        [1, 0],
        [&[range_check, result], &[range_check, result]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u128_sqrt` libfunc.
pub fn build_square_root<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let i64_ty = IntegerType::new(context, 64).into();
    let i128_ty = IntegerType::new(context, 128).into();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(1, i128_ty).into(),
            location,
        ))
        .result(0)?
        .into();

    let is_small = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ule,
            entry.argument(1)?.into(),
            k1,
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(scf::r#if(
            is_small,
            &[i128_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));

                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k128 = entry
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(128, i128_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                let leading_zeros = block
                    .append_operation(
                        OperationBuilder::new("llvm.intr.ctlz", location)
                            .add_attributes(&[(
                                Identifier::new(context, "is_zero_poison"),
                                IntegerAttribute::new(1, IntegerType::new(context, 1).into())
                                    .into(),
                            )])
                            .add_operands(&[entry.argument(1)?.into()])
                            .add_results(&[i128_ty])
                            .build()?,
                    )
                    .result(0)?
                    .into();

                let num_bits = block
                    .append_operation(arith::subi(k128, leading_zeros, location))
                    .result(0)?
                    .into();

                let shift_amount = block
                    .append_operation(arith::addi(num_bits, k1, location))
                    .result(0)?
                    .into();

                let parity_mask = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(-2, i128_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let shift_amount = block
                    .append_operation(arith::andi(shift_amount, parity_mask, location))
                    .result(0)?
                    .into();

                let k0 = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(0, i128_ty).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let result = block
                    .append_operation(scf::r#while(
                        &[k0, shift_amount],
                        &[i128_ty, i128_ty],
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i128_ty, location),
                                (i128_ty, location),
                            ]));

                            let result = block
                                .append_operation(arith::shli(
                                    block.argument(0)?.into(),
                                    k1,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let large_candidate = block
                                .append_operation(arith::xori(result, k1, location))
                                .result(0)?
                                .into();

                            let large_candidate_squared = block
                                .append_operation(arith::muli(
                                    large_candidate,
                                    large_candidate,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let threshold = block
                                .append_operation(arith::shrui(
                                    entry.argument(1)?.into(),
                                    block.argument(1)?.into(),
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold_is_poison = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Eq,
                                    block.argument(1)?.into(),
                                    k128,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[threshold_is_poison, k0, threshold])
                                        .add_results(&[i128_ty])
                                        .build()?,
                                )
                                .result(0)?
                                .into();

                            let is_in_range = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Ule,
                                    large_candidate_squared,
                                    threshold,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let result = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[is_in_range, large_candidate, result])
                                        .add_results(&[i128_ty])
                                        .build()?,
                                )
                                .result(0)?
                                .into();

                            let k2 = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(2, i128_ty).into(),
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let shift_amount = block
                                .append_operation(arith::subi(
                                    block.argument(1)?.into(),
                                    k2,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let should_continue = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Sge,
                                    shift_amount,
                                    k0,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            block.append_operation(scf::condition(
                                should_continue,
                                &[result, shift_amount],
                                location,
                            ));

                            region
                        },
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i128_ty, location),
                                (i128_ty, location),
                            ]));

                            block.append_operation(scf::r#yield(
                                &[block.argument(0)?.into(), block.argument(1)?.into()],
                                location,
                            ));

                            region
                        },
                        location,
                    ))
                    .result(0)?
                    .into();

                block.append_operation(scf::r#yield(&[result], location));

                region
            },
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(arith::trunci(result, i64_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), result], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
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
    let op = entry.append_operation(arith::extui(
        entry.argument(0)?.into(),
        IntegerType::new(context, 252).into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_guarantee_mul` libfunc.
pub fn build_guarantee_mul<'ctx, 'this, TType, TLibfunc>(
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
    let lhs: Value = entry.argument(0)?.into();
    let rhs: Value = entry.argument(1)?.into();

    let origin_type = lhs.r#type();

    let target_type = IntegerType::new(context, 256).into();
    let guarantee_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][2],
    )?;

    let op = entry.append_operation(arith::extui(lhs, target_type, location));
    let lhs = op.result(0)?.into();

    let op = entry.append_operation(arith::extui(rhs, target_type, location));
    let rhs = op.result(0)?.into();

    let op = entry.append_operation(arith::muli(lhs, rhs, location));
    let result = op.result(0)?.into();

    let op = entry.append_operation(arith::trunci(result, origin_type, location));
    let result_lo = op.result(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(128, target_type).into(),
        location,
    ));
    let const_128 = op.result(0)?.into();

    let op = entry.append_operation(arith::shrui(result, const_128, location));
    let result_hi = op.result(0)?.into();
    let op = entry.append_operation(arith::trunci(result_hi, origin_type, location));
    let result_hi = op.result(0)?.into();

    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let guarantee = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[result_hi, result_lo, guarantee], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_guarantee_verify` libfunc.
pub fn build_guarantee_verify<'ctx, 'this, TType, TLibfunc>(
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
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
        values::JitValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::BigUint;

    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref U128_BYTE_REVERSE: (String, Program) = load_cairo! {
            extern fn u128_byte_reverse(input: u128) -> u128 implicits(Bitwise) nopanic;

            fn run_test(value: u128) -> u128 {
                u128_byte_reverse(value)
            }
        };
        static ref U128_CONST: (String, Program) = load_cairo! {
            fn run_test() -> u128 {
                1234567890
            }
        };
        static ref U128_SAFE_DIVMOD: (String, Program) = load_cairo! {
            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
                let q = lhs / rhs;
                let r = lhs % rhs;

                (q, r)
            }
        };
        static ref U128_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: u128, rhs: u128) -> bool {
                lhs == rhs
            }
        };
        static ref U128_FROM_FELT252: (String, Program) = load_cairo! {
            enum U128sFromFelt252Result {
                Narrow: u128,
                Wide: (u128, u128),
            }

            extern fn u128s_from_felt252(a: felt252) -> U128sFromFelt252Result implicits(RangeCheck) nopanic;

            fn run_test(value: felt252) -> U128sFromFelt252Result {
                u128s_from_felt252(value)
            }
        };
        static ref U128_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn u128_is_zero(a: u128) -> IsZeroResult<u128> implicits() nopanic;

            fn run_test(value: u128) -> bool {
                match u128_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref U128_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: u128, rhs: u128) -> u128 {
                lhs + rhs
            }
        };
        static ref U128_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: u128, rhs: u128) -> u128 {
                lhs - rhs
            }
        };
        static ref U128_WIDEMUL: (String, Program) = load_cairo! {
            use integer::u128_wide_mul;
            fn run_test(lhs: u128, rhs: u128) -> (u128, u128) {
                u128_wide_mul(lhs, rhs)
            }
        };
        static ref U128_TO_FELT252: (String, Program) = load_cairo! {
            extern fn u128_to_felt252(a: u128) -> felt252 nopanic;

            fn run_test(value: u128) -> felt252 {
                u128_to_felt252(value)
            }
        };
        static ref U128_SQRT: (String, Program) = load_cairo! {
            use core::integer::u128_sqrt;

            fn run_test(value: u128) -> u64 {
                u128_sqrt(value)
            }
        };
    }

    #[test]
    fn u128_byte_reverse() {
        run_program_assert_output(
            &U128_BYTE_REVERSE,
            "run_test",
            &[0x00000000_00000000_00000000_00000000u128.into()],
            0x00000000_00000000_00000000_00000000u128.into(),
        );
        run_program_assert_output(
            &U128_BYTE_REVERSE,
            "run_test",
            &[0x00000000_00000000_00000000_00000001u128.into()],
            0x01000000_00000000_00000000_00000000u128.into(),
        );
        run_program_assert_output(
            &U128_BYTE_REVERSE,
            "run_test",
            &[0x12345678_90ABCDEF_12345678_90ABCDEFu128.into()],
            0xEFCDAB90_78563412_EFCDAB90_78563412u128.into(),
        );
    }

    #[test]
    fn u128_const() {
        run_program_assert_output(&U128_CONST, "run_test", &[], 1234567890_u128.into());
    }

    #[test]
    fn u128_safe_divmod() {
        let program = &U128_SAFE_DIVMOD;
        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;
        let error = JitValue::Felt252(Felt::from_bytes_be_slice(b"u128 is 0"));

        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 0u128.into()],
            jit_panic!(error.clone()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 1u128.into()],
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), max_value.into()],
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 0u128.into()))),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 0u128.into()],
            jit_panic!(error.clone()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 1u128.into()],
            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), max_value.into()],
            jit_enum!(0, jit_struct!(jit_struct!(0u128.into(), 1u128.into()))),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[max_value.into(), 0u128.into()],
            jit_panic!(error),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[max_value.into(), 1u128.into()],
            jit_enum!(0, jit_struct!(jit_struct!(u128::MAX.into(), 0u128.into()))),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[max_value.into(), max_value.into()],
            jit_enum!(0, jit_struct!(jit_struct!(1u128.into(), 0u128.into()))),
        );
    }

    #[test]
    fn u128_equal() {
        let program = &U128_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 0u128.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 0u128.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 1u128.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 1u128.into()],
            jit_enum!(1, jit_struct!()),
        );
    }

    #[test]
    fn u128_from_felt252() {
        run_program_assert_output(
            &U128_FROM_FELT252,
            "run_test",
            &[Felt::from(0).into()],
            jit_enum!(0, 0u128.into()),
        );

        run_program_assert_output(
            &U128_FROM_FELT252,
            "run_test",
            &[Felt::from(1).into()],
            jit_enum!(0, 1u128.into()),
        );

        run_program_assert_output(
            &U128_FROM_FELT252,
            "run_test",
            &[Felt::from(u128::MAX).into()],
            jit_enum!(0, u128::MAX.into()),
        );

        run_program_assert_output(
            &U128_FROM_FELT252,
            "run_test",
            &[
                Felt::from_dec_str("340282366920938463463374607431768211456")
                    .unwrap()
                    .into(),
            ],
            jit_enum!(1, jit_struct!(1u128.into(), 0u128.into())),
        );
    }

    #[test]
    fn u128_is_zero() {
        run_program_assert_output(
            &U128_IS_ZERO,
            "run_test",
            &[0u128.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            &U128_IS_ZERO,
            "run_test",
            &[1u128.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn u128_add() {
        #[track_caller]
        fn run(lhs: u128, rhs: u128) {
            let program = &U128_ADD;
            let error = Felt::from_bytes_be_slice(b"u128_add Overflow");

            let add = lhs.checked_add(rhs);

            match add {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        jit_enum!(0, jit_struct!(result.into())),
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        jit_panic!(JitValue::Felt252(error)),
                    );
                }
            }
        }

        const MAX: u128 = u128::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn u128_sub() {
        #[track_caller]
        fn run(lhs: u128, rhs: u128) {
            let program = &U128_SUB;
            let error = Felt::from_bytes_be_slice(b"u128_sub Overflow");

            let res = lhs.checked_sub(rhs);

            match res {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        jit_enum!(0, jit_struct!(result.into())),
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        jit_panic!(JitValue::Felt252(error)),
                    );
                }
            }
        }

        const MAX: u128 = u128::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn u128_to_felt252() {
        let program = &U128_TO_FELT252;

        run_program_assert_output(program, "run_test", &[0u128.into()], Felt::from(0).into());
        run_program_assert_output(program, "run_test", &[1u128.into()], Felt::from(1).into());
        run_program_assert_output(
            program,
            "run_test",
            &[u128::MAX.into()],
            Felt::from(u128::MAX).into(),
        );
    }

    #[test]
    fn u128_sqrt() {
        let program = &U128_SQRT;

        run_program_assert_output(program, "run_test", &[0u128.into()], 0u64.into());
        run_program_assert_output(program, "run_test", &[u128::MAX.into()], u64::MAX.into());

        for i in 0..u128::BITS {
            let x = 1u128 << i;
            let y: u64 = BigUint::from(x)
                .sqrt()
                .try_into()
                .expect("should always fit into a u128");

            run_program_assert_output(program, "run_test", &[x.into()], y.into());
        }
    }

    #[test]
    fn u128_widemul() {
        let program = &U128_WIDEMUL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 0u128.into()],
            jit_struct!(0u128.into(), 0u128.into()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u128.into(), 1u128.into()],
            jit_struct!(0u128.into(), 0u128.into()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 0u128.into()],
            jit_struct!(0u128.into(), 0u128.into()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u128.into(), 1u128.into()],
            jit_struct!(0u128.into(), 1u128.into()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[u128::MAX.into(), u128::MAX.into()],
            jit_struct!((u128::MAX - 1).into(), 1u128.into()),
        );
    }
}
