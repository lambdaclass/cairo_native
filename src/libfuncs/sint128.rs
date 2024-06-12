////! # `i128`-related libfuncs
//! # `i128`-related libfuncs
//use super::LibfuncHelper;
use super::LibfuncHelper;
//use std::ops::Shr;
use std::ops::Shr;
//

//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        int::{
        int::{
//            signed128::{Sint128Concrete, Sint128Traits},
            signed128::{Sint128Concrete, Sint128Traits},
//            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
//        },
        },
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith::{self, CmpiPredicate},
        arith::{self, CmpiPredicate},
//        cf, llvm,
        cf, llvm,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
//        operation::OperationBuilder,
        operation::OperationBuilder,
//        r#type::IntegerType,
        r#type::IntegerType,
//        Attribute, Block, Location, Value, ValueLike,
        Attribute, Block, Location, Value, ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &Sint128Concrete,
    selector: &Sint128Concrete,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        Sint128Concrete::Const(info) => {
        Sint128Concrete::Const(info) => {
//            build_const(context, registry, entry, location, helper, metadata, info)
            build_const(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Sint128Concrete::Operation(info) => {
        Sint128Concrete::Operation(info) => {
//            build_operation(context, registry, entry, location, helper, info)
            build_operation(context, registry, entry, location, helper, info)
//        }
        }
//        Sint128Concrete::Equal(info) => {
        Sint128Concrete::Equal(info) => {
//            build_equal(context, registry, entry, location, helper, info)
            build_equal(context, registry, entry, location, helper, info)
//        }
        }
//        Sint128Concrete::ToFelt252(info) => {
        Sint128Concrete::ToFelt252(info) => {
//            build_to_felt252(context, registry, entry, location, helper, metadata, info)
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Sint128Concrete::FromFelt252(info) => {
        Sint128Concrete::FromFelt252(info) => {
//            build_from_felt252(context, registry, entry, location, helper, metadata, info)
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        Sint128Concrete::IsZero(info) => {
        Sint128Concrete::IsZero(info) => {
//            build_is_zero(context, registry, entry, location, helper, info)
            build_is_zero(context, registry, entry, location, helper, info)
//        }
        }
//        Sint128Concrete::Diff(info) => build_diff(context, registry, entry, location, helper, info),
        Sint128Concrete::Diff(info) => build_diff(context, registry, entry, location, helper, info),
//    }
    }
//}
}
//

///// Generate MLIR operations for the `i128_const` libfunc.
/// Generate MLIR operations for the `i128_const` libfunc.
//pub fn build_const<'ctx, 'this>(
pub fn build_const<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &IntConstConcreteLibfunc<Sint128Traits>,
    info: &IntConstConcreteLibfunc<Sint128Traits>,
//) -> Result<()> {
) -> Result<()> {
//    let value = info.c;
    let value = info.c;
//    let value_ty = registry.build_type(
    let value_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.signature.branch_signatures[0].vars[0].ty,
        &info.signature.branch_signatures[0].vars[0].ty,
//    )?;
    )?;
//

//    let value = entry.const_int_from_type(context, location, value, value_ty)?;
    let value = entry.const_int_from_type(context, location, value, value_ty)?;
//

//    entry.append_operation(helper.br(0, &[value], location));
    entry.append_operation(helper.br(0, &[value], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the i128 operation libfunc.
/// Generate MLIR operations for the i128 operation libfunc.
//pub fn build_operation<'ctx, 'this>(
pub fn build_operation<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    info: &IntOperationConcreteLibfunc,
    info: &IntOperationConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check: Value =
    let range_check: Value =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let lhs: Value = entry.argument(1)?.into();
    let lhs: Value = entry.argument(1)?.into();
//    let rhs: Value = entry.argument(2)?.into();
    let rhs: Value = entry.argument(2)?.into();
//

//    let op_name = match info.operator {
    let op_name = match info.operator {
//        IntOperator::OverflowingAdd => "llvm.intr.sadd.with.overflow",
        IntOperator::OverflowingAdd => "llvm.intr.sadd.with.overflow",
//        IntOperator::OverflowingSub => "llvm.intr.ssub.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.ssub.with.overflow",
//    };
    };
//

//    let values_type = lhs.r#type();
    let values_type = lhs.r#type();
//

//    let result_type = llvm::r#type::r#struct(
    let result_type = llvm::r#type::r#struct(
//        context,
        context,
//        &[values_type, IntegerType::new(context, 1).into()],
        &[values_type, IntegerType::new(context, 1).into()],
//        false,
        false,
//    );
    );
//

//    let result = entry
    let result = entry
//        .append_operation(
        .append_operation(
//            OperationBuilder::new(op_name, location)
            OperationBuilder::new(op_name, location)
//                .add_operands(&[lhs, rhs])
                .add_operands(&[lhs, rhs])
//                .add_results(&[result_type])
                .add_results(&[result_type])
//                .build()?,
                .build()?,
//        )
        )
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let op_result = entry
    let op_result = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[0]),
            DenseI64ArrayAttribute::new(context, &[0]),
//            values_type,
            values_type,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Create a const operation to get the 0 value to compare against
    // Create a const operation to get the 0 value to compare against
//    let zero_const = entry
    let zero_const = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(values_type, 0.into()).into(),
            IntegerAttribute::new(values_type, 0.into()).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//    // Check if the result is positive
    // Check if the result is positive
//    let is_positive = entry
    let is_positive = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Sge,
            CmpiPredicate::Sge,
//            op_result,
            op_result,
//            zero_const,
            zero_const,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    // Check overflow flag
    // Check overflow flag
//    let op_overflow = entry
    let op_overflow = entry
//        .append_operation(llvm::extract_value(
        .append_operation(llvm::extract_value(
//            context,
            context,
//            result,
            result,
//            DenseI64ArrayAttribute::new(context, &[1]),
            DenseI64ArrayAttribute::new(context, &[1]),
//            IntegerType::new(context, 1).into(),
            IntegerType::new(context, 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let block_not_overflow = helper.append_block(Block::new(&[]));
    let block_not_overflow = helper.append_block(Block::new(&[]));
//    let block_overflow = helper.append_block(Block::new(&[]));
    let block_overflow = helper.append_block(Block::new(&[]));
//

//    // The libfunc has three possible outputs: In Range, Overflow & Underflow
    // The libfunc has three possible outputs: In Range, Overflow & Underflow
//    entry.append_operation(cf::cond_br(
    entry.append_operation(cf::cond_br(
//        context,
        context,
//        op_overflow,
        op_overflow,
//        block_overflow,
        block_overflow,
//        block_not_overflow,
        block_not_overflow,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//    // Check wether the result is positive to distinguish between undeflowing & overflowing results
    // Check wether the result is positive to distinguish between undeflowing & overflowing results
//    block_overflow.append_operation(helper.cond_br(
    block_overflow.append_operation(helper.cond_br(
//        context,
        context,
//        is_positive,
        is_positive,
//        [1, 2],
        [1, 2],
//        [&[range_check, op_result], &[range_check, op_result]],
        [&[range_check, op_result], &[range_check, op_result]],
//        location,
        location,
//    ));
    ));
//    // No Oveflow/Underflow -> In range result
    // No Oveflow/Underflow -> In range result
//    block_not_overflow.append_operation(helper.br(0, &[range_check, op_result], location));
    block_not_overflow.append_operation(helper.br(0, &[range_check, op_result], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `i128_eq` libfunc.
/// Generate MLIR operations for the `i128_eq` libfunc.
//pub fn build_equal<'ctx, 'this>(
pub fn build_equal<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let arg0: Value = entry.argument(0)?.into();
    let arg0: Value = entry.argument(0)?.into();
//    let arg1: Value = entry.argument(1)?.into();
    let arg1: Value = entry.argument(1)?.into();
//

//    let op0 = entry.append_operation(arith::cmpi(
    let op0 = entry.append_operation(arith::cmpi(
//        context,
        context,
//        CmpiPredicate::Eq,
        CmpiPredicate::Eq,
//        arg0,
        arg0,
//        arg1,
        arg1,
//        location,
        location,
//    ));
    ));
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        op0.result(0)?.into(),
        op0.result(0)?.into(),
//        [1, 0],
        [1, 0],
//        [&[]; 2],
        [&[]; 2],
//        location,
        location,
//    ));
    ));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `i128_is_zero` libfunc.
/// Generate MLIR operations for the `i128_is_zero` libfunc.
//pub fn build_is_zero<'ctx, 'this>(
pub fn build_is_zero<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let arg0: Value = entry.argument(0)?.into();
    let arg0: Value = entry.argument(0)?.into();
//

//    let op = entry.append_operation(arith::constant(
    let op = entry.append_operation(arith::constant(
//        context,
        context,
//        IntegerAttribute::new(arg0.r#type(), 0).into(),
        IntegerAttribute::new(arg0.r#type(), 0).into(),
//        location,
        location,
//    ));
    ));
//    let const_0 = op.result(0)?.into();
    let const_0 = op.result(0)?.into();
//

//    let condition = entry
    let condition = entry
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Eq,
            CmpiPredicate::Eq,
//            arg0,
            arg0,
//            const_0,
            const_0,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `i128_to_felt252` libfunc.
/// Generate MLIR operations for the `i128_to_felt252` libfunc.
//pub fn build_to_felt252<'ctx, 'this>(
pub fn build_to_felt252<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//    let value: Value = entry.argument(0)?.into();
    let value: Value = entry.argument(0)?.into();
//

//    let result = entry
    let result = entry
//        .append_operation(arith::extui(value, felt252_ty, location))
        .append_operation(arith::extui(value, felt252_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.br(0, &[result], location));
    entry.append_operation(helper.br(0, &[result], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `i128_from_felt252` libfunc.
/// Generate MLIR operations for the `i128_from_felt252` libfunc.
//pub fn build_from_felt252<'ctx, 'this>(
pub fn build_from_felt252<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check: Value =
    let range_check: Value =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let value: Value = entry.argument(1)?.into();
    let value: Value = entry.argument(1)?.into();
//

//    let felt252_ty = registry.build_type(
    let felt252_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[1].ty,
        &info.param_signatures()[1].ty,
//    )?;
    )?;
//    let result_ty = registry.build_type(
    let result_ty = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[1].ty,
        &info.branch_signatures()[0].vars[1].ty,
//    )?;
    )?;
//

//    let const_max = entry
    let const_max = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            Attribute::parse(context, &format!("{} : {}", i128::MAX, felt252_ty))
            Attribute::parse(context, &format!("{} : {}", i128::MAX, felt252_ty))
//                .ok_or(Error::ParseAttributeError)?,
                .ok_or(Error::ParseAttributeError)?,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let const_min = entry
    let const_min = entry
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            Attribute::parse(context, &format!("{} : {}", i128::MIN, felt252_ty))
            Attribute::parse(context, &format!("{} : {}", i128::MIN, felt252_ty))
//                .ok_or(Error::ParseAttributeError)?,
                .ok_or(Error::ParseAttributeError)?,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let mut block = entry;
    let mut block = entry;
//

//    // make unsigned felt into signed felt
    // make unsigned felt into signed felt
//    // felt > half prime = negative
    // felt > half prime = negative
//    let value = {
    let value = {
//        let attr_halfprime_i252 = Attribute::parse(
        let attr_halfprime_i252 = Attribute::parse(
//            context,
            context,
//            &format!(
            &format!(
//                "{} : {}",
                "{} : {}",
//                metadata
                metadata
//                    .get::<PrimeModuloMeta<Felt>>()
                    .get::<PrimeModuloMeta<Felt>>()
//                    .ok_or(Error::MissingMetadata)?
                    .ok_or(Error::MissingMetadata)?
//                    .prime()
                    .prime()
//                    .shr(1),
                    .shr(1),
//                felt252_ty
                felt252_ty
//            ),
            ),
//        )
        )
//        .ok_or(Error::ParseAttributeError)?;
        .ok_or(Error::ParseAttributeError)?;
//        let half_prime: melior::ir::Value = block
        let half_prime: melior::ir::Value = block
//            .append_operation(arith::constant(context, attr_halfprime_i252, location))
            .append_operation(arith::constant(context, attr_halfprime_i252, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let is_felt_neg = block
        let is_felt_neg = block
//            .append_operation(arith::cmpi(
            .append_operation(arith::cmpi(
//                context,
                context,
//                CmpiPredicate::Ugt,
                CmpiPredicate::Ugt,
//                value,
                value,
//                half_prime,
                half_prime,
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let is_neg_block = helper.append_block(Block::new(&[]));
        let is_neg_block = helper.append_block(Block::new(&[]));
//        let is_not_neg_block = helper.append_block(Block::new(&[]));
        let is_not_neg_block = helper.append_block(Block::new(&[]));
//        let final_block = helper.append_block(Block::new(&[(felt252_ty, location)]));
        let final_block = helper.append_block(Block::new(&[(felt252_ty, location)]));
//

//        block.append_operation(cf::cond_br(
        block.append_operation(cf::cond_br(
//            context,
            context,
//            is_felt_neg,
            is_felt_neg,
//            is_neg_block,
            is_neg_block,
//            is_not_neg_block,
            is_not_neg_block,
//            &[],
            &[],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        {
        {
//            let prime = is_neg_block
            let prime = is_neg_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    Attribute::parse(
                    Attribute::parse(
//                        context,
                        context,
//                        &format!(
                        &format!(
//                            "{} : {}",
                            "{} : {}",
//                            metadata
                            metadata
//                                .get::<PrimeModuloMeta<Felt>>()
                                .get::<PrimeModuloMeta<Felt>>()
//                                .ok_or(Error::MissingMetadata)?
                                .ok_or(Error::MissingMetadata)?
//                                .prime(),
                                .prime(),
//                            felt252_ty
                            felt252_ty
//                        ),
                        ),
//                    )
                    )
//                    .ok_or(Error::ParseAttributeError)?,
                    .ok_or(Error::ParseAttributeError)?,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let mut src_value_is_neg: melior::ir::Value = is_neg_block
            let mut src_value_is_neg: melior::ir::Value = is_neg_block
//                .append_operation(arith::subi(prime, value, location))
                .append_operation(arith::subi(prime, value, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            let kneg1 = is_neg_block
            let kneg1 = is_neg_block
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    Attribute::parse(context, &format!("-1 : {}", felt252_ty))
                    Attribute::parse(context, &format!("-1 : {}", felt252_ty))
//                        .ok_or(Error::ParseAttributeError)?,
                        .ok_or(Error::ParseAttributeError)?,
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            src_value_is_neg = is_neg_block
            src_value_is_neg = is_neg_block
//                .append_operation(arith::muli(src_value_is_neg, kneg1, location))
                .append_operation(arith::muli(src_value_is_neg, kneg1, location))
//                .result(0)?
                .result(0)?
//                .into();
                .into();
//

//            is_neg_block.append_operation(cf::br(final_block, &[src_value_is_neg], location));
            is_neg_block.append_operation(cf::br(final_block, &[src_value_is_neg], location));
//        }
        }
//

//        is_not_neg_block.append_operation(cf::br(final_block, &[value], location));
        is_not_neg_block.append_operation(cf::br(final_block, &[value], location));
//

//        block = final_block;
        block = final_block;
//

//        block.argument(0)?.into()
        block.argument(0)?.into()
//    };
    };
//

//    let is_smaller_eq = block
    let is_smaller_eq = block
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Sle,
            CmpiPredicate::Sle,
//            value,
            value,
//            const_max,
            const_max,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let is_bigger_eq = block
    let is_bigger_eq = block
//        .append_operation(arith::cmpi(
        .append_operation(arith::cmpi(
//            context,
            context,
//            CmpiPredicate::Sge,
            CmpiPredicate::Sge,
//            value,
            value,
//            const_min,
            const_min,
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let is_ok = block
    let is_ok = block
//        .append_operation(arith::andi(is_smaller_eq, is_bigger_eq, location))
        .append_operation(arith::andi(is_smaller_eq, is_bigger_eq, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let block_success = helper.append_block(Block::new(&[]));
    let block_success = helper.append_block(Block::new(&[]));
//    let block_failure = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));
//

//    block.append_operation(cf::cond_br(
    block.append_operation(cf::cond_br(
//        context,
        context,
//        is_ok,
        is_ok,
//        block_success,
        block_success,
//        block_failure,
        block_failure,
//        &[],
        &[],
//        &[],
        &[],
//        location,
        location,
//    ));
    ));
//

//    let value = block_success
    let value = block_success
//        .append_operation(arith::trunci(value, result_ty, location))
        .append_operation(arith::trunci(value, result_ty, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    block_success.append_operation(helper.br(0, &[range_check, value], location));
    block_success.append_operation(helper.br(0, &[range_check, value], location));
//    block_failure.append_operation(helper.br(1, &[range_check], location));
    block_failure.append_operation(helper.br(1, &[range_check], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `i128_diff` libfunc.
/// Generate MLIR operations for the `i128_diff` libfunc.
//pub fn build_diff<'ctx, 'this>(
pub fn build_diff<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check: Value =
    let range_check: Value =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let lhs: Value = entry.argument(1)?.into();
    let lhs: Value = entry.argument(1)?.into();
//    let rhs: Value = entry.argument(2)?.into();
    let rhs: Value = entry.argument(2)?.into();
//

//    // Check if lhs >= rhs
    // Check if lhs >= rhs
//    let is_ge = entry
    let is_ge = entry
//        .append_operation(arith::cmpi(context, CmpiPredicate::Sge, lhs, rhs, location))
        .append_operation(arith::cmpi(context, CmpiPredicate::Sge, lhs, rhs, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    let result = entry
    let result = entry
//        .append_operation(arith::subi(lhs, rhs, location))
        .append_operation(arith::subi(lhs, rhs, location))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    entry.append_operation(helper.cond_br(
    entry.append_operation(helper.cond_br(
//        context,
        context,
//        is_ge,
        is_ge,
//        [0, 1],
        [0, 1],
//        [&[range_check, result], &[range_check, result]],
        [&[range_check, result], &[range_check, result]],
//        location,
        location,
//    ));
    ));
//    Ok(())
    Ok(())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{
    use crate::{
//        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//

//    lazy_static! {
    lazy_static! {
//        static ref I128_OVERFLOWING_ADD: (String, Program) = load_cairo! {
        static ref I128_OVERFLOWING_ADD: (String, Program) = load_cairo! {
//            fn run_test(lhs: i128, rhs: i128) -> i128 {
            fn run_test(lhs: i128, rhs: i128) -> i128 {
//                lhs + rhs
                lhs + rhs
//            }
            }
//        };
        };
//        static ref I128_OVERFLOWING_SUB: (String, Program) = load_cairo! {
        static ref I128_OVERFLOWING_SUB: (String, Program) = load_cairo! {
//            fn run_test(lhs: i128, rhs: i128) -> i128 {
            fn run_test(lhs: i128, rhs: i128) -> i128 {
//                lhs - rhs
                lhs - rhs
//            }
            }
//        };
        };
//        static ref I128_EQUAL: (String, Program) = load_cairo! {
        static ref I128_EQUAL: (String, Program) = load_cairo! {
//            fn run_test(lhs: i128, rhs: i128) -> bool {
            fn run_test(lhs: i128, rhs: i128) -> bool {
//                lhs == rhs
                lhs == rhs
//            }
            }
//        };
        };
//        static ref I128_IS_ZERO: (String, Program) = load_cairo! {
        static ref I128_IS_ZERO: (String, Program) = load_cairo! {
//            use zeroable::IsZeroResult;
            use zeroable::IsZeroResult;
//

//            extern fn i128_is_zero(a: i128) -> IsZeroResult<i128> implicits() nopanic;
            extern fn i128_is_zero(a: i128) -> IsZeroResult<i128> implicits() nopanic;
//

//            fn run_test(value: i128) -> bool {
            fn run_test(value: i128) -> bool {
//                match i128_is_zero(value) {
                match i128_is_zero(value) {
//                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::Zero(_) => true,
//                    IsZeroResult::NonZero(_) => false,
                    IsZeroResult::NonZero(_) => false,
//                }
                }
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_const_min() {
    fn i128_const_min() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> i128 {
            fn run_test() -> i128 {
//                -170141183460469231731687303715884105728_i128
                -170141183460469231731687303715884105728_i128
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], i128::MIN.into());
        run_program_assert_output(&program, "run_test", &[], i128::MIN.into());
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_const_max() {
    fn i128_const_max() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            fn run_test() -> i128 {
            fn run_test() -> i128 {
//                170141183460469231731687303715884105727_i128
                170141183460469231731687303715884105727_i128
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], (i128::MAX).into());
        run_program_assert_output(&program, "run_test", &[], (i128::MAX).into());
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_to_felt252() {
    fn i128_to_felt252() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::Into;
            use traits::Into;
//

//            fn run_test() -> felt252 {
            fn run_test() -> felt252 {
//                2_i128.into()
                2_i128.into()
//            }
            }
//        );
        );
//

//        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_from_felt252() {
    fn i128_from_felt252() {
//        let program = load_cairo!(
        let program = load_cairo!(
//            use traits::TryInto;
            use traits::TryInto;
//

//            fn run_test() -> (Option<i128>, Option<i128>) {
            fn run_test() -> (Option<i128>, Option<i128>) {
//                (
                (
//                    170141183460469231731687303715884105727.try_into(),
                    170141183460469231731687303715884105727.try_into(),
//                    170141183460469231731687303715884105728.try_into(),
                    170141183460469231731687303715884105728.try_into(),
//                )
                )
//            }
            }
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            &program,
            &program,
//            "run_test",
            "run_test",
//            &[],
            &[],
//            jit_struct!(
            jit_struct!(
//                jit_enum!(0, 170141183460469231731687303715884105727i128.into()),
                jit_enum!(0, 170141183460469231731687303715884105727i128.into()),
//                jit_enum!(1, jit_struct!()),
                jit_enum!(1, jit_struct!()),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_overflowing_add() {
    fn i128_overflowing_add() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: i128, rhs: i128) {
        fn run(lhs: i128, rhs: i128) {
//            let program = &I128_OVERFLOWING_ADD;
            let program = &I128_OVERFLOWING_ADD;
//            let error = Felt::from_bytes_be_slice(b"i128_add Overflow");
            let error = Felt::from_bytes_be_slice(b"i128_add Overflow");
//

//            let add = lhs.checked_add(rhs);
            let add = lhs.checked_add(rhs);
//

//            match add {
            match add {
//                Some(result) => {
                Some(result) => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_enum!(0, jit_struct!(result.into())),
                        jit_enum!(0, jit_struct!(result.into())),
//                    );
                    );
//                }
                }
//                None => {
                None => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_panic!(JitValue::Felt252(error)),
                        jit_panic!(JitValue::Felt252(error)),
//                    );
                    );
//                }
                }
//            }
            }
//        }
        }
//

//        const MAX: i128 = i128::MAX;
        const MAX: i128 = i128::MAX;
//

//        run(0, 0);
        run(0, 0);
//        run(0, 1);
        run(0, 1);
//        run(0, MAX - 1);
        run(0, MAX - 1);
//        run(0, MAX);
        run(0, MAX);
//

//        run(1, 0);
        run(1, 0);
//        run(1, 1);
        run(1, 1);
//        run(1, MAX - 1);
        run(1, MAX - 1);
//        run(1, MAX);
        run(1, MAX);
//

//        run(MAX - 1, 0);
        run(MAX - 1, 0);
//        run(MAX - 1, 1);
        run(MAX - 1, 1);
//        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX - 1);
//        run(MAX - 1, MAX);
        run(MAX - 1, MAX);
//

//        run(MAX, 0);
        run(MAX, 0);
//        run(MAX, 1);
        run(MAX, 1);
//        run(MAX, MAX - 1);
        run(MAX, MAX - 1);
//        run(MAX, MAX);
        run(MAX, MAX);
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_overflowing_sub() {
    fn i128_overflowing_sub() {
//        #[track_caller]
        #[track_caller]
//        fn run(lhs: i128, rhs: i128) {
        fn run(lhs: i128, rhs: i128) {
//            let program = &I128_OVERFLOWING_SUB;
            let program = &I128_OVERFLOWING_SUB;
//            let error = Felt::from_bytes_be_slice(b"i128_sub Overflow");
            let error = Felt::from_bytes_be_slice(b"i128_sub Overflow");
//

//            let add = lhs.checked_sub(rhs);
            let add = lhs.checked_sub(rhs);
//

//            match add {
            match add {
//                Some(result) => {
                Some(result) => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_enum!(0, jit_struct!(result.into())),
                        jit_enum!(0, jit_struct!(result.into())),
//                    );
                    );
//                }
                }
//                None => {
                None => {
//                    run_program_assert_output(
                    run_program_assert_output(
//                        program,
                        program,
//                        "run_test",
                        "run_test",
//                        &[lhs.into(), rhs.into()],
                        &[lhs.into(), rhs.into()],
//                        jit_panic!(JitValue::Felt252(error)),
                        jit_panic!(JitValue::Felt252(error)),
//                    );
                    );
//                }
                }
//            }
            }
//        }
        }
//

//        const MAX: i128 = i128::MAX;
        const MAX: i128 = i128::MAX;
//

//        run(0, 0);
        run(0, 0);
//        run(0, 1);
        run(0, 1);
//        run(0, MAX - 1);
        run(0, MAX - 1);
//        run(0, MAX);
        run(0, MAX);
//

//        run(1, 0);
        run(1, 0);
//        run(1, 1);
        run(1, 1);
//        run(1, MAX - 1);
        run(1, MAX - 1);
//        run(1, MAX);
        run(1, MAX);
//

//        run(MAX - 1, 0);
        run(MAX - 1, 0);
//        run(MAX - 1, 1);
        run(MAX - 1, 1);
//        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX - 1);
//        run(MAX - 1, MAX);
        run(MAX - 1, MAX);
//

//        run(MAX, 0);
        run(MAX, 0);
//        run(MAX, 1);
        run(MAX, 1);
//        run(MAX, MAX - 1);
        run(MAX, MAX - 1);
//        run(MAX, MAX);
        run(MAX, MAX);
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_equal() {
    fn i128_equal() {
//        let program = &I128_EQUAL;
        let program = &I128_EQUAL;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into(), 0i128.into()],
            &[0i128.into(), 0i128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into(), 0i128.into()],
            &[1i128.into(), 0i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into(), 1i128.into()],
            &[0i128.into(), 1i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into(), 1i128.into()],
            &[1i128.into(), 1i128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_is_zero() {
    fn i128_is_zero() {
//        let program = &I128_IS_ZERO;
        let program = &I128_IS_ZERO;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into()],
            &[0i128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into()],
            &[1i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn i128_safe_divmod() {
    fn i128_safe_divmod() {
//        let program = &I128_IS_ZERO;
        let program = &I128_IS_ZERO;
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into(), 0i128.into()],
            &[0i128.into(), 0i128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into(), 1i128.into()],
            &[0i128.into(), 1i128.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[0i128.into(), i128::MAX.into()],
            &[0i128.into(), i128::MAX.into()],
//            jit_enum!(1, jit_struct!()),
            jit_enum!(1, jit_struct!()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into(), 0i128.into()],
            &[1i128.into(), 0i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into(), 1i128.into()],
            &[1i128.into(), 1i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[1i128.into(), i128::MAX.into()],
            &[1i128.into(), i128::MAX.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//

//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[i128::MAX.into(), 0i128.into()],
            &[i128::MAX.into(), 0i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[i128::MAX.into(), 1i128.into()],
            &[i128::MAX.into(), 1i128.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//        run_program_assert_output(
        run_program_assert_output(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[i128::MAX.into(), i128::MAX.into()],
            &[i128::MAX.into(), i128::MAX.into()],
//            jit_enum!(0, jit_struct!()),
            jit_enum!(0, jit_struct!()),
//        );
        );
//    }
    }
//}
}
