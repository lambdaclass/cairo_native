////! # Casting libfuncs
//! # Casting libfuncs
//

//use std::ops::Shr;
use std::ops::Shr;
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt,
    block_ext::BlockExt,
//    error::{Error, Result},
    error::{Error, Result},
//    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
//    types::TypeBuilder,
    types::TypeBuilder,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
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
//        cf,
        cf,
//    },
    },
//    ir::{r#type::IntegerType, Block, Location},
    ir::{r#type::IntegerType, Block, Location},
//    Context,
    Context,
//};
};
//use num_bigint::{BigInt, ToBigInt};
use num_bigint::{BigInt, ToBigInt};
//use num_traits::Euclid;
use num_traits::Euclid;
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
//    selector: &CastConcreteLibfunc,
    selector: &CastConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        CastConcreteLibfunc::Downcast(info) => {
        CastConcreteLibfunc::Downcast(info) => {
//            build_downcast(context, registry, entry, location, helper, metadata, info)
            build_downcast(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        CastConcreteLibfunc::Upcast(info) => {
        CastConcreteLibfunc::Upcast(info) => {
//            build_upcast(context, registry, entry, location, helper, metadata, info)
            build_upcast(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `downcast` libfunc.
/// Generate MLIR operations for the `downcast` libfunc.
//pub fn build_downcast<'ctx, 'this>(
pub fn build_downcast<'ctx, 'this>(
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
//    info: &DowncastConcreteLibfunc,
    info: &DowncastConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let range_check =
    let range_check =
//        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
//

//    let src_type = registry.get_type(&info.from_ty)?;
    let src_type = registry.get_type(&info.from_ty)?;
//    let dst_type = registry.get_type(&info.to_ty)?;
    let dst_type = registry.get_type(&info.to_ty)?;
//    let src_width = src_type.integer_width().ok_or_else(|| {
    let src_width = src_type.integer_width().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//    let dst_width = dst_type.integer_width().ok_or_else(|| {
    let dst_width = dst_type.integer_width().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//

//    let src_ty = src_type.build(context, helper, registry, metadata, &info.from_ty)?;
    let src_ty = src_type.build(context, helper, registry, metadata, &info.from_ty)?;
//    let dst_ty = dst_type.build(context, helper, registry, metadata, &info.to_ty)?;
    let dst_ty = dst_type.build(context, helper, registry, metadata, &info.to_ty)?;
//

//    let location = Location::name(
    let location = Location::name(
//        context,
        context,
//        &format!("downcast<{:?}, {:?}>", src_ty, dst_ty),
        &format!("downcast<{:?}, {:?}>", src_ty, dst_ty),
//        location,
        location,
//    );
    );
//

//    let src_is_signed = src_type.is_integer_signed().ok_or_else(|| {
    let src_is_signed = src_type.is_integer_signed().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//    let dst_is_signed = dst_type.is_integer_signed().ok_or_else(|| {
    let dst_is_signed = dst_type.is_integer_signed().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//    let any_is_signed = src_is_signed | dst_is_signed;
    let any_is_signed = src_is_signed | dst_is_signed;
//    let src_is_felt = matches!(
    let src_is_felt = matches!(
//        src_type,
        src_type,
//        CoreTypeConcrete::Felt252(_) | CoreTypeConcrete::BoundedInt(_)
        CoreTypeConcrete::Felt252(_) | CoreTypeConcrete::BoundedInt(_)
//    );
    );
//    let dst_is_felt = matches!(
    let dst_is_felt = matches!(
//        dst_type,
        dst_type,
//        CoreTypeConcrete::Felt252(_) | CoreTypeConcrete::BoundedInt(_)
        CoreTypeConcrete::Felt252(_) | CoreTypeConcrete::BoundedInt(_)
//    );
    );
//    let src_value: melior::ir::Value = entry.argument(1)?.into();
    let src_value: melior::ir::Value = entry.argument(1)?.into();
//

//    let mut block = entry;
    let mut block = entry;
//

//    let (is_in_range, result) = if info.from_ty == info.to_ty {
    let (is_in_range, result) = if info.from_ty == info.to_ty {
//        // can't cast to the same type
        // can't cast to the same type
//        let k0 = block.const_int(context, location, 0, 1)?;
        let k0 = block.const_int(context, location, 0, 1)?;
//        (k0, src_value)
        (k0, src_value)
//    } else {
    } else {
//        // make unsigned felt into signed felt
        // make unsigned felt into signed felt
//        // felt > half prime = negative
        // felt > half prime = negative
//        let felt_to_int = src_is_felt && !dst_is_felt;
        let felt_to_int = src_is_felt && !dst_is_felt;
//        let src_value = if felt_to_int {
        let src_value = if felt_to_int {
//            let attr_halfprime_i252 = metadata
            let attr_halfprime_i252 = metadata
//                .get::<PrimeModuloMeta<Felt>>()
                .get::<PrimeModuloMeta<Felt>>()
//                .ok_or(Error::MissingMetadata)?
                .ok_or(Error::MissingMetadata)?
//                .prime()
                .prime()
//                .shr(1);
                .shr(1);
//

//            let half_prime =
            let half_prime =
//                block.const_int_from_type(context, location, attr_halfprime_i252, src_ty)?;
                block.const_int_from_type(context, location, attr_halfprime_i252, src_ty)?;
//

//            let is_felt_neg = block.append_op_result(arith::cmpi(
            let is_felt_neg = block.append_op_result(arith::cmpi(
//                context,
                context,
//                CmpiPredicate::Ugt,
                CmpiPredicate::Ugt,
//                src_value,
                src_value,
//                half_prime,
                half_prime,
//                location,
                location,
//            ))?;
            ))?;
//            let is_neg_block = helper.append_block(Block::new(&[]));
            let is_neg_block = helper.append_block(Block::new(&[]));
//            let is_not_neg_block = helper.append_block(Block::new(&[]));
            let is_not_neg_block = helper.append_block(Block::new(&[]));
//            let final_block = helper.append_block(Block::new(&[(src_ty, location)]));
            let final_block = helper.append_block(Block::new(&[(src_ty, location)]));
//

//            block.append_operation(cf::cond_br(
            block.append_operation(cf::cond_br(
//                context,
                context,
//                is_felt_neg,
                is_felt_neg,
//                is_neg_block,
                is_neg_block,
//                is_not_neg_block,
                is_not_neg_block,
//                &[],
                &[],
//                &[],
                &[],
//                location,
                location,
//            ));
            ));
//

//            {
            {
//                let value = metadata
                let value = metadata
//                    .get::<PrimeModuloMeta<Felt>>()
                    .get::<PrimeModuloMeta<Felt>>()
//                    .ok_or(Error::MissingMetadata)?
                    .ok_or(Error::MissingMetadata)?
//                    .prime();
                    .prime();
//                let prime = is_neg_block.const_int_from_type(
                let prime = is_neg_block.const_int_from_type(
//                    context,
                    context,
//                    location,
                    location,
//                    value.to_bigint().unwrap(),
                    value.to_bigint().unwrap(),
//                    src_ty,
                    src_ty,
//                )?;
                )?;
//

//                let mut src_value_is_neg =
                let mut src_value_is_neg =
//                    is_neg_block.append_op_result(arith::subi(prime, src_value, location))?;
                    is_neg_block.append_op_result(arith::subi(prime, src_value, location))?;
//

//                let kneg1 = is_neg_block.const_int_from_type(context, location, -1, src_ty)?;
                let kneg1 = is_neg_block.const_int_from_type(context, location, -1, src_ty)?;
//

//                src_value_is_neg = is_neg_block.append_op_result(arith::muli(
                src_value_is_neg = is_neg_block.append_op_result(arith::muli(
//                    src_value_is_neg,
                    src_value_is_neg,
//                    kneg1,
                    kneg1,
//                    location,
                    location,
//                ))?;
                ))?;
//

//                is_neg_block.append_operation(cf::br(final_block, &[src_value_is_neg], location));
                is_neg_block.append_operation(cf::br(final_block, &[src_value_is_neg], location));
//            }
            }
//

//            is_not_neg_block.append_operation(cf::br(final_block, &[src_value], location));
            is_not_neg_block.append_operation(cf::br(final_block, &[src_value], location));
//

//            block = final_block;
            block = final_block;
//

//            block.argument(0)?.into()
            block.argument(0)?.into()
//        } else {
        } else {
//            src_value
            src_value
//        };
        };
//

//        let result = if src_width > dst_width {
        let result = if src_width > dst_width {
//            block.append_op_result(arith::trunci(src_value, dst_ty, location))?
            block.append_op_result(arith::trunci(src_value, dst_ty, location))?
//        } else if src_is_signed {
        } else if src_is_signed {
//            block.append_op_result(arith::extsi(src_value, dst_ty, location))?
            block.append_op_result(arith::extsi(src_value, dst_ty, location))?
//        } else {
        } else {
//            block.append_op_result(arith::extui(src_value, dst_ty, location))?
            block.append_op_result(arith::extui(src_value, dst_ty, location))?
//        };
        };
//

//        let (compare_value, compare_ty) = if src_width > dst_width {
        let (compare_value, compare_ty) = if src_width > dst_width {
//            (src_value, src_ty)
            (src_value, src_ty)
//        } else {
        } else {
//            (result, dst_ty)
            (result, dst_ty)
//        };
        };
//

//        let mut int_max_value: BigInt = info
        let mut int_max_value: BigInt = info
//            .to_range
            .to_range
//            .intersection(&info.from_range)
            .intersection(&info.from_range)
//            .ok_or_else(|| Error::SierraAssert("range should always interesct".to_string()))?
            .ok_or_else(|| Error::SierraAssert("range should always interesct".to_string()))?
//            .upper
            .upper
//            - 1;
            - 1;
//

//        let mut int_min_value = info
        let mut int_min_value = info
//            .to_range
            .to_range
//            .intersection(&info.from_range)
            .intersection(&info.from_range)
//            .ok_or_else(|| Error::SierraAssert("range should always interesct".to_string()))?
            .ok_or_else(|| Error::SierraAssert("range should always interesct".to_string()))?
//            .lower;
            .lower;
//

//        if dst_is_felt {
        if dst_is_felt {
//            let prime = &metadata
            let prime = &metadata
//                .get::<PrimeModuloMeta<Felt>>()
                .get::<PrimeModuloMeta<Felt>>()
//                .ok_or(Error::MissingMetadata)?
                .ok_or(Error::MissingMetadata)?
//                .prime()
                .prime()
//                .to_bigint()
                .to_bigint()
//                .expect("biguint should be casted to bigint");
                .expect("biguint should be casted to bigint");
//

//            int_min_value = int_min_value.rem_euclid(prime);
            int_min_value = int_min_value.rem_euclid(prime);
//            int_max_value = int_max_value.rem_euclid(prime);
            int_max_value = int_max_value.rem_euclid(prime);
//        }
        }
//

//        let max_value = block.const_int_from_type(context, location, int_max_value, compare_ty)?;
        let max_value = block.const_int_from_type(context, location, int_max_value, compare_ty)?;
//        let min_value = block.const_int_from_type(context, location, int_min_value, compare_ty)?;
        let min_value = block.const_int_from_type(context, location, int_min_value, compare_ty)?;
//

//        let is_in_range_upper = block.append_op_result(arith::cmpi(
        let is_in_range_upper = block.append_op_result(arith::cmpi(
//            context,
            context,
//            if any_is_signed {
            if any_is_signed {
//                CmpiPredicate::Sle
                CmpiPredicate::Sle
//            } else {
            } else {
//                CmpiPredicate::Ule
                CmpiPredicate::Ule
//            },
            },
//            compare_value,
            compare_value,
//            max_value,
            max_value,
//            location,
            location,
//        ))?;
        ))?;
//

//        let is_in_range_lower = block.append_op_result(arith::cmpi(
        let is_in_range_lower = block.append_op_result(arith::cmpi(
//            context,
            context,
//            if any_is_signed {
            if any_is_signed {
//                CmpiPredicate::Sge
                CmpiPredicate::Sge
//            } else {
            } else {
//                CmpiPredicate::Uge
                CmpiPredicate::Uge
//            },
            },
//            compare_value,
            compare_value,
//            min_value,
            min_value,
//            location,
            location,
//        ))?;
        ))?;
//

//        let is_in_range =
        let is_in_range =
//            block.append_op_result(arith::andi(is_in_range_upper, is_in_range_lower, location))?;
            block.append_op_result(arith::andi(is_in_range_upper, is_in_range_lower, location))?;
//

//        (is_in_range, result)
        (is_in_range, result)
//    };
    };
//

//    block.append_operation(helper.cond_br(
    block.append_operation(helper.cond_br(
//        context,
        context,
//        is_in_range,
        is_in_range,
//        [0, 1],
        [0, 1],
//        [&[range_check, result], &[range_check]],
        [&[range_check, result], &[range_check]],
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

///// Generate MLIR operations for the `upcast` libfunc.
/// Generate MLIR operations for the `upcast` libfunc.
//pub fn build_upcast<'ctx, 'this>(
pub fn build_upcast<'ctx, 'this>(
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
//    let src_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let src_ty = registry.get_type(&info.param_signatures()[0].ty)?;
//    let dst_ty = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
    let dst_ty = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
//    let src_type = src_ty.build(
    let src_type = src_ty.build(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.param_signatures()[0].ty,
        &info.param_signatures()[0].ty,
//    )?;
    )?;
//    let dst_type = dst_ty.build(
    let dst_type = dst_ty.build(
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
//

//    let location = Location::name(
    let location = Location::name(
//        context,
        context,
//        &format!("upcast<{:?}, {:?}>", src_type, dst_type),
        &format!("upcast<{:?}, {:?}>", src_type, dst_type),
//        location,
        location,
//    );
    );
//

//    let src_width = src_ty.integer_width().ok_or_else(|| {
    let src_width = src_ty.integer_width().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//    let dst_width = dst_ty.integer_width().ok_or_else(|| {
    let dst_width = dst_ty.integer_width().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//    assert!(src_width <= dst_width);
    assert!(src_width <= dst_width);
//

//    let is_signed = src_ty.is_integer_signed().ok_or_else(|| {
    let is_signed = src_ty.is_integer_signed().ok_or_else(|| {
//        Error::SierraAssert("casts always happen between numerical types".to_string())
        Error::SierraAssert("casts always happen between numerical types".to_string())
//    })?;
    })?;
//

//    let is_felt = matches!(dst_ty, CoreTypeConcrete::Felt252(_));
    let is_felt = matches!(dst_ty, CoreTypeConcrete::Felt252(_));
//

//    let block = entry;
    let block = entry;
//

//    let result = if src_width == dst_width {
    let result = if src_width == dst_width {
//        block.argument(0)?.into()
        block.argument(0)?.into()
//    } else if is_signed || is_felt {
    } else if is_signed || is_felt {
//        if is_felt {
        if is_felt {
//            let result = block.append_op_result(arith::extsi(
            let result = block.append_op_result(arith::extsi(
//                block.argument(0)?.into(),
                block.argument(0)?.into(),
//                IntegerType::new(context, dst_width.try_into()?).into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
//                location,
                location,
//            ))?;
            ))?;
//

//            let kzero = block.const_int_from_type(context, location, 0, dst_type)?;
            let kzero = block.const_int_from_type(context, location, 0, dst_type)?;
//

//            let is_neg = block.append_op_result(arith::cmpi(
            let is_neg = block.append_op_result(arith::cmpi(
//                context,
                context,
//                CmpiPredicate::Slt,
                CmpiPredicate::Slt,
//                result,
                result,
//                kzero,
                kzero,
//                location,
                location,
//            ))?;
            ))?;
//

//            let is_neg_block = helper.append_block(Block::new(&[]));
            let is_neg_block = helper.append_block(Block::new(&[]));
//            let is_not_neg_block = helper.append_block(Block::new(&[]));
            let is_not_neg_block = helper.append_block(Block::new(&[]));
//            let final_block = helper.append_block(Block::new(&[(dst_type, location)]));
            let final_block = helper.append_block(Block::new(&[(dst_type, location)]));
//

//            block.append_operation(cf::cond_br(
            block.append_operation(cf::cond_br(
//                context,
                context,
//                is_neg,
                is_neg,
//                is_neg_block,
                is_neg_block,
//                is_not_neg_block,
                is_not_neg_block,
//                &[],
                &[],
//                &[],
                &[],
//                location,
                location,
//            ));
            ));
//

//            {
            {
//                let result = is_not_neg_block.append_op_result(arith::extui(
                let result = is_not_neg_block.append_op_result(arith::extui(
//                    entry.argument(0)?.into(),
                    entry.argument(0)?.into(),
//                    IntegerType::new(context, dst_width.try_into()?).into(),
                    IntegerType::new(context, dst_width.try_into()?).into(),
//                    location,
                    location,
//                ))?;
                ))?;
//

//                is_not_neg_block.append_operation(cf::br(final_block, &[result], location));
                is_not_neg_block.append_operation(cf::br(final_block, &[result], location));
//            }
            }
//

//            {
            {
//                let mut result = is_neg_block.append_op_result(arith::extsi(
                let mut result = is_neg_block.append_op_result(arith::extsi(
//                    entry.argument(0)?.into(),
                    entry.argument(0)?.into(),
//                    IntegerType::new(context, dst_width.try_into()?).into(),
                    IntegerType::new(context, dst_width.try_into()?).into(),
//                    location,
                    location,
//                ))?;
                ))?;
//

//                let value = metadata
                let value = metadata
//                    .get::<PrimeModuloMeta<Felt>>()
                    .get::<PrimeModuloMeta<Felt>>()
//                    .ok_or(Error::MissingMetadata)?
                    .ok_or(Error::MissingMetadata)?
//                    .prime()
                    .prime()
//                    .to_bigint()
                    .to_bigint()
//                    .unwrap();
                    .unwrap();
//

//                let prime = is_neg_block.const_int_from_type(context, location, value, dst_type)?;
                let prime = is_neg_block.const_int_from_type(context, location, value, dst_type)?;
//

//                result = is_neg_block.append_op_result(arith::addi(result, prime, location))?;
                result = is_neg_block.append_op_result(arith::addi(result, prime, location))?;
//                is_neg_block.append_operation(cf::br(final_block, &[result], location));
                is_neg_block.append_operation(cf::br(final_block, &[result], location));
//            }
            }
//

//            let result = final_block.argument(0)?.into();
            let result = final_block.argument(0)?.into();
//            final_block.append_operation(helper.br(0, &[result], location));
            final_block.append_operation(helper.br(0, &[result], location));
//            return Ok(());
            return Ok(());
//        } else {
        } else {
//            block.append_op_result(arith::extsi(
            block.append_op_result(arith::extsi(
//                entry.argument(0)?.into(),
                entry.argument(0)?.into(),
//                IntegerType::new(context, dst_width.try_into()?).into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
//                location,
                location,
//            ))?
            ))?
//        }
        }
//    } else {
    } else {
//        block.append_op_result(arith::extui(
        block.append_op_result(arith::extui(
//            block.argument(0)?.into(),
            block.argument(0)?.into(),
//            IntegerType::new(context, dst_width.try_into()?).into(),
            IntegerType::new(context, dst_width.try_into()?).into(),
//            location,
            location,
//        ))?
        ))?
//    };
    };
//

//    block.append_operation(helper.br(0, &[result], location));
    block.append_operation(helper.br(0, &[result], location));
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
//        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use lazy_static::lazy_static;
    use lazy_static::lazy_static;
//

//    lazy_static! {
    lazy_static! {
//        static ref DOWNCAST: (String, Program) = load_cairo! {
        static ref DOWNCAST: (String, Program) = load_cairo! {
//            use core::integer::downcast;
            use core::integer::downcast;
//

//            fn run_test(
            fn run_test(
//                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128
//            ) -> (
            ) -> (
//                (Option<u8>, Option<u8>, Option<u8>, Option<u8>, Option<u8>),
                (Option<u8>, Option<u8>, Option<u8>, Option<u8>, Option<u8>),
//                (Option<u16>, Option<u16>, Option<u16>, Option<u16>),
                (Option<u16>, Option<u16>, Option<u16>, Option<u16>),
//                (Option<u32>, Option<u32>, Option<u32>),
                (Option<u32>, Option<u32>, Option<u32>),
//                (Option<u64>, Option<u64>),
                (Option<u64>, Option<u64>),
//                (Option<u128>,),
                (Option<u128>,),
//            ) {
            ) {
//                (
                (
//                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16), downcast(v8)),
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16), downcast(v8)),
//                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16)),
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16)),
//                    (downcast(v128), downcast(v64), downcast(v32)),
                    (downcast(v128), downcast(v64), downcast(v32)),
//                    (downcast(v128), downcast(v64)),
                    (downcast(v128), downcast(v64)),
//                    (downcast(v128),),
                    (downcast(v128),),
//                )
                )
//            }
            }
//        };
        };
//        static ref UPCAST: (String, Program) = load_cairo! {
        static ref UPCAST: (String, Program) = load_cairo! {
//            use core::integer::upcast;
            use core::integer::upcast;
//

//            fn run_test(
            fn run_test(
//                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
//            ) -> (
            ) -> (
//                (u8,),
                (u8,),
//                (u16, u16),
                (u16, u16),
//                (u32, u32, u32),
                (u32, u32, u32),
//                (u64, u64, u64, u64),
                (u64, u64, u64, u64),
//                (u128, u128, u128, u128, u128),
                (u128, u128, u128, u128, u128),
//                (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
                (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
//            ) {
            ) {
//                (
                (
//                    (upcast(v8),),
                    (upcast(v8),),
//                    (upcast(v8), upcast(v16)),
                    (upcast(v8), upcast(v16)),
//                    (upcast(v8), upcast(v16), upcast(v32)),
                    (upcast(v8), upcast(v16), upcast(v32)),
//                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
//                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
//                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
//                )
                )
//            }
            }
//        };
        };
//    }
    }
//

//    #[test]
    #[test]
//    fn downcast() {
    fn downcast() {
//        run_program_assert_output(
        run_program_assert_output(
//            &DOWNCAST,
            &DOWNCAST,
//            "run_test",
            "run_test",
//            &[
            &[
//                u8::MAX.into(),
                u8::MAX.into(),
//                u16::MAX.into(),
                u16::MAX.into(),
//                u32::MAX.into(),
                u32::MAX.into(),
//                u64::MAX.into(),
                u64::MAX.into(),
//                u128::MAX.into(),
                u128::MAX.into(),
//            ],
            ],
//            jit_struct!(
            jit_struct!(
//                jit_struct!(
                jit_struct!(
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                ),
                ),
//                jit_struct!(
                jit_struct!(
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                ),
                ),
//                jit_struct!(
                jit_struct!(
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
//                ),
                ),
//                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())),
                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())),
//                jit_struct!(jit_enum!(1, jit_struct!())),
                jit_struct!(jit_enum!(1, jit_struct!())),
//            ),
            ),
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn upcast() {
    fn upcast() {
//        run_program_assert_output(
        run_program_assert_output(
//            &UPCAST,
            &UPCAST,
//            "run_test",
            "run_test",
//            &[
            &[
//                u8::MAX.into(),
                u8::MAX.into(),
//                u16::MAX.into(),
                u16::MAX.into(),
//                u32::MAX.into(),
                u32::MAX.into(),
//                u64::MAX.into(),
                u64::MAX.into(),
//                u128::MAX.into(),
                u128::MAX.into(),
//                JitValue::Bytes31([0xFF; 31]),
                JitValue::Bytes31([0xFF; 31]),
//            ],
            ],
//            jit_struct!(
            jit_struct!(
//                jit_struct!(u8::MAX.into()),
                jit_struct!(u8::MAX.into()),
//                jit_struct!((u8::MAX as u16).into(), u16::MAX.into()),
                jit_struct!((u8::MAX as u16).into(), u16::MAX.into()),
//                jit_struct!(
                jit_struct!(
//                    (u8::MAX as u32).into(),
                    (u8::MAX as u32).into(),
//                    (u16::MAX as u32).into(),
                    (u16::MAX as u32).into(),
//                    u32::MAX.into()
                    u32::MAX.into()
//                ),
                ),
//                jit_struct!(
                jit_struct!(
//                    (u8::MAX as u64).into(),
                    (u8::MAX as u64).into(),
//                    (u16::MAX as u64).into(),
                    (u16::MAX as u64).into(),
//                    (u32::MAX as u64).into(),
                    (u32::MAX as u64).into(),
//                    u64::MAX.into()
                    u64::MAX.into()
//                ),
                ),
//                jit_struct!(
                jit_struct!(
//                    (u8::MAX as u128).into(),
                    (u8::MAX as u128).into(),
//                    (u16::MAX as u128).into(),
                    (u16::MAX as u128).into(),
//                    (u32::MAX as u128).into(),
                    (u32::MAX as u128).into(),
//                    (u64::MAX as u128).into(),
                    (u64::MAX as u128).into(),
//                    u128::MAX.into()
                    u128::MAX.into()
//                ),
                ),
//                jit_struct!(
                jit_struct!(
//                    JitValue::Bytes31([
                    JitValue::Bytes31([
//                        u8::MAX,
                        u8::MAX,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                    ]),
                    ]),
//                    JitValue::Bytes31([
                    JitValue::Bytes31([
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                    ]),
                    ]),
//                    JitValue::Bytes31([
                    JitValue::Bytes31([
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                    ]),
                    ]),
//                    JitValue::Bytes31([
                    JitValue::Bytes31([
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                    ]),
                    ]),
//                    JitValue::Bytes31([
                    JitValue::Bytes31([
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        u8::MAX,
                        u8::MAX,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                        0,
                        0,
//                    ]),
                    ]),
//                    JitValue::Bytes31([u8::MAX; 31]),
                    JitValue::Bytes31([u8::MAX; 31]),
//                ),
                ),
//            ),
            ),
//        );
        );
//    }
    }
//}
}
