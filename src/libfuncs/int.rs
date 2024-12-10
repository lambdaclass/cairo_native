use super::{BlockExt, LibfuncHelper};
use crate::{
    error::Result,
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{ProgramRegistryExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        int::{
            signed::{SintConcrete, SintTraits},
            unsigned::{UintConcrete, UintTraits},
            IntConstConcreteLibfunc, IntMulTraits, IntOperationConcreteLibfunc, IntOperator,
            IntTraits,
        },
        is_zero::IsZeroTraits,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm, ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Location, Region, ValueLike,
    },
    Context,
};
use num_bigint::{BigInt, Sign};
use num_traits::Zero;

pub fn build_unsigned<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &UintConcrete<T>,
) -> Result<()>
where
    T: IntMulTraits + IsZeroTraits + UintTraits,
{
    match selector {
        UintConcrete::Bitwise(info) => {
            build_bitwise(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::WideMul(info) => {
            build_wide_mul(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_signed<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &SintConcrete<T>,
) -> Result<()>
where
    T: IntMulTraits + IsZeroTraits + SintTraits,
{
    match selector {
        SintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Diff(_) => todo!(),
        SintConcrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::WideMul(info) => {
            build_wide_mul(context, registry, entry, location, helper, metadata, info)
        }
    }
}

fn build_bitwise<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let bitwise = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    let logical_and = entry.append_op_result(arith::andi(lhs, rhs, location))?;
    let logical_xor = entry.append_op_result(arith::xori(lhs, rhs, location))?;
    let logical_or = entry.append_op_result(arith::ori(lhs, rhs, location))?;

    entry.append_operation(helper.br(
        0,
        &[bitwise, logical_and, logical_xor, logical_or],
        location,
    ));
    Ok(())
}

fn build_const<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<T>,
) -> Result<()>
where
    T: IntTraits,
{
    let value_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let value = entry.const_int_from_type(context, location, info.c, value_ty)?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

fn build_divmod<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    let result_div = entry.append_op_result(arith::divui(lhs, rhs, location))?;
    let result_rem = entry.append_op_result(arith::remui(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
    Ok(())
}

fn build_equal<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let are_equal = entry.cmpi(
        context,
        CmpiPredicate::Eq,
        entry.arg(0)?,
        entry.arg(1)?,
        location,
    )?;

    entry.append_operation(helper.cond_br(context, are_equal, [1, 0], [&[]; 2], location));
    Ok(())
}

fn build_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;
    let threshold = value_ty.integer_range(registry)?;

    let value_ty = value_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.signature.branch_signatures[0].vars[1].ty,
    )?;

    let input = entry.arg(1)?;

    // Handle signedness separately.
    let (is_in_range, value) = if threshold.lower.is_zero() {
        let upper_threshold =
            entry.const_int_from_type(context, location, threshold.upper, input.r#type())?;
        let is_in_range = entry.cmpi(
            context,
            CmpiPredicate::Ult,
            input,
            upper_threshold,
            location,
        )?;

        (is_in_range, input)
    } else {
        let lower_threshold = entry.const_int_from_type(
            context,
            location,
            if threshold.lower.sign() == Sign::Minus {
                &*PRIME - threshold.lower.magnitude()
            } else {
                threshold.lower.magnitude().clone()
            },
            input.r#type(),
        )?;
        let upper_threshold = entry.const_int_from_type(
            context,
            location,
            if threshold.upper.sign() == Sign::Minus {
                &*PRIME - threshold.upper.magnitude()
            } else {
                threshold.upper.magnitude().clone()
            },
            input.r#type(),
        )?;

        let lower_check = entry.cmpi(
            context,
            CmpiPredicate::Sge,
            input,
            lower_threshold,
            location,
        )?;
        let upper_check = entry.cmpi(
            context,
            CmpiPredicate::Slt,
            input,
            upper_threshold,
            location,
        )?;

        let is_in_range =
            entry.append_op_result(arith::andi(lower_check, upper_check, location))?;

        let value = entry.append_op_result(scf::r#if(
            lower_check,
            &[input.r#type()],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let prime = block.const_int_from_type(
                    context,
                    location,
                    BigInt::from_biguint(Sign::Plus, PRIME.clone()),
                    input.r#type(),
                )?;
                let value = block.append_op_result(arith::subi(input, prime, location))?;

                block.append_operation(scf::r#yield(&[value], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[input], location));
                region
            },
            location,
        ))?;

        (is_in_range, value)
    };

    let value = entry.trunci(value, value_ty, location)?;

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));

    Ok(())
}

fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let input = entry.arg(0)?;

    let k0 = entry.const_int_from_type(context, location, 0, input.r#type())?;
    let is_zero = entry.cmpi(context, CmpiPredicate::Eq, input, k0, location)?;

    entry.append_operation(helper.cond_br(context, is_zero, [0, 1], [&[], &[input]], location));
    Ok(())
}

fn build_operation<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntOperationConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let is_signed = !value_ty.integer_range(registry)?.lower.is_zero();
    let value_ty = value_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let op_name = match (is_signed, info.operator) {
        (false, IntOperator::OverflowingAdd) => "llvm.intr.uadd.with.overflow",
        (false, IntOperator::OverflowingSub) => "llvm.intr.usub.with.overflow",
        (true, IntOperator::OverflowingAdd) => "llvm.intr.sadd.with.overflow",
        (true, IntOperator::OverflowingSub) => "llvm.intr.ssub.with.overflow",
    };
    let result_with_overflow = entry.append_op_result(
        OperationBuilder::new(op_name, location)
            .add_operands(&[entry.arg(1)?, entry.arg(2)?])
            .add_results(&[llvm::r#type::r#struct(
                context,
                &[value_ty, IntegerType::new(context, 1).into()],
                false,
            )])
            .build()?,
    )?;

    // TODO: For signed operands, distinguish underflow and overflow cases.
    let result = entry.extract_value(context, location, result_with_overflow, value_ty, 0)?;
    let overflow = entry.extract_value(
        context,
        location,
        result_with_overflow,
        IntegerType::new(context, 1).into(),
        1,
    )?;

    entry.append_operation(helper.cond_br(
        context,
        overflow,
        [1, 0],
        [&[range_check, result]; 2],
        location,
    ));
    Ok(())
}

fn build_square_root<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let input = entry.arg(1)?;
    let (input_bits, value_bits) =
        match registry.get_type(&info.signature.param_signatures[1].ty)? {
            CoreTypeConcrete::Uint8(_) => (8, 8),
            CoreTypeConcrete::Uint16(_) => (16, 8),
            CoreTypeConcrete::Uint32(_) => (32, 16),
            CoreTypeConcrete::Uint64(_) => (64, 32),
            _ => unreachable!(),
        };

    let k1 = entry.const_int(context, location, 1, input_bits)?;
    let is_small = entry.cmpi(context, CmpiPredicate::Ule, input, k1, location)?;

    let value = entry.append_op_result(scf::r#if(
        is_small,
        &[IntegerType::new(context, value_bits).into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let value = block.trunci(
                input,
                IntegerType::new(context, value_bits).into(),
                location,
            )?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let leading_zeros = block.append_op_result(
                ods::llvm::intr_ctlz(
                    context,
                    IntegerType::new(context, input_bits).into(),
                    input,
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                    location,
                )
                .into(),
            )?;

            let k_bits = block.const_int(context, location, input_bits, input_bits)?;
            let num_bits = block.append_op_result(arith::subi(k_bits, leading_zeros, location))?;
            let shift_amount = block.addi(num_bits, k1, location)?;

            let parity_mask = block.const_int(context, location, -2, input_bits)?;
            let shift_amount =
                block.append_op_result(arith::andi(shift_amount, parity_mask, location))?;

            let k0 = block.const_int(context, location, 0, input_bits)?;
            let value = block.append_op_result(scf::r#while(
                &[k0, shift_amount],
                &[
                    IntegerType::new(context, input_bits).into(),
                    IntegerType::new(context, input_bits).into(),
                ],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[
                        (IntegerType::new(context, input_bits).into(), location),
                        (IntegerType::new(context, input_bits).into(), location),
                    ]));

                    let value = block.shli(block.arg(0)?, k1, location)?;
                    let large_candidate =
                        block.append_op_result(arith::xori(value, k1, location))?;
                    let large_candidate_squared =
                        block.muli(large_candidate, large_candidate, location)?;

                    let threshold = block.shrui(input, block.arg(1)?, location)?;
                    let threshold_is_poison =
                        block.cmpi(context, CmpiPredicate::Eq, block.arg(1)?, k_bits, location)?;
                    let threshold = block.append_op_result(arith::select(
                        threshold_is_poison,
                        k0,
                        threshold,
                        location,
                    ))?;

                    let is_in_range = block.cmpi(
                        context,
                        CmpiPredicate::Ule,
                        large_candidate_squared,
                        threshold,
                        location,
                    )?;
                    let value = block.append_op_result(arith::select(
                        is_in_range,
                        large_candidate,
                        value,
                        location,
                    ))?;

                    let k2 = block.const_int(context, location, 2, input_bits)?;
                    let shift_amount =
                        block.append_op_result(arith::subi(block.arg(1)?, k2, location))?;

                    let should_continue =
                        block.cmpi(context, CmpiPredicate::Sge, shift_amount, k0, location)?;
                    block.append_operation(scf::condition(
                        should_continue,
                        &[value, shift_amount],
                        location,
                    ));

                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[
                        (IntegerType::new(context, input_bits).into(), location),
                        (IntegerType::new(context, input_bits).into(), location),
                    ]));

                    block.append_operation(scf::r#yield(&[block.arg(0)?, block.arg(1)?], location));
                    region
                },
                location,
            ))?;

            let value = if input_bits == value_bits {
                value
            } else {
                block.trunci(
                    value,
                    IntegerType::new(context, value_bits).into(),
                    location,
                )?
            };

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        location,
    ))?;

    entry.append_operation(helper.br(0, &[range_check, value], location));
    Ok(())
}

fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let value_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let is_signed = !value_ty.integer_range(registry)?.lower.is_zero();

    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let value = if is_signed {
        let prime = entry.const_int_from_type(
            context,
            location,
            BigInt::from_biguint(Sign::Plus, PRIME.clone()),
            felt252_ty,
        )?;

        let value = entry.extui(entry.arg(0)?, felt252_ty, location)?;
        entry.append_op_result(arith::subi(prime, value, location))?
    } else {
        entry.extui(entry.arg(0)?, felt252_ty, location)?
    };

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

fn build_wide_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let result_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let lhs = entry.extui(entry.arg(0)?, result_ty, location)?;
    let rhs = entry.extui(entry.arg(1)?, result_ty, location)?;
    let result = entry.muli(lhs, rhs, location)?;

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{context::NativeContext, executor::JitNativeExecutor, OptLevel, Value};
    use cairo_lang_sierra::ProgramParser;
    use itertools::Itertools;
    use num_integer::Roots;
    use num_traits::{
        ops::overflowing::{OverflowingAdd, OverflowingSub},
        Bounded, Num,
    };
    use starknet_types_core::felt::Felt;
    use std::{
        fmt::Display,
        mem,
        ops::{BitAnd, BitOr, BitXor, Bound},
    };

    fn test_bitwise<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type Bitwise = Bitwise;
                    type {type_id} = {type_id};
                    type Tuple<{type_id}, {type_id}, {type_id}> = Struct<ut@Tuple, {type_id}, {type_id}, {type_id}>;

                    libfunc {type_id}_bitwise = {type_id}_bitwise;
                    libfunc struct_construct<Tuple<{type_id}, {type_id}, {type_id}>> = struct_construct<Tuple<{type_id}, {type_id}, {type_id}>>;

                    {type_id}_bitwise([0], [1], [2]) -> ([3], [4], [5], [6]);
                    struct_construct<Tuple<{type_id}, {type_id}, {type_id}>>([4], [5], [6]) -> ([7]);
                    return([3], [7]);

                    [0]@0([0]: Bitwise, [1]: {type_id}, [2]: {type_id}) -> (Bitwise, Tuple<{type_id}, {type_id}, {type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(result.builtin_stats.bitwise, 1);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        (perm[0] & perm[1]).into(),
                        (perm[0] ^ perm[1]).into(),
                        (perm[0] | perm[1]).into(),
                    ],
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_const<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Display + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let min = T::min_value();
        let max = T::max_value();
        let program = if min.is_zero() {
            ProgramParser::new()
                .parse(&format!(
                    r#"
                        type {type_id} = {type_id};

                        libfunc {type_id}_const<0> = {type_id}_const<0>;
                        libfunc {type_id}_const<1> = {type_id}_const<1>;
                        libfunc {type_id}_const<{max}> = {type_id}_const<{max}>;

                        {type_id}_const<0>() -> ([0]);
                        return([0]);
                        {type_id}_const<1>() -> ([0]);
                        return([0]);
                        {type_id}_const<{max}>() -> ([0]);
                        return([0]);

                        test_zero@0() -> ({type_id});
                        test_one@2() -> ({type_id});
                        test_max@4() -> ({type_id});
                    "#,
                ))
                .map_err(|e| e.to_string())?
        } else {
            ProgramParser::new()
                .parse(&format!(
                    r#"
                        type {type_id} = {type_id};

                        libfunc {type_id}_const<{min}> = {type_id}_const<{min}>;
                        libfunc {type_id}_const<0> = {type_id}_const<0>;
                        libfunc {type_id}_const<1> = {type_id}_const<1>;
                        libfunc {type_id}_const<{max}> = {type_id}_const<{max}>;

                        {type_id}_const<{min}>() -> ([0]);
                        return([0]);
                        {type_id}_const<0>() -> ([0]);
                        return([0]);
                        {type_id}_const<1>() -> ([0]);
                        return([0]);
                        {type_id}_const<{max}>() -> ([0]);
                        return([0]);

                        test_min@0() -> ({type_id});
                        test_zero@2() -> ({type_id});
                        test_one@4() -> ({type_id});
                        test_max@6() -> ({type_id});
                    "#,
                ))
                .map_err(|e| e.to_string())?
        };

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        if min.is_zero() {
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[0].id, &[], None)?
                    .return_value,
                T::zero().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[1].id, &[], None)?
                    .return_value,
                T::one().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[2].id, &[], None)?
                    .return_value,
                max.into(),
            );
        } else {
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[0].id, &[], None)?
                    .return_value,
                min.into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[1].id, &[], None)?
                    .return_value,
                T::zero().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[2].id, &[], None)?
                    .return_value,
                T::one().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[3].id, &[], None)?
                    .return_value,
                max.into(),
            );
        }

        Ok(())
    }

    fn test_divmod<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type {type_id} = {type_id};
                    type NonZero<{type_id}> = NonZero<{type_id}>;
                    type Tuple<{type_id}, {type_id}> = Struct<ut@Tuple, {type_id}, {type_id}>;

                    libfunc {type_id}_safe_divmod = {type_id}_safe_divmod;
                    libfunc struct_construct<Tuple<{type_id}, {type_id}>> = struct_construct<Tuple<{type_id}, {type_id}>>;

                    {type_id}_safe_divmod([0], [1], [2]) -> ([3], [4], [5]);
                    struct_construct<Tuple<{type_id}, {type_id}>>([4], [5]) -> ([6]);
                    return([3], [6]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: NonZero<{type_id}>) -> (RangeCheck, Tuple<{type_id}, {type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            if perm[1].is_zero() {
                continue;
            }

            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![(perm[0] / perm[1]).into(), (perm[0] % perm[1]).into()],
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_equal<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type Unit = Struct<ut@Tuple>;
                    type {type_id} = {type_id};
                    type core::bool = Enum<ut@core::bool, Unit, Unit>;

                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc {type_id}_eq = {type_id}_eq;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<core::bool, 0> = enum_init<core::bool, 0>;
                    libfunc enum_init<core::bool, 1> = enum_init<core::bool, 1>;

                    struct_construct<Unit>() -> ([2]);
                    {type_id}_eq([0], [1]) {{ fallthrough() 5() }};
                    branch_align() -> ();
                    enum_init<core::bool, 0>([2]) -> ([3]);
                    return([3]);
                    branch_align() -> ();
                    enum_init<core::bool, 1>([2]) -> ([3]);
                    return([3]);

                    [0]@0([0]: {type_id}, [1]: {type_id}) -> (core::bool);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: (perm[0] == perm[1]) as usize,
                    value: Box::new(Value::Struct {
                        fields: Vec::new(),
                        debug_name: None,
                    }),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_from_felt252<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Felt: From<T>,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type felt252 = felt252;
                    type {type_id} = {type_id};
                    type Unit = Struct<ut@Tuple>;
                    type core::option::Option::<core::integer::{type_id}> = Enum<ut@core::option::Option::<core::integer::{type_id}>, {type_id}, Unit>;

                    libfunc {type_id}_try_from_felt252 = {type_id}_try_from_felt252;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<core::option::Option::<core::integer::{type_id}>, 0> = enum_init<core::option::Option::<core::integer::{type_id}>, 0>;
                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc enum_init<core::option::Option::<core::integer::{type_id}>, 1> = enum_init<core::option::Option::<core::integer::{type_id}>, 1>;

                    {type_id}_try_from_felt252([0], [1]) {{ fallthrough([2], [3]) 4([2]) }};
                    branch_align() -> ();
                    enum_init<core::option::Option::<core::integer::{type_id}>, 0>([3]) -> ([4]);
                    return([2], [4]);
                    branch_align() -> ();
                    struct_construct<Unit>() -> ([3]);
                    enum_init<core::option::Option::<core::integer::{type_id}>, 1>([3]) -> ([4]);
                    return([2], [4]);

                    [0]@0([0]: RangeCheck, [1]: felt252) -> (RangeCheck, core::option::Option::<core::integer::{type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        // TODO: Test invalid values too.
        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data {
            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[Felt::from(value).into()], None)?;

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: 0,
                    value: Box::new(value.into()),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_is_zero<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type Unit = Struct<ut@Tuple>;
                    type NonZero<{type_id}> = NonZero<{type_id}>;
                    type IsZeroResult<{type_id}> = Enum<ut@core::zeroable::IsZeroResult::<core::integer::{type_id}>, Unit, NonZero<{type_id}>>;

                    libfunc {type_id}_is_zero = {type_id}_is_zero;
                    libfunc branch_align = branch_align;
                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc enum_init<IsZeroResult<{type_id}>, 0> = enum_init<IsZeroResult<{type_id}>, 0>;
                    libfunc enum_init<IsZeroResult<{type_id}>, 1> = enum_init<IsZeroResult<{type_id}>, 1>;

                    {type_id}_is_zero([0]) {{ fallthrough() 5([2]) }};
                    branch_align() -> ();
                    struct_construct<Unit>() -> ([2]);
                    enum_init<IsZeroResult<{type_id}>, 0>([2]) -> ([3]);
                    return([3]);
                    branch_align() -> ();
                    enum_init<IsZeroResult<{type_id}>, 1>([2]) -> ([3]);
                    return([3]);

                    [0]@0([0]: {type_id}) -> (IsZeroResult<{type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: !value.is_zero() as usize,
                    value: Box::new(if value.is_zero() {
                        Value::Struct {
                            fields: Vec::new(),
                            debug_name: None,
                        }
                    } else {
                        value.into()
                    }),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_unsigned_operation<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + OverflowingAdd + OverflowingSub,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("u{n_bits}");

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type Result<{type_id}, {type_id}> = Enum<ut@core::result::Result::<core::integer::{type_id}, core::integer::{type_id}>, {type_id}, {type_id}>;
                    type Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>> = Struct<ut@Tuple, Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>;
                    type RangeCheck = RangeCheck;

                    libfunc dup<{type_id}> = dup<{type_id}>;
                    libfunc {type_id}_overflowing_add = {type_id}_overflowing_add;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<Result<{type_id}, {type_id}>, 0> = enum_init<Result<{type_id}, {type_id}>, 0>;
                    libfunc jump = jump;
                    libfunc enum_init<Result<{type_id}, {type_id}>, 1> = enum_init<Result<{type_id}, {type_id}>, 1>;
                    libfunc {type_id}_overflowing_sub = {type_id}_overflowing_sub;
                    libfunc struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>> = struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>>;

                    dup<{type_id}>([1]) -> ([1], [3]);
                    dup<{type_id}>([2]) -> ([2], [4]);
                    {type_id}_overflowing_add([0], [1], [2]) {{ fallthrough([5], [6]) 6([5], [6]) }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 0>([6]) -> ([6]);
                    jump() {{ 8() }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 1>([6]) -> ([6]);
                    {type_id}_overflowing_sub([5], [3], [4]) {{ fallthrough([7], [8]) 12([7], [8]) }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 0>([8]) -> ([8]);
                    jump() {{ 14() }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 1>([8]) -> ([8]);
                    struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>>([6], [8]) -> ([9]);
                    return([7], [9]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: {type_id}) -> (RangeCheck, Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let lhs = values[0];
            let rhs = values[1];

            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[lhs.into(), rhs.into()], None)?;

            let (add_result, add_overflow) = lhs.overflowing_add(&rhs);
            let (sub_result, sub_overflow) = lhs.overflowing_sub(&rhs);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        Value::Enum {
                            tag: add_overflow as usize,
                            value: Box::new(add_result.into()),
                            debug_name: None,
                        },
                        Value::Enum {
                            tag: sub_overflow as usize,
                            value: Box::new(sub_result.into()),
                            debug_name: None,
                        },
                    ],
                    debug_name: None
                },
            );
        }

        Ok(())
    }

    fn test_signed_operation<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + Ord + OverflowingAdd + OverflowingSub,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("i{n_bits}");

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type SignedIntegerResult<{type_id}> = Enum<ut@core::integer::SignedIntegerResult::<core::integer::{type_id}>, {type_id}, {type_id}, {type_id}>;
                    type Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>> = Struct<ut@Tuple, SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>;
                    type RangeCheck = RangeCheck;

                    libfunc dup<{type_id}> = dup<{type_id}>;
                    libfunc {type_id}_overflowing_add_impl = {type_id}_overflowing_add_impl;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 0> = enum_init<SignedIntegerResult<{type_id}>, 0>;
                    libfunc jump = jump;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 1> = enum_init<SignedIntegerResult<{type_id}>, 1>;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 2> = enum_init<SignedIntegerResult<{type_id}>, 2>;
                    libfunc {type_id}_overflowing_sub_impl = {type_id}_overflowing_sub_impl;
                    libfunc struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>> = struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>>;

                    dup<{type_id}>([1]) -> ([1], [3]);
                    dup<{type_id}>([2]) -> ([2], [4]);
                    {type_id}_overflowing_add_impl([0], [1], [2]) {{ fallthrough([5], [6]) 6([5], [6]) 9([5], [6]) }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 0>([6]) -> ([6]);
                    jump() {{ 11() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 1>([6]) -> ([6]);
                    jump() {{ 11() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 2>([6]) -> ([6]);
                    {type_id}_overflowing_sub_impl([5], [3], [4]) {{ fallthrough([7], [8]) 15([7], [8]) 18([7], [8]) }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 0>([8]) -> ([8]);
                    jump() {{ 20() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 1>([8]) -> ([8]);
                    jump() {{ 20() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 2>([8]) -> ([8]);
                    struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>>([6], [8]) -> ([9]);
                    return([7], [9]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: {type_id}) -> (RangeCheck, Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let lhs = values[0];
            let rhs = values[1];

            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[lhs.into(), rhs.into()], None)?;

            let (add_result, add_overflow) = lhs.overflowing_add(&rhs);
            let (sub_result, sub_overflow) = lhs.overflowing_sub(&rhs);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        Value::Enum {
                            tag: add_overflow
                                .then(|| lhs >= T::zero() || rhs >= T::zero())
                                .map(|x| (x as usize) + 1)
                                .unwrap_or_default(),
                            value: Box::new(add_result.into()),
                            debug_name: None,
                        },
                        Value::Enum {
                            tag: sub_overflow
                                .then(|| lhs > rhs)
                                .map(|x| (x as usize) + 1)
                                .unwrap_or_default(),
                            value: Box::new(sub_result.into()),
                            debug_name: None,
                        },
                    ],
                    debug_name: None
                },
            );
        }

        Ok(())
    }

    fn test_square_root<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + Eq,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("u{n_bits}");
        let target_type_id = format!("u{}", (n_bits >> 1).max(8));

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type {type_id} = {type_id};{}

                    libfunc {type_id}_sqrt = {type_id}_sqrt;

                    {type_id}_sqrt([0], [1]) -> ([2], [3]);
                    return([2], [3]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}) -> (RangeCheck, {target_type_id});
                "#,
                match n_bits {
                    8 => "".to_string(),
                    _ => format!("\ntype {target_type_id} = {target_type_id};"),
                }
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            match (Value::from(value), result.return_value) {
                (Value::Uint8(target), Value::Uint8(result)) => {
                    assert_eq!(result, target.sqrt());
                }
                (Value::Uint16(target), Value::Uint8(result)) => {
                    assert_eq!(result as u16, target.sqrt());
                }
                (Value::Uint32(target), Value::Uint16(result)) => {
                    assert_eq!(result as u32, target.sqrt());
                }
                (Value::Uint64(target), Value::Uint32(result)) => {
                    assert_eq!(result as u64, target.sqrt());
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn test_to_felt252<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Felt: From<T>,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type felt252 = felt252;
                    type {type_id} = {type_id};

                    libfunc {type_id}_to_felt252 = {type_id}_to_felt252;

                    {type_id}_to_felt252([0]) -> ([1]);
                    return([1]);

                    [0]@0([0]: {type_id}) -> (felt252);
                "#
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(result.return_value, Value::Felt252(value.into()));
        }

        Ok(())
    }

    fn test_wide_mul<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );
        let target_type_id = format!(
            "{}{}",
            if T::min_value().is_zero() { 'u' } else { 'i' },
            n_bits << 1,
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type {target_type_id} = {target_type_id};

                    libfunc {type_id}_wide_mul = {type_id}_wide_mul;

                    {type_id}_wide_mul([0], [1]) -> ([2]);
                    return([2]);

                    [0]@0([0]: {type_id}, [1]: {type_id}) -> ({target_type_id});
                "#
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[values[0].into(), values[1].into()],
                None,
            )?;

            match (
                Value::from(values[0]),
                Value::from(values[1]),
                result.return_value,
            ) {
                (Value::Uint8(lhs), Value::Uint8(rhs), Value::Uint16(result)) => {
                    assert_eq!(result, (lhs as u16) * (rhs as u16));
                }
                (Value::Uint16(lhs), Value::Uint16(rhs), Value::Uint32(result)) => {
                    assert_eq!(result, (lhs as u32) * (rhs as u32));
                }
                (Value::Uint32(lhs), Value::Uint32(rhs), Value::Uint64(result)) => {
                    assert_eq!(result, (lhs as u64) * (rhs as u64));
                }
                (Value::Uint64(lhs), Value::Uint64(rhs), Value::Uint128(result)) => {
                    assert_eq!(result, (lhs as u128) * (rhs as u128));
                }
                (Value::Sint8(lhs), Value::Sint8(rhs), Value::Sint16(result)) => {
                    assert_eq!(result, (lhs as i16) * (rhs as i16));
                }
                (Value::Sint16(lhs), Value::Sint16(rhs), Value::Sint32(result)) => {
                    assert_eq!(result, (lhs as i32) * (rhs as i32));
                }
                (Value::Sint32(lhs), Value::Sint32(rhs), Value::Sint64(result)) => {
                    assert_eq!(result, (lhs as i64) * (rhs as i64));
                }
                (Value::Sint64(lhs), Value::Sint64(rhs), Value::Sint128(result)) => {
                    assert_eq!(result, (lhs as i128) * (rhs as i128));
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    macro_rules! impl_tests {
        ( $( $target:ident for $( $ty:ty as $name:ident ),+ ; )+ ) => {
            $( $(
                #[test]
                fn $name() {
                    $target::<$ty>().unwrap();
                }
            )+ )+
        };
    }

    impl_tests! {
        test_bitwise for
            u8 as u8_bitwise,
            u16 as u16_bitwise,
            u32 as u32_bitwise,
            u64 as u64_bitwise;

        test_const for
            u8 as u8_const,
            u16 as u16_const,
            u32 as u32_const,
            u64 as u64_const,
            i8 as i8_const,
            i16 as i16_const,
            i32 as i32_const,
            i64 as i64_const;

        test_divmod for
            u8 as u8_divmod,
            u16 as u16_divmod,
            u32 as u32_divmod,
            u64 as u64_divmod;

        test_equal for
            u8 as u8_equal,
            u16 as u16_equal,
            u32 as u32_equal,
            u64 as u64_equal,
            i8 as i8_equal,
            i16 as i16_equal,
            i32 as i32_equal,
            i64 as i64_equal;

        test_from_felt252 for
            u8 as u8_from_felt252,
            u16 as u16_from_felt252,
            u32 as u32_from_felt252,
            u64 as u64_from_felt252,
            i8 as i8_from_felt252,
            i16 as i16_from_felt252,
            i32 as i32_from_felt252,
            i64 as i64_from_felt252;

        test_is_zero for
            u8 as u8_is_zero,
            u16 as u16_is_zero,
            u32 as u32_is_zero,
            u64 as u64_is_zero,
            i8 as i8_is_zero,
            i16 as i16_is_zero,
            i32 as i32_is_zero,
            i64 as i64_is_zero;

        test_unsigned_operation for
            u8 as u8_operation,
            u16 as u16_operation,
            u32 as u32_operation,
            u64 as u64_operation;
        test_signed_operation for
            i8 as i8_operation,
            i16 as i16_operation,
            i32 as i32_operation,
            i64 as i64_operation;

        test_square_root for
            u8 as u8_square_root,
            u16 as u16_square_root,
            u32 as u32_square_root,
            u64 as u64_square_root;

        test_to_felt252 for
            u8 as u8_to_felt252,
            u16 as u16_to_felt252,
            u32 as u32_to_felt252,
            u64 as u64_to_felt252,
            i8 as i8_to_felt252,
            i16 as i16_to_felt252,
            i32 as i32_to_felt252,
            i64 as i64_to_felt252;

        test_wide_mul for
            u8 as u8_wide_mul,
            u16 as u16_wide_mul,
            u32 as u32_wide_mul,
            u64 as u64_wide_mul,
            i8 as i8_wide_mul,
            i16 as i16_wide_mul,
            i32 as i32_wide_mul,
            i64 as i64_wide_mul;
    }
}
