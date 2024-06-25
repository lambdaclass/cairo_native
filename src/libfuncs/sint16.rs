//! # `i16`-related libfuncs
use super::LibfuncHelper;
use std::ops::Shr;

use crate::{
    block_ext::BlockExt,
    error::{Error, Result},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            signed::{Sint16Concrete, Sint16Traits, SintConcrete},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm,
    },
    ir::{operation::OperationBuilder, r#type::IntegerType, Block, Location, Value, ValueLike},
    Context,
};
use starknet_types_core::felt::Felt;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Sint16Concrete,
) -> Result<()> {
    match selector {
        SintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        SintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        SintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        SintConcrete::WideMul(info) => {
            build_widemul(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Diff(info) => build_diff(context, registry, entry, location, helper, info),
    }
}

/// Generate MLIR operations for the `i16_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Sint16Traits>,
) -> Result<()> {
    let value = info.c;
    let value_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let value = entry.const_int_from_type(context, location, value, value_ty)?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the i16 operation libfunc.
pub fn build_operation<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &IntOperationConcreteLibfunc,
) -> Result<()> {
    let range_check: Value =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op_name = match info.operator {
        IntOperator::OverflowingAdd => "llvm.intr.sadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.ssub.with.overflow",
    };

    let values_type = lhs.r#type();

    let result_type = llvm::r#type::r#struct(
        context,
        &[values_type, IntegerType::new(context, 1).into()],
        false,
    );

    let result = entry.append_op_result(
        OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?,
    )?;

    let op_result = entry.extract_value(context, location, result, values_type, 0)?;

    // Create a const operation to get the 0 value to compare against
    let zero_const = entry.const_int_from_type(context, location, 0, values_type)?;
    // Check if the result is positive
    let is_positive = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Sge,
        op_result,
        zero_const,
        location,
    ))?;

    // Check overflow flag
    let op_overflow = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        1,
    )?;

    let block_not_overflow = helper.append_block(Block::new(&[]));
    let block_overflow = helper.append_block(Block::new(&[]));

    // The libfunc has three possible outputs: In Range, Overflow & Underflow
    entry.append_operation(cf::cond_br(
        context,
        op_overflow,
        block_overflow,
        block_not_overflow,
        &[],
        &[],
        location,
    ));
    // Check wether the result is positive to distinguish between underflowing & overflowing results
    block_overflow.append_operation(helper.cond_br(
        context,
        is_positive,
        [1, 2],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));
    // No Overflow/Underflow -> In range result
    block_not_overflow.append_operation(helper.br(0, &[range_check, op_result], location));

    Ok(())
}

/// Generate MLIR operations for the `i16_eq` libfunc.
pub fn build_equal<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
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

/// Generate MLIR operations for the `i16_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let arg0: Value = entry.argument(0)?.into();

    let const_0 = entry.const_int_from_type(context, location, 0, arg0.r#type())?;

    let condition = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        const_0,
        location,
    ))?;

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

/// Generate MLIR operations for the `i16_widemul` libfunc.
pub fn build_widemul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let target_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;
    let lhs: Value = entry.argument(0)?.into();
    let rhs: Value = entry.argument(1)?.into();

    let lhs = entry.append_op_result(arith::extsi(lhs, target_type, location))?;
    let rhs = entry.append_op_result(arith::extsi(rhs, target_type, location))?;

    let result = entry.append_op_result(arith::muli(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

/// Generate MLIR operations for the `i16_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value: Value = entry.argument(0)?.into();

    let result = entry.append_op_result(arith::extui(value, felt252_ty, location))?;

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `i16_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check: Value =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let value: Value = entry.argument(1)?.into();

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let result_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let const_max = entry.const_int_from_type(context, location, i16::MAX, felt252_ty)?;
    let const_min = entry.const_int_from_type(context, location, i16::MIN, felt252_ty)?;

    let mut block = entry;

    // make unsigned felt into signed felt
    // felt > half prime = negative
    let value = {
        let half_prime: melior::ir::Value = block.const_int_from_type(
            context,
            location,
            metadata
                .get::<PrimeModuloMeta<Felt>>()
                .ok_or(Error::MissingMetadata)?
                .prime()
                .shr(1),
            felt252_ty,
        )?;

        let is_felt_neg = block.append_op_result(arith::cmpi(
            context,
            CmpiPredicate::Ugt,
            value,
            half_prime,
            location,
        ))?;

        let is_neg_block = helper.append_block(Block::new(&[]));
        let is_not_neg_block = helper.append_block(Block::new(&[]));
        let final_block = helper.append_block(Block::new(&[(felt252_ty, location)]));

        block.append_operation(cf::cond_br(
            context,
            is_felt_neg,
            is_neg_block,
            is_not_neg_block,
            &[],
            &[],
            location,
        ));

        {
            let prime = is_neg_block.const_int_from_type(
                context,
                location,
                metadata
                    .get::<PrimeModuloMeta<Felt>>()
                    .ok_or(Error::MissingMetadata)?
                    .prime()
                    .clone(),
                felt252_ty,
            )?;

            let mut src_value_is_neg: melior::ir::Value =
                is_neg_block.append_op_result(arith::subi(prime, value, location))?;

            let kneg1 = is_neg_block.const_int_from_type(context, location, -1, felt252_ty)?;

            src_value_is_neg =
                is_neg_block.append_op_result(arith::muli(src_value_is_neg, kneg1, location))?;

            is_neg_block.append_operation(cf::br(final_block, &[src_value_is_neg], location));
        }

        is_not_neg_block.append_operation(cf::br(final_block, &[value], location));

        block = final_block;

        block.argument(0)?.into()
    };

    let is_smaller_eq = block.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Sle,
        value,
        const_max,
        location,
    ))?;

    let is_bigger_eq = block.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Sge,
        value,
        const_min,
        location,
    ))?;

    let is_ok = block.append_op_result(arith::andi(is_smaller_eq, is_bigger_eq, location))?;

    let block_success = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));

    block.append_operation(cf::cond_br(
        context,
        is_ok,
        block_success,
        block_failure,
        &[],
        &[],
        location,
    ));

    let value = block_success.append_op_result(arith::trunci(value, result_ty, location))?;

    block_success.append_operation(helper.br(0, &[range_check, value], location));
    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

/// Generate MLIR operations for the `i16_diff` libfunc.
pub fn build_diff<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check: Value =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    // Check if lhs >= rhs
    let is_ge =
        entry.append_op_result(arith::cmpi(context, CmpiPredicate::Sge, lhs, rhs, location))?;

    let result = entry.append_op_result(arith::subi(lhs, rhs, location))?;

    entry.append_operation(helper.cond_br(
        context,
        is_ge,
        [0, 1],
        [&[range_check, result], &[range_check, result]],
        location,
    ));
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
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref I16_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: i16, rhs: i16) -> i16 {
                lhs + rhs
            }
        };
        static ref I16_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: i16, rhs: i16) -> i16 {
                lhs - rhs
            }
        };
        static ref I16_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: i16, rhs: i16) -> bool {
                lhs == rhs
            }
        };
        static ref I16_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn i16_is_zero(a: i16) -> IsZeroResult<i16> implicits() nopanic;

            fn run_test(value: i16) -> bool {
                match i16_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref I16_WIDEMUL: (String, Program) = load_cairo! {
            use integer::i16_wide_mul;
            fn run_test(lhs: i16, rhs: i16) -> i32 {
                i16_wide_mul(lhs, rhs)
            }
        };
    }

    #[test]
    fn i16_const_min() {
        let program = load_cairo!(
            fn run_test() -> i16 {
                -32768_i16
            }
        );

        run_program_assert_output(&program, "run_test", &[], i16::MIN.into());
    }

    #[test]
    fn i16_const_max() {
        let program = load_cairo!(
            fn run_test() -> i16 {
                32767_i16
            }
        );

        run_program_assert_output(&program, "run_test", &[], (i16::MAX).into());
    }

    #[test]
    fn i16_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_i16.into()
            }
        );

        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
    }

    #[test]
    fn i16_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<i16>, Option<i16>) {
                (32767.try_into(), 32768.try_into())
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_struct!(jit_enum!(0, 32767i16.into()), jit_enum!(1, jit_struct!()),),
        );
    }

    #[test]
    fn i16_overflowing_add() {
        #[track_caller]
        fn run(lhs: i16, rhs: i16) {
            let program = &I16_OVERFLOWING_ADD;
            let error = Felt::from_bytes_be_slice(b"i16_add Overflow");

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

        const MAX: i16 = i16::MAX;

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
    fn i16_overflowing_sub() {
        #[track_caller]
        fn run(lhs: i16, rhs: i16) {
            let program = &I16_OVERFLOWING_SUB;
            let error = Felt::from_bytes_be_slice(b"i16_sub Overflow");

            let add = lhs.checked_sub(rhs);

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

        const MAX: i16 = i16::MAX;

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
    fn i16_equal() {
        let program = &I16_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 0i16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 0i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 1i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 1i16.into()],
            jit_enum!(1, jit_struct!()),
        );
    }

    #[test]
    fn i16_is_zero() {
        let program = &I16_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn i16_safe_divmod() {
        let program = &I16_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 0i16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 1i16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), i16::MAX.into()],
            jit_enum!(1, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 0i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 1i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), i16::MAX.into()],
            jit_enum!(0, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[i16::MAX.into(), 0i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[i16::MAX.into(), 1i16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[i16::MAX.into(), i16::MAX.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn i16_widemul() {
        let program = &I16_WIDEMUL;

        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 0i16.into()],
            0i32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0i16.into(), 1i16.into()],
            0i32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 0i16.into()],
            0i32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1i16.into(), 1i16.into()],
            1i32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[i16::MAX.into(), i16::MAX.into()],
            (i16::MAX as i32 * i16::MAX as i32).into(),
        );
    }
}
