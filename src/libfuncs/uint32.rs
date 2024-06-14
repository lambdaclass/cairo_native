//! # `u32`-related libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            unsigned::{Uint32Concrete, Uint32Traits, UintConcrete},
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
        cf, llvm, ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Location, Region, Value, ValueLike,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Uint32Concrete,
) -> Result<()> {
    match selector {
        UintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        UintConcrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        UintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        UintConcrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, info)
        }
        UintConcrete::WideMul(info) => {
            build_widemul(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Bitwise(info) => {
            super::bitwise::build(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `u32_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Uint32Traits>,
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

/// Generate MLIR operations for the u32 operation libfunc.
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
        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
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

    let op_overflow = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        1,
    )?;

    entry.append_operation(helper.cond_br(
        context,
        op_overflow,
        [1, 0],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u32_eq` libfunc.
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

/// Generate MLIR operations for the `u32_is_zero` libfunc.
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

/// Generate MLIR operations for the `u32_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let result_div = entry.append_op_result(arith::divui(lhs, rhs, location))?;
    let result_rem = entry.append_op_result(arith::remui(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
    Ok(())
}

/// Generate MLIR operations for the `u32_widemul` libfunc.
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

    let lhs = entry.append_op_result(arith::extui(lhs, target_type, location))?;
    let rhs = entry.append_op_result(arith::extui(rhs, target_type, location))?;
    let result = entry.append_op_result(arith::muli(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

/// Generate MLIR operations for the `u32_to_felt252` libfunc.
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

/// Generate MLIR operations for the `u32_sqrt` libfunc.
pub fn build_square_root<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let i16_ty = IntegerType::new(context, 16).into();
    let i32_ty = IntegerType::new(context, 32).into();

    let k1 = entry.const_int_from_type(context, location, 1, i32_ty)?;

    let is_small = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        entry.argument(1)?.into(),
        k1,
        location,
    ))?;

    let result = entry.append_op_result(scf::r#if(
        is_small,
        &[i32_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let k32 = entry.const_int_from_type(context, location, 32, i32_ty)?;

            let leading_zeros = block.append_op_result(
                ods::llvm::intr_ctlz(
                    context,
                    i32_ty,
                    entry.argument(1)?.into(),
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                    location,
                )
                .into(),
            )?;

            let num_bits = block.append_op_result(arith::subi(k32, leading_zeros, location))?;

            let shift_amount = block.append_op_result(arith::addi(num_bits, k1, location))?;

            let parity_mask = block.const_int_from_type(context, location, -2, i32_ty)?;
            let shift_amount =
                block.append_op_result(arith::andi(shift_amount, parity_mask, location))?;

            let k0 = block.const_int_from_type(context, location, 0, i32_ty)?;
            let result = block.append_op_result(scf::r#while(
                &[k0, shift_amount],
                &[i32_ty, i32_ty],
                {
                    let region = Region::new();
                    let block =
                        region.append_block(Block::new(&[(i32_ty, location), (i32_ty, location)]));

                    let result = block.append_op_result(arith::shli(
                        block.argument(0)?.into(),
                        k1,
                        location,
                    ))?;
                    let large_candidate =
                        block.append_op_result(arith::xori(result, k1, location))?;

                    let large_candidate_squared = block.append_op_result(arith::muli(
                        large_candidate,
                        large_candidate,
                        location,
                    ))?;

                    let threshold = block.append_op_result(arith::shrui(
                        entry.argument(1)?.into(),
                        block.argument(1)?.into(),
                        location,
                    ))?;
                    let threshold_is_poison = block.append_op_result(arith::cmpi(
                        context,
                        CmpiPredicate::Eq,
                        block.argument(1)?.into(),
                        k32,
                        location,
                    ))?;
                    let threshold = block.append_op_result(
                        OperationBuilder::new("arith.select", location)
                            .add_operands(&[threshold_is_poison, k0, threshold])
                            .add_results(&[i32_ty])
                            .build()?,
                    )?;

                    let is_in_range = block.append_op_result(arith::cmpi(
                        context,
                        CmpiPredicate::Ule,
                        large_candidate_squared,
                        threshold,
                        location,
                    ))?;

                    let result = block.append_op_result(
                        OperationBuilder::new("arith.select", location)
                            .add_operands(&[is_in_range, large_candidate, result])
                            .add_results(&[i32_ty])
                            .build()?,
                    )?;

                    let k2 = block.const_int_from_type(context, location, 2, i32_ty)?;

                    let shift_amount = block.append_op_result(arith::subi(
                        block.argument(1)?.into(),
                        k2,
                        location,
                    ))?;

                    let should_continue = block.append_op_result(arith::cmpi(
                        context,
                        CmpiPredicate::Sge,
                        shift_amount,
                        k0,
                        location,
                    ))?;
                    block.append_operation(scf::condition(
                        should_continue,
                        &[result, shift_amount],
                        location,
                    ));

                    region
                },
                {
                    let region = Region::new();
                    let block =
                        region.append_block(Block::new(&[(i32_ty, location), (i32_ty, location)]));

                    block.append_operation(scf::r#yield(
                        &[block.argument(0)?.into(), block.argument(1)?.into()],
                        location,
                    ));

                    region
                },
                location,
            ))?;

            block.append_operation(scf::r#yield(&[result], location));

            region
        },
        location,
    ))?;

    let result = entry.append_op_result(arith::trunci(result, i16_ty, location))?;

    entry.append_operation(helper.br(0, &[range_check, result], location));
    Ok(())
}

/// Generate MLIR operations for the `u32_from_felt252` libfunc.
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

    let const_max = entry.const_int_from_type(context, location, u32::MAX, felt252_ty)?;

    let is_ule = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        value,
        const_max,
        location,
    ))?;

    let block_success = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_ule,
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

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo},
        values::JitValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::BigUint;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref U32_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: u32, rhs: u32) -> u32 {
                lhs + rhs
            }
        };
        static ref U32_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: u32, rhs: u32) -> u32 {
                lhs - rhs
            }
        };
        static ref U32_SAFE_DIVMOD: (String, Program) = load_cairo! {
            fn run_test(lhs: u32, rhs: u32) -> (u32, u32) {
                let q = lhs / rhs;
                let r = lhs % rhs;

                (q, r)
            }
        };
        static ref U32_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: u32, rhs: u32) -> bool {
                lhs == rhs
            }
        };
        static ref U32_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn u32_is_zero(a: u32) -> IsZeroResult<u32> implicits() nopanic;

            fn run_test(value: u32) -> bool {
                match u32_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref U32_SQRT: (String, Program) = load_cairo! {
            use core::integer::u32_sqrt;

            fn run_test(value: u32) -> u16 {
                u32_sqrt(value)
            }
        };
        static ref U32_WIDEMUL: (String, Program) = load_cairo! {
            use integer::u32_wide_mul;
            fn run_test(lhs: u32, rhs: u32) -> u64 {
                u32_wide_mul(lhs, rhs)
            }
        };
    }

    use crate::utils::test::run_program_assert_output;

    #[test]
    fn u32_const_min() {
        let program = load_cairo!(
            fn run_test() -> u32 {
                0_u32
            }
        );

        run_program_assert_output(&program, "run_test", &[], 0u32.into());
    }

    #[test]
    fn u32_const_max() {
        let program = load_cairo!(
            fn run_test() -> u32 {
                4294967295_u32
            }
        );

        run_program_assert_output(&program, "run_test", &[], u32::MAX.into());
    }

    #[test]
    fn u32_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_u32.into()
            }
        );

        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
    }

    #[test]
    fn u32_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<u32>, Option<u32>) {
                (4294967295.try_into(), 4294967296.try_into())
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_struct!(
                jit_enum!(0, 4294967295u32.into()),
                jit_enum!(1, jit_struct!()),
            ),
        );
    }

    #[test]
    fn u32_overflowing_add() {
        #[track_caller]
        fn run(lhs: u32, rhs: u32) {
            let program = &U32_OVERFLOWING_ADD;
            let error = Felt::from_bytes_be_slice(b"u32_add Overflow");

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

        const MAX: u32 = u32::MAX;

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
    fn u32_overflowing_sub() {
        #[track_caller]
        fn run(lhs: u32, rhs: u32) {
            let program = &U32_OVERFLOWING_SUB;
            let error = Felt::from_bytes_be_slice(b"u32_sub Overflow");

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

        const MAX: u32 = u32::MAX;

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
    fn u32_equal() {
        let program = &U32_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 0u32.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 0u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 1u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 1u32.into()],
            jit_enum!(1, jit_struct!()),
        );
    }

    #[test]
    fn u32_is_zero() {
        let program = &U32_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn u32_safe_divmod() {
        let program = &U32_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 0u32.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 1u32.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 0xFFFFFFFFu32.into()],
            jit_enum!(1, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 0u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 1u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 0xFFFFFFFFu32.into()],
            jit_enum!(0, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFFFFFu32.into(), 0u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFFFFFu32.into(), 1u32.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFFFFFu32.into(), 0xFFFFFFFFu32.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn u32_sqrt() {
        let program = &U32_SQRT;

        run_program_assert_output(program, "run_test", &[0u32.into()], 0u16.into());
        run_program_assert_output(program, "run_test", &[u32::MAX.into()], 0xFFFFu16.into());

        for i in 0..u32::BITS {
            let x = 1u32 << i;
            let y: u16 = BigUint::from(x)
                .sqrt()
                .try_into()
                .expect("should always fit into a u32");

            run_program_assert_output(program, "run_test", &[x.into()], y.into());
        }
    }

    #[test]
    fn u32_widemul() {
        let program = &U32_WIDEMUL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 0u32.into()],
            0u64.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u32.into(), 1u32.into()],
            0u64.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 0u32.into()],
            0u64.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u32.into(), 1u32.into()],
            1u64.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[u32::MAX.into(), u32::MAX.into()],
            (u32::MAX as u64 * u32::MAX as u64).into(),
        );
    }
}
