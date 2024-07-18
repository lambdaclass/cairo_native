//! # `u16`-related libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::{Error, Result},
    metadata::MetadataStorage,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            unsigned::{Uint16Concrete, Uint16Traits, UintConcrete},
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
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Attribute,
        Block, Location, Region, Value, ValueLike,
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
    selector: &Uint16Concrete,
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

/// Generate MLIR operations for the `u16_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Uint16Traits>,
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

/// Generate MLIR operations for the u16 operation libfunc.
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

/// Generate MLIR operations for the `u16_eq` libfunc.
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

/// Generate MLIR operations for the `u16_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let arg0: Value = entry.argument(0)?.into();

    let const_0 = entry.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(arg0.r#type(), 0).into(),
        location,
    ))?;

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

/// Generate MLIR operations for the `u16_safe_divmod` libfunc.
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

/// Generate MLIR operations for the `u16_widemul` libfunc.
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

/// Generate MLIR operations for the `u16_to_felt252` libfunc.
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

/// Generate MLIR operations for the `u16_sqrt` libfunc.
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

    let i8_ty = IntegerType::new(context, 8).into();
    let i16_ty = IntegerType::new(context, 16).into();

    let k1 = entry.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(i16_ty, 1).into(),
        location,
    ))?;

    let is_small = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        entry.argument(1)?.into(),
        k1,
        location,
    ))?;

    let result = entry.append_op_result(scf::r#if(
        is_small,
        &[i16_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[entry.argument(1)?.into()], location));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let k16 = entry.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i16_ty, 16).into(),
                location,
            ))?;

            let leading_zeros = block.append_op_result(
                ods::llvm::intr_ctlz(
                    context,
                    i16_ty,
                    entry.argument(1)?.into(),
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                    location,
                )
                .into(),
            )?;

            let num_bits = block.append_op_result(arith::subi(k16, leading_zeros, location))?;

            let shift_amount = block.append_op_result(arith::addi(num_bits, k1, location))?;

            let parity_mask = block.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i16_ty, -2).into(),
                location,
            ))?;
            let shift_amount =
                block.append_op_result(arith::andi(shift_amount, parity_mask, location))?;

            let k0 = block.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i16_ty, 0).into(),
                location,
            ))?;
            let result = block.append_op_result(scf::r#while(
                &[k0, shift_amount],
                &[i16_ty, i16_ty],
                {
                    let region = Region::new();
                    let block =
                        region.append_block(Block::new(&[(i16_ty, location), (i16_ty, location)]));

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
                        k16,
                        location,
                    ))?;
                    let threshold = block.append_op_result(
                        OperationBuilder::new("arith.select", location)
                            .add_operands(&[threshold_is_poison, k0, threshold])
                            .add_results(&[i16_ty])
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
                            .add_results(&[i16_ty])
                            .build()?,
                    )?;

                    let k2 = block.append_op_result(arith::constant(
                        context,
                        IntegerAttribute::new(i16_ty, 2).into(),
                        location,
                    ))?;

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
                        region.append_block(Block::new(&[(i16_ty, location), (i16_ty, location)]));

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

    let result = entry.append_op_result(arith::trunci(result, i8_ty, location))?;

    entry.append_operation(helper.br(0, &[range_check, result], location));
    Ok(())
}

/// Generate MLIR operations for the `u16_from_felt252` libfunc.
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

    let const_max = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", u16::MAX, felt252_ty))
            .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;

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
        static ref U16_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: u16, rhs: u16) -> u16 {
                lhs + rhs
            }
        };
        static ref U16_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: u16, rhs: u16) -> u16 {
                lhs - rhs
            }
        };
        static ref U16_SAFE_DIVMOD: (String, Program) = load_cairo! {
            fn run_test(lhs: u16, rhs: u16) -> (u16, u16) {
                let q = lhs / rhs;
                let r = lhs % rhs;

                (q, r)
            }
        };
        static ref U16_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: u16, rhs: u16) -> bool {
                lhs == rhs
            }
        };
        static ref U16_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn u16_is_zero(a: u16) -> IsZeroResult<u16> implicits() nopanic;

            fn run_test(value: u16) -> bool {
                match u16_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref U16_SQRT: (String, Program) = load_cairo! {
            use core::integer::u16_sqrt;
            use core::num::traits::Sqrt;
            fn run_test(value: u16) -> u8 {
                u16_sqrt(value)
            }
        };
        static ref U16_WIDEMUL: (String, Program) = load_cairo! {
            use integer::u16_wide_mul;
            use core::num::traits::WideMul;
            fn run_test(lhs: u16, rhs: u16) -> u32 {
                u16_wide_mul(lhs, rhs)
            }
        };
    }

    use crate::utils::test::run_program_assert_output;

    #[test]
    fn u16_const_min() {
        let program = load_cairo!(
            fn run_test() -> u16 {
                0_u16
            }
        );

        run_program_assert_output(&program, "run_test", &[], 0u16.into());
    }

    #[test]
    fn u16_const_max() {
        let program = load_cairo!(
            fn run_test() -> u16 {
                65535_u16
            }
        );

        run_program_assert_output(&program, "run_test", &[], u16::MAX.into());
    }

    #[test]
    fn u16_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_u16.into()
            }
        );

        run_program_assert_output(&program, "run_test", &[], Felt::from(2).into());
    }

    #[test]
    fn u16_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<u16>, Option<u16>) {
                (65535.try_into(), 65536.try_into())
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_struct!(jit_enum!(0, 65535u16.into()), jit_enum!(1, jit_struct!()),),
        );
    }

    #[test]
    fn u16_overflowing_add() {
        #[track_caller]
        fn run(lhs: u16, rhs: u16) {
            let program = &U16_OVERFLOWING_ADD;
            let error = Felt::from_bytes_be_slice(b"u16_add Overflow");

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

        const MAX: u16 = u16::MAX;

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
    fn u16_overflowing_sub() {
        #[track_caller]
        fn run(lhs: u16, rhs: u16) {
            let program = &U16_OVERFLOWING_SUB;
            let error = Felt::from_bytes_be_slice(b"u16_sub Overflow");

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

        const MAX: u16 = u16::MAX;

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
    fn u16_equal() {
        let program = &U16_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 0u16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 0u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 1u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 1u16.into()],
            jit_enum!(1, jit_struct!()),
        );
    }

    #[test]
    fn u16_is_zero() {
        let program = &U16_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn u16_safe_divmod() {
        let program = &U16_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 0u16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 1u16.into()],
            jit_enum!(1, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 0xFFFFu16.into()],
            jit_enum!(1, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 0u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 1u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 0xFFFFu16.into()],
            jit_enum!(0, jit_struct!()),
        );

        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFu16.into(), 0u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFu16.into(), 1u16.into()],
            jit_enum!(0, jit_struct!()),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFFFu16.into(), 0xFFFFu16.into()],
            jit_enum!(0, jit_struct!()),
        );
    }

    #[test]
    fn u16_sqrt() {
        let program = &U16_SQRT;

        run_program_assert_output(program, "run_test", &[0u16.into()], 0u8.into());
        run_program_assert_output(program, "run_test", &[u16::MAX.into()], 0xFFu8.into());

        for i in 0..u16::BITS {
            let x = 1u16 << i;
            let y: u8 = BigUint::from(x)
                .sqrt()
                .try_into()
                .expect("should always fit into a u16");

            run_program_assert_output(program, "run_test", &[x.into()], y.into());
        }
    }

    #[test]
    fn u16_widemul() {
        let program = &U16_WIDEMUL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 0u16.into()],
            0u32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u16.into(), 1u16.into()],
            0u32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 0u16.into()],
            0u32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u16.into(), 1u16.into()],
            1u32.into(),
        );
        run_program_assert_output(
            program,
            "run_test",
            &[u16::MAX.into(), u16::MAX.into()],
            (u16::MAX as u32 * u16::MAX as u32).into(),
        );
    }
}
