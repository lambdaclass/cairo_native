//! # `u128`-related libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::{
            unsigned128::{Uint128Concrete, Uint128Traits},
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
        llvm, ods, scf,
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
    selector: &Uint128Concrete,
) -> Result<()> {
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
pub fn build_byte_reverse<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let bitwise =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let arg1 = entry.argument(1)?.into();

    let res = entry.append_op_result(ods::llvm::intr_bswap(context, arg1, location).into())?;

    entry.append_operation(helper.br(0, &[bitwise, res], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Uint128Traits>,
) -> Result<()> {
    let value = info.c;

    let value_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let value = entry.const_int_from_type(context, location, value, value_ty)?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `u128_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this>(
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

    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let result_div = entry.append_op_result(arith::divui(lhs, rhs, location))?;
    let result_rem = entry.append_op_result(arith::remui(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_equal` libfunc.
pub fn build_equal<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
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

/// Generate MLIR operations for the `u128s_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this>(
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

    let arg1 = entry.argument(1)?.into();

    let k1 = entry.const_int(context, location, 1, 252)?;
    let k128 = entry.const_int(context, location, 128, 252)?;

    let min_wide_val = entry.append_op_result(arith::shli(k1, k128, location))?;
    let is_wide = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Uge,
        arg1,
        min_wide_val,
        location,
    ))?;

    let lsb_bits = entry.append_op_result(arith::trunci(
        arg1,
        IntegerType::new(context, 128).into(),
        location,
    ))?;

    let msb_bits = entry.append_op_result(arith::shrui(arg1, k128, location))?;
    let msb_bits = entry.append_op_result(arith::trunci(
        msb_bits,
        IntegerType::new(context, 128).into(),
        location,
    ))?;

    entry.append_operation(helper.cond_br(
        context,
        is_wide,
        [1, 0],
        [&[range_check, msb_bits, lsb_bits], &[range_check, lsb_bits]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u128_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
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

/// Generate MLIR operations for the `u128_add` and `u128_sub` libfuncs.
pub fn build_operation<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
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

    let result_struct: Value = entry.append_op_result(
        OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?,
    )?;

    let result = entry.extract_value(context, location, result_struct, values_type, 0)?;
    let overflow = entry.extract_value(
        context,
        location,
        result_struct,
        IntegerType::new(context, 1).into(),
        1,
    )?;

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

    let i64_ty = IntegerType::new(context, 64).into();
    let i128_ty = IntegerType::new(context, 128).into();

    let k1 = entry.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(i128_ty, 1).into(),
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

            let k128 = entry.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i128_ty, 128).into(),
                location,
            ))?;

            let leading_zeros = block.append_op_result(
                ods::llvm::intr_ctlz(
                    context,
                    i128_ty,
                    entry.argument(1)?.into(),
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                    location,
                )
                .into(),
            )?;

            let num_bits = block.append_op_result(arith::subi(k128, leading_zeros, location))?;

            let shift_amount = block.append_op_result(arith::addi(num_bits, k1, location))?;

            let parity_mask = block.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i128_ty, -2).into(),
                location,
            ))?;
            let shift_amount =
                block.append_op_result(arith::andi(shift_amount, parity_mask, location))?;

            let k0 = block.append_op_result(arith::constant(
                context,
                IntegerAttribute::new(i128_ty, 0).into(),
                location,
            ))?;
            let result = block.append_op_result(scf::r#while(
                &[k0, shift_amount],
                &[i128_ty, i128_ty],
                {
                    let region = Region::new();
                    let block = region
                        .append_block(Block::new(&[(i128_ty, location), (i128_ty, location)]));

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
                        k128,
                        location,
                    ))?;
                    let threshold = block.append_op_result(
                        OperationBuilder::new("arith.select", location)
                            .add_operands(&[threshold_is_poison, k0, threshold])
                            .add_results(&[i128_ty])
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
                            .add_results(&[i128_ty])
                            .build()?,
                    )?;

                    let k2 = block.append_op_result(arith::constant(
                        context,
                        IntegerAttribute::new(i128_ty, 2).into(),
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
                    let block = region
                        .append_block(Block::new(&[(i128_ty, location), (i128_ty, location)]));

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

    let result = entry.append_op_result(arith::trunci(result, i64_ty, location))?;

    entry.append_operation(helper.br(0, &[range_check, result], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let op = entry.append_operation(arith::extui(
        entry.argument(0)?.into(),
        IntegerType::new(context, 252).into(),
        location,
    ));

    entry.append_operation(helper.br(0, &[op.result(0)?.into()], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_guarantee_mul` libfunc.
pub fn build_guarantee_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
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

    let lhs = entry.append_op_result(arith::extui(lhs, target_type, location))?;
    let rhs = entry.append_op_result(arith::extui(rhs, target_type, location))?;
    let result = entry.append_op_result(arith::muli(lhs, rhs, location))?;
    let result_lo = entry.append_op_result(arith::trunci(result, origin_type, location))?;

    let const_128 = entry.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(target_type, 128).into(),
        location,
    ))?;

    let result_hi = entry.append_op_result(arith::shrui(result, const_128, location))?;
    let result_hi = entry.append_op_result(arith::trunci(result_hi, origin_type, location))?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(0, &[result_hi, result_lo, guarantee], location));
    Ok(())
}

/// Generate MLIR operations for the `u128_guarantee_verify` libfunc.
pub fn build_guarantee_verify<'ctx, 'this>(
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

    entry.append_operation(helper.br(0, &[range_check], location));
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
        let error = JitValue::Felt252(Felt::from_bytes_be_slice(b"Division by 0"));

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
            &[Felt::ZERO.into()],
            jit_enum!(0, 0u128.into()),
        );

        run_program_assert_output(
            &U128_FROM_FELT252,
            "run_test",
            &[Felt::ONE.into()],
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

        run_program_assert_output(program, "run_test", &[0u128.into()], Felt::ZERO.into());
        run_program_assert_output(program, "run_test", &[1u128.into()], Felt::ONE.into());
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
