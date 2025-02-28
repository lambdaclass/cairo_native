//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    native_assert, native_panic,
    types::TypeBuilder,
    utils::{BlockExt, RangeExt, HALF_PRIME, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        utils::Range,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};
use num_traits::One;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CastConcreteLibfunc,
) -> Result<()> {
    match selector {
        CastConcreteLibfunc::Downcast(info) => {
            build_downcast(context, registry, entry, location, helper, metadata, info)
        }
        CastConcreteLibfunc::Upcast(info) => {
            build_upcast(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `downcast` libfunc.
pub fn build_downcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &DowncastConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let src_value: Value = entry.arg(1)?;

    if info.signature.param_signatures[1].ty == info.signature.branch_signatures[0].vars[1].ty {
        let k0 = entry.const_int(context, location, 0, 1)?;
        entry.append_operation(helper.cond_br(
            context,
            k0,
            [0, 1],
            [&[range_check, src_value], &[range_check]],
            location,
        ));
        return Ok(());
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;

    let dst_range = dst_ty.integer_range(registry)?;
    let src_range = if src_ty.is_felt252(registry)? && dst_range.lower.sign() == Sign::Minus {
        if dst_range.upper.sign() != Sign::Plus {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, PRIME.clone()) + 1,
                upper: BigInt::one(),
            }
        } else {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            }
        }
    } else {
        src_ty.integer_range(registry)?
    };

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_ty.integer_range(registry)?.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    let compute_width = src_range
        .zero_based_bit_width()
        .max(dst_range.zero_based_bit_width());

    let is_signed = src_range.lower.sign() == Sign::Minus;

    let src_value = if compute_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? && !src_ty.is_felt252(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let src_value = if is_signed && src_ty.is_felt252(registry)? {
        if src_range.upper.is_one() {
            let adj_offset =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            entry.append_op_result(arith::subi(src_value, adj_offset, location))?
        } else {
            let adj_offset = entry.const_int_from_type(
                context,
                location,
                HALF_PRIME.clone(),
                src_value.r#type(),
            )?;
            let is_negative =
                entry.cmpi(context, CmpiPredicate::Ugt, src_value, adj_offset, location)?;

            let k_prime =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            let adj_value = entry.append_op_result(arith::subi(src_value, k_prime, location))?;

            entry.append_op_result(arith::select(is_negative, adj_value, src_value, location))?
        }
    } else if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            src_range.lower.clone(),
            src_value.r#type(),
        )?;
        entry.addi(src_value, dst_offset, location)?
    } else {
        src_value
    };

    if !(dst_range.lower > src_range.lower || dst_range.upper < src_range.upper) {
        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };

        let is_in_bounds = entry.const_int(context, location, 1, 1)?;

        entry.append_operation(helper.cond_br(
            context,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        ));
    } else {
        let lower_check = if dst_range.lower > src_range.lower {
            let dst_lower = entry.const_int_from_type(
                context,
                location,
                dst_range.lower.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Uge
                } else {
                    CmpiPredicate::Sge
                },
                src_value,
                dst_lower,
                location,
            )?)
        } else {
            None
        };
        let upper_check = if dst_range.upper < src_range.upper {
            let dst_upper = entry.const_int_from_type(
                context,
                location,
                dst_range.upper.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Ult
                } else {
                    CmpiPredicate::Slt
                },
                src_value,
                dst_upper,
                location,
            )?)
        } else {
            None
        };

        let is_in_bounds = match (lower_check, upper_check) {
            (Some(lower_check), Some(upper_check)) => {
                entry.append_op_result(arith::andi(lower_check, upper_check, location))?
            }
            (Some(lower_check), None) => lower_check,
            (None, Some(upper_check)) => upper_check,
            // its always in bounds since dst is larger than src (i.e no bounds checks needed)
            (None, None) => {
                native_panic!("matched an unreachable: no bounds checks are being performed")
            }
        };

        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };
        entry.append_operation(helper.cond_br(
            context,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        ));
    }

    Ok(())
}

/// Generate MLIR operations for the `upcast` libfunc.
pub fn build_upcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_value = entry.arg(0)?;

    if info.signature.param_signatures[0].ty == info.signature.branch_signatures[0].vars[0].ty {
        entry.append_operation(helper.br(0, &[src_value], location));
        return Ok(());
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;

    let src_range = src_ty.integer_range(registry)?;
    let dst_range = dst_ty.integer_range(registry)?;
    native_assert!(
        if dst_ty.is_felt252(registry)? {
            let alt_range = Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            };

            (dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper)
                || (alt_range.lower <= src_range.lower && alt_range.upper >= src_range.upper)
        } else {
            dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper
        },
        "invalid upcast `{:?}` into `{:?}`: target range doesn't contain the source range",
        info.signature.param_signatures[0].ty,
        info.signature.branch_signatures[0].vars[0].ty
    );

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_range.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    // If the source can be negative, the target type must also contain negatives when upcasting.
    native_assert!(
        src_range.lower.sign() != Sign::Minus
            || dst_ty.is_felt252(registry)?
            || dst_range.lower.sign() == Sign::Minus,
        "if the source range contains negatives, the target range must always contain negatives",
    );
    let is_signed = src_range.lower.sign() == Sign::Minus;

    let dst_value = if dst_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let dst_value = if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            if dst_ty.is_bounded_int(registry)? {
                &src_range.lower - &dst_range.lower
            } else {
                src_range.lower.clone()
            },
            dst_value.r#type(),
        )?;
        entry.addi(dst_value, dst_offset, location)?
    } else {
        dst_value
    };

    let dst_value = if dst_ty.is_felt252(registry)? && src_range.lower.sign() == Sign::Minus {
        let k0 = entry.const_int(context, location, 0, 252)?;
        let is_negative = entry.cmpi(context, CmpiPredicate::Slt, dst_value, k0, location)?;

        let k_prime = entry.const_int(context, location, PRIME.clone(), 252)?;
        let adj_value = entry.addi(dst_value, k_prime, location)?;

        entry.append_op_result(arith::select(is_negative, adj_value, dst_value, location))?
    } else {
        dst_value
    };

    entry.append_operation(helper.br(0, &[dst_value], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{jit_enum, jit_struct, run_sierra_program},
        },
        values::Value,
    };
    use cairo_lang_sierra::{
        extensions::{
            bytes31::Bytes31Type,
            casts::{DowncastLibfunc, UpcastLibfunc},
            int::{
                unsigned::{Uint16Type, Uint32Type, Uint64Type, Uint8Type},
                unsigned128::Uint128Type,
            },
        },
        program::GenericArg,
    };

    macro_rules! cast {
        ($from:ty, $to:ty, $is_up_cast:expr) => {
            if $is_up_cast {
                let mut generator = SierraGenerator::<UpcastLibfunc>::default();

                let from_ty = generator.push_type_declaration::<$from>(&[]).clone();
                let to_ty = generator.push_type_declaration::<$to>(&[]).clone();

                generator.build(&[GenericArg::Type(from_ty), GenericArg::Type(to_ty)])
            } else {
                let mut generator = SierraGenerator::<DowncastLibfunc>::default();

                let from_ty = generator.push_type_declaration::<$from>(&[]).clone();
                let to_ty = generator.push_type_declaration::<$to>(&[]).clone();

                generator.build(&[GenericArg::Type(from_ty), GenericArg::Type(to_ty)])
            }
        };
    }

    fn bytes21_with_filled_u8_max(times: u8) -> Value {
        let mut vec = [0; 31];

        for i in 0..times {
            vec[i as usize] = u8::MAX
        }

        Value::Bytes31(vec)
    }

    #[test]
    fn downcast() {
        let result = run_sierra_program(&cast!(Uint128Type, Uint8Type, false), &[u128::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result =
            run_sierra_program(&cast!(Uint128Type, Uint16Type, false), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result =
            run_sierra_program(&cast!(Uint128Type, Uint32Type, false), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result =
            run_sierra_program(&cast!(Uint128Type, Uint64Type, false), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result =
            run_sierra_program(&cast!(Uint128Type, Uint128Type, false), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint64Type, Uint8Type, false), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint64Type, Uint16Type, false), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint64Type, Uint32Type, false), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint64Type, Uint64Type, false), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint32Type, Uint8Type, false), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint32Type, Uint16Type, false), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint32Type, Uint32Type, false), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint16Type, Uint8Type, false), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result = run_sierra_program(&cast!(Uint16Type, Uint16Type, false), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
        let result =
            run_sierra_program(&cast!(Uint8Type, Uint8Type, false), &[u8::MAX.into()]).return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));
    }

    #[test]
    fn upcast() {
        let result =
            run_sierra_program(&cast!(Uint128Type, Uint128Type, true), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, u128::MAX.into());
        let result = run_sierra_program(&cast!(Uint64Type, Uint128Type, true), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, (u64::MAX as u128).into());
        let result = run_sierra_program(&cast!(Uint64Type, Uint64Type, true), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, u64::MAX.into());
        let result = run_sierra_program(&cast!(Uint32Type, Uint128Type, true), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, (u32::MAX as u128).into());
        let result = run_sierra_program(&cast!(Uint32Type, Uint64Type, true), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, (u32::MAX as u64).into());
        let result = run_sierra_program(&cast!(Uint32Type, Uint32Type, true), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, u32::MAX.into());
        let result = run_sierra_program(&cast!(Uint16Type, Uint128Type, true), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, (u16::MAX as u128).into());
        let result = run_sierra_program(&cast!(Uint16Type, Uint64Type, true), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, (u16::MAX as u64).into());
        let result = run_sierra_program(&cast!(Uint16Type, Uint32Type, true), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, (u16::MAX as u32).into());
        let result = run_sierra_program(&cast!(Uint16Type, Uint16Type, true), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, u16::MAX.into());
        let result = run_sierra_program(&cast!(Uint8Type, Uint128Type, true), &[u8::MAX.into()])
            .return_value;
        assert_eq!(result, (u8::MAX as u128).into());
        let result =
            run_sierra_program(&cast!(Uint8Type, Uint64Type, true), &[u8::MAX.into()]).return_value;
        assert_eq!(result, (u8::MAX as u64).into());
        let result =
            run_sierra_program(&cast!(Uint8Type, Uint32Type, true), &[u8::MAX.into()]).return_value;
        assert_eq!(result, (u8::MAX as u32).into());
        let result =
            run_sierra_program(&cast!(Uint8Type, Uint16Type, true), &[u8::MAX.into()]).return_value;
        assert_eq!(result, (u8::MAX as u16).into());
        let result =
            run_sierra_program(&cast!(Uint8Type, Uint8Type, true), &[u8::MAX.into()]).return_value;
        assert_eq!(result, u8::MAX.into());
        let result = run_sierra_program(
            &cast!(Bytes31Type, Bytes31Type, true),
            &[Value::Bytes31([0xFF; 31])],
        )
        .return_value;
        assert_eq!(result, Value::Bytes31([0xFF; 31]));
        let result =
            run_sierra_program(&cast!(Uint128Type, Bytes31Type, true), &[u128::MAX.into()])
                .return_value;
        assert_eq!(result, bytes21_with_filled_u8_max(16));
        let result = run_sierra_program(&cast!(Uint64Type, Bytes31Type, true), &[u64::MAX.into()])
            .return_value;
        assert_eq!(result, bytes21_with_filled_u8_max(8));
        let result = run_sierra_program(&cast!(Uint32Type, Bytes31Type, true), &[u32::MAX.into()])
            .return_value;
        assert_eq!(result, bytes21_with_filled_u8_max(4));
        let result = run_sierra_program(&cast!(Uint16Type, Bytes31Type, true), &[u16::MAX.into()])
            .return_value;
        assert_eq!(result, bytes21_with_filled_u8_max(2));
        let result = run_sierra_program(&cast!(Uint8Type, Bytes31Type, true), &[u8::MAX.into()])
            .return_value;
        assert_eq!(result, bytes21_with_filled_u8_max(1));
    }
}
