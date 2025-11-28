//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result, libfuncs::increment_builtin_counter, metadata::MetadataStorage, native_assert, native_panic, types::TypeBuilder, utils::{HALF_PRIME, PRIME, RangeExt}
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
    helpers::{ArithBlockExt, BuiltinBlockExt},
    ir::{r#type::IntegerType, Block, Location, Value, ValueLike},
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

/// Generate MLIR operations for the `downcast` libfunc which converts from a
/// source type `T` to a target type `U`, where `U` might not fully include `T`.
/// This means that the operation can fail.
///
/// ## Signature
/// ```cairo
/// pub extern const fn downcast<FromType, ToType>(
///     x: FromType,
/// ) -> Option<ToType> implicits(RangeCheck) nopanic;
/// ```
pub fn build_downcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &DowncastConcreteLibfunc,
) -> Result<()> {
    let range_check = entry.arg(0)?;
    let src_value: Value = entry.arg(1)?;

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

    // This is the trivial case, so we just return the value.
    if info.signature.param_signatures[1].ty == info.signature.branch_signatures[0].vars[1].ty {
        // if it is a trivial case and the source type's lower bound is equal 
        // to zero then the cairo compiler checks the upper bound: 
        // https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-sierra/src/extensions/modules/casts.rs#L67.
        // This means the range check gets incremented by one:
        // https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-sierra-to-casm/src/invocations/casts.rs#L56.
        let range_check = if src_range.lower == 0.into() {
            increment_builtin_counter(context, entry, location, range_check)?
        } else {
            range_check
        };
        let k1 = entry.const_int(context, location, 1, 1)?;
        return helper.cond_br(
            context,
            entry,
            k1,
            [0, 1],
            [&[range_check, src_value], &[range_check]],
            location,
        );
    }

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

    // If the target type is wider than the source type, extend the value representation width.
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

    // Correct the value representation accordingly.
    // 1. if it is a felt, then we need to convert the value from [0,P) to
    //    [-P/2, P/2].
    // 2. if it is a bounded_int, we need to offset the value to get the
    //    actual value.
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

    // Check if the source type is included in the target type. If it is not
    // then check if the value is in bounds. If the value is also not in
    // bounds then return an error.
    if dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper {
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

        helper.cond_br(
            context,
            entry,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        )?;
    } else {
        // Check if the value is in bounds with respect to the lower bound.
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
        // Check if the value is in bounds with respect to the upper bound.
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

        // Incrementing the range check depends on whether the source range can hold a felt252 or not
        let range_check = if info.from_range.is_full_felt252_range() {
            let rc_size = BigInt::from(1) << 128;
            // If the range can contain a felt252, how the range check is increased depends on whether the value is in bounds or not:
            // * If it is in bounds, we check whether the destination range size is less than range check size. If it is, increment
            //   the range check builtin by 2. Otherwise, increment it by 1.
            //   https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/range_reduction.rs#L87
            // * If it is not in bounds, increment the range check builtin by 3.
            //   https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/range_reduction.rs#L79
            super::increment_builtin_counter_conditionally_by(
                context,
                entry,
                location,
                range_check,
                if dst_range.size() < rc_size { 2 } else { 1 },
                3,
                is_in_bounds,
            )?
        } else {
            match (lower_check, upper_check) {
                (Some(_), None) | (None, Some(_)) => {
                    // If either the lower or the upper bound was checked, increment the range check builtin by 1.
                    // * In case the lower bound was checked: https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/casts.rs#L135
                    // * In case the upper bound was checked: https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/casts.rs#L111
                    super::increment_builtin_counter_by(context, entry, location, range_check, 1)?
                }
                (Some(lower_check), Some(upper_check)) => {
                    let is_in_range =
                        entry.append_op_result(arith::andi(lower_check, upper_check, location))?;

                    // If the result is in range, increment the range check builtin by 2. Otherwise, increment it by 1.
                    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/casts.rs#L160
                    super::increment_builtin_counter_conditionally_by(
                        context,
                        entry,
                        location,
                        range_check,
                        2,
                        1,
                        is_in_range,
                    )?
                }
                (None, None) => range_check,
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

        helper.cond_br(
            context,
            entry,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        )?;
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
        return helper.br(entry, 0, &[src_value], location);
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

    helper.br(entry, 0, &[dst_value], location)
}

#[cfg(test)]
mod test {
    use crate::{
        jit_enum, jit_struct, load_cairo, utils::testing::run_program_assert_output, values::Value,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;
    use test_case::test_case;

    lazy_static! {
        static ref DOWNCAST: (String, Program) = load_cairo! {
            extern const fn downcast<FromType, ToType>( x: FromType, ) -> Option<ToType> implicits(RangeCheck) nopanic;

            fn run_test(
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128
            ) -> (
                (Option<u8>, Option<u8>, Option<u8>, Option<u8>, Option<u8>),
                (Option<u16>, Option<u16>, Option<u16>, Option<u16>),
                (Option<u32>, Option<u32>, Option<u32>),
                (Option<u64>, Option<u64>),
                (Option<u128>,),
            ) {
                (
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16), downcast(v8)),
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16)),
                    (downcast(v128), downcast(v64), downcast(v32)),
                    (downcast(v128), downcast(v64)),
                    (downcast(v128),),
                )
            }
        };
        static ref DOWNCAST_BOUNDED_INT: (String, Program) = load_cairo! {
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::BoundedInt;

            extern const fn downcast<FromType, ToType>( x: FromType, ) -> Option<ToType> implicits(RangeCheck) nopanic;

            fn test_x_y<
                X,
                Y,
                +TryInto<felt252, X>,
                +Into<Y, felt252>
            >(v: felt252) -> felt252 {
                let v: X = v.try_into().unwrap();
                let v: Y = downcast(v).unwrap();
                v.into()
            }

            fn b0x30_b0x30(v: felt252) -> felt252 { test_x_y::<BoundedInt<0,30>, BoundedInt<0,30>>(v) }
            fn bm31x30_b31x30(v: felt252) -> felt252 { test_x_y::<BoundedInt<-31,30>, BoundedInt<-31,30>>(v) }
            fn bm31x30_bm5x30(v: felt252) -> felt252 { test_x_y::<BoundedInt<-31,30>, BoundedInt<-5,30>>(v) }
            fn bm31x30_b5x30(v: felt252) -> felt252 { test_x_y::<BoundedInt<-31,30>, BoundedInt<5,30>>(v) }
            fn b5x30_b31x31(v: felt252) -> felt252 { test_x_y::<BoundedInt<5,31>, BoundedInt<31,31>>(v) }
            fn bm100x100_bm100xm1(v: felt252) -> felt252 { test_x_y::<BoundedInt<-100,100>, BoundedInt<-100,-1>>(v) }
            fn bm31xm31_bm31xm31(v: felt252) -> felt252 { test_x_y::<BoundedInt<-31,-31>, BoundedInt<-31,-31>>(v) }
            // Check if the target type is wider than the source type
            fn b0x30_b5x40(v: felt252) -> felt252 { test_x_y::<BoundedInt<0,30>, BoundedInt<5,40>>(v) }
            // Check if the source's lower and upper bound are included in the
            // target type.
            fn b0x30_bm40x40(v: felt252) -> felt252 { test_x_y::<BoundedInt<0,30>, BoundedInt<-40,40>>(v) }
        };
        static ref DOWNCAST_FELT: (String, Program) = load_cairo! {
            extern const fn downcast<FromType, ToType>( x: FromType, ) -> Option<ToType> implicits(RangeCheck) nopanic;

            fn test_x_y<
                X,
                Y,
                +TryInto<felt252, X>,
                +Into<Y, felt252>
            >(v: felt252) -> felt252 {
                let v: X = v.try_into().unwrap();
                let v: Y = downcast(v).unwrap();
                v.into()
            }

            fn felt252_i8(v: felt252) -> felt252 { test_x_y::<felt252, i8>(v) }
            fn felt252_i16(v: felt252) -> felt252 { test_x_y::<felt252, i16>(v) }
            fn felt252_i32(v: felt252) -> felt252 { test_x_y::<felt252, i32>(v) }
            fn felt252_i64(v: felt252) -> felt252 { test_x_y::<felt252, i64>(v) }
        };
        static ref UPCAST: (String, Program) = load_cairo! {
            extern const fn upcast<FromType, ToType>(x: FromType) -> ToType nopanic;

            fn run_test(
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
            ) -> (
                (u8,),
                (u16, u16),
                (u32, u32, u32),
                (u64, u64, u64, u64),
                (u128, u128, u128, u128, u128),
                (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
            ) {
                (
                    (upcast(v8),),
                    (upcast(v8), upcast(v16)),
                    (upcast(v8), upcast(v16), upcast(v32)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
                )
            }
        };
    }

    #[test]
    fn downcast() {
        run_program_assert_output(
            &DOWNCAST,
            "run_test",
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
            ],
            jit_struct!(
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(0, u8::MAX.into()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(0, u16::MAX.into()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(0, u32::MAX.into()),
                ),
                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(0, u64::MAX.into())),
                jit_struct!(jit_enum!(0, u128::MAX.into())),
            ),
        );
    }

    #[test_case("b0x30_b0x30", 5.into())]
    #[test_case("bm31x30_b31x30", 5.into())]
    #[test_case("bm31x30_bm5x30", (-5).into())]
    #[test_case("bm31x30_b5x30", 30.into())]
    #[test_case("b5x30_b31x31", 31.into())]
    #[test_case("bm100x100_bm100xm1", (-90).into())]
    #[test_case("bm31xm31_bm31xm31", (-31).into())]
    #[test_case("b0x30_b5x40", 10.into())]
    #[test_case("b0x30_bm40x40", 10.into())]
    fn downcast_bounded_int(entry_point: &str, value: Felt) {
        run_program_assert_output(
            &DOWNCAST_BOUNDED_INT,
            entry_point,
            &[Value::Felt252(value)],
            jit_enum!(0, jit_struct!(Value::Felt252(value))),
        );
    }

    #[test_case("felt252_i8", i8::MAX.into())]
    #[test_case("felt252_i8", i8::MIN.into())]
    #[test_case("felt252_i16", i16::MAX.into())]
    #[test_case("felt252_i16", i16::MIN.into())]
    #[test_case("felt252_i32", i32::MAX.into())]
    #[test_case("felt252_i32", i32::MIN.into())]
    #[test_case("felt252_i64", i64::MAX.into())]
    #[test_case("felt252_i64", i64::MIN.into())]
    fn downcast_felt(entry_point: &str, value: Felt) {
        run_program_assert_output(
            &DOWNCAST_FELT,
            entry_point,
            &[Value::Felt252(value)],
            jit_enum!(0, jit_struct!(Value::Felt252(value))),
        );
    }

    #[test]
    fn upcast() {
        run_program_assert_output(
            &UPCAST,
            "run_test",
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
                Value::Bytes31([0xFF; 31]),
            ],
            jit_struct!(
                jit_struct!(u8::MAX.into()),
                jit_struct!((u8::MAX as u16).into(), u16::MAX.into()),
                jit_struct!(
                    (u8::MAX as u32).into(),
                    (u16::MAX as u32).into(),
                    u32::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u64).into(),
                    (u16::MAX as u64).into(),
                    (u32::MAX as u64).into(),
                    u64::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u128).into(),
                    (u16::MAX as u128).into(),
                    (u32::MAX as u128).into(),
                    (u64::MAX as u128).into(),
                    u128::MAX.into()
                ),
                jit_struct!(
                    Value::Bytes31([
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([u8::MAX; 31]),
                ),
            ),
        );
    }
}
