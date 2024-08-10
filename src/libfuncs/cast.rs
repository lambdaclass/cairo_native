//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, types::TypeBuilder,
    utils::RangeExt,
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::BoundedIntConcreteType,
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{r#type::IntegerType, Block, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};

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
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;
    let src_value: Value = entry.argument(1)?.into();

    let src_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;

    match (src_ty, dst_ty) {
        (
            CoreTypeConcrete::BoundedInt(BoundedIntConcreteType {
                range: src_range, ..
            }),
            CoreTypeConcrete::Sint8(_)
            | CoreTypeConcrete::Sint16(_)
            | CoreTypeConcrete::Sint32(_)
            | CoreTypeConcrete::Sint64(_)
            | CoreTypeConcrete::Sint128(_)
            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint128(_),
        ) => {
            let dst_range = dst_ty.integer_range(registry).unwrap();
            assert!(
                dst_range.lower > src_range.lower || dst_range.upper < src_range.upper,
                "invalid downcast `{}` into `{}`: target range contains the source range",
                info.signature.param_signatures[1].ty,
                info.signature.branch_signatures[0].vars[1].ty
            );

            let dst_value = if src_range.lower != BigInt::ZERO {
                let dst_offset = entry.const_int_from_type(
                    context,
                    location,
                    src_range.lower.clone(),
                    src_value.r#type(),
                )?;
                entry.append_op_result(arith::subi(src_value, dst_offset, location))?
            } else {
                src_value
            };

            let lower_check = if dst_range.lower > src_range.lower {
                let dst_lower = entry.const_int_from_type(
                    context,
                    location,
                    dst_range.lower.clone(),
                    dst_value.r#type(),
                )?;
                Some(entry.append_op_result(arith::cmpi(
                    context,
                    if src_range.lower.sign() != Sign::Minus {
                        CmpiPredicate::Uge
                    } else {
                        CmpiPredicate::Sge
                    },
                    dst_value,
                    dst_lower,
                    location,
                ))?)
            } else {
                None
            };
            let upper_check = if dst_range.upper < src_range.upper {
                let dst_upper = entry.const_int_from_type(
                    context,
                    location,
                    dst_range.upper.clone(),
                    dst_value.r#type(),
                )?;
                Some(entry.append_op_result(arith::cmpi(
                    context,
                    if src_range.lower.sign() != Sign::Minus {
                        CmpiPredicate::Ult
                    } else {
                        CmpiPredicate::Slt
                    },
                    dst_value,
                    dst_upper,
                    location,
                ))?)
            } else {
                None
            };

            let is_in_bounds = match (lower_check, upper_check) {
                (Some(lower_check), Some(upper_check)) => {
                    entry.append_op_result(arith::andi(lower_check, upper_check, location))?
                }
                (Some(lower_check), None) => lower_check,
                (None, Some(upper_check)) => upper_check,
                (None, None) => unreachable!(),
            };

            let dst_value = if dst_range.bit_width() < src_range.bit_width() {
                entry.append_op_result(arith::trunci(
                    dst_value,
                    IntegerType::new(context, dst_range.bit_width()).into(),
                    location,
                ))?
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

        _ => todo!(
            "unhandled downcast from {} to {}",
            info.signature.param_signatures[1].ty,
            info.signature.branch_signatures[0].vars[1].ty
        ),
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
    let src_value = entry.argument(0)?.into();

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;

    let dst_value = match (src_ty, dst_ty) {
        // (CoreTypeConcrete::BoundedInt(_), CoreTypeConcrete::BoundedInt(_)) => todo!(),
        // (CoreTypeConcrete::BoundedInt(_), CoreTypeConcrete::Felt252(_)) => todo!(),
        (
            CoreTypeConcrete::BoundedInt(BoundedIntConcreteType {
                range: src_range, ..
            }),
            CoreTypeConcrete::Sint8(_)
            | CoreTypeConcrete::Sint16(_)
            | CoreTypeConcrete::Sint32(_)
            | CoreTypeConcrete::Sint64(_)
            | CoreTypeConcrete::Sint128(_)
            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint128(_),
        ) => {
            let dst_range = dst_ty.integer_range(registry).unwrap();
            assert!(
                dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper,
                "invalid upcast `{}` into `{}`: target range doesn't contain the source range",
                info.signature.param_signatures[0].ty,
                info.signature.branch_signatures[0].vars[0].ty
            );

            let dst_value = if dst_range.bit_width() > src_range.bit_width() {
                entry.append_op_result(arith::extui(
                    src_value,
                    IntegerType::new(context, dst_range.bit_width()).into(),
                    location,
                ))?
            } else {
                src_value
            };

            if src_range.lower != BigInt::ZERO {
                let dst_offset = entry.const_int_from_type(
                    context,
                    location,
                    src_range.lower.clone(),
                    dst_value.r#type(),
                )?;
                entry.append_op_result(arith::addi(dst_value, dst_offset, location))?
            } else {
                dst_value
            }
        }

        _ => todo!(
            "unhandled upcast from {} to {}",
            info.signature.param_signatures[0].ty,
            info.signature.branch_signatures[0].vars[0].ty
        ),
    };

    entry.append_operation(helper.br(0, &[dst_value], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
        values::JitValue,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref DOWNCAST: (String, Program) = load_cairo! {
            use core::integer::downcast;

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
        static ref UPCAST: (String, Program) = load_cairo! {
            use core::integer::upcast;

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
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())),
                jit_struct!(jit_enum!(1, jit_struct!())),
            ),
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
                JitValue::Bytes31([0xFF; 31]),
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
                    JitValue::Bytes31([
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
                    JitValue::Bytes31([
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
                    JitValue::Bytes31([
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
                    JitValue::Bytes31([
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
                    JitValue::Bytes31([
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
                    JitValue::Bytes31([u8::MAX; 31]),
                ),
            ),
        );
    }
}
