//! # Casting libfuncs

use std::ops::Shr;

use super::LibfuncHelper;
use crate::{
    error::libfuncs::{ErrorImpl, Result},
    metadata::{prime_modulo::PrimeModuloMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, ConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Attribute, Block, Location, ValueLike},
    Context,
};
use num_traits::Signed;
use starknet_types_core::felt::Felt;

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
    metadata: &mut MetadataStorage,
    info: &DowncastConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let src_type = registry.get_type(&info.from_ty)?;
    let dst_type = registry.get_type(&info.to_ty)?;

    dbg!(&info.from_ty);
    dbg!(&info.to_ty);
    let src_width = src_type
        .integer_width()
        .expect("casts always happen between numerical types");
    let dst_width = dst_type
        .integer_width()
        .expect("casts always happen between numerical types");
    dbg!(src_width);
    dbg!(dst_width);
    dbg!(&info.to_range);
    dbg!(&info.from_range);

    let src_ty = src_type.build(context, helper, registry, metadata, &info.from_ty)?;
    let dst_ty = dst_type.build(context, helper, registry, metadata, &info.to_ty)?;
    let is_signed = src_type
        .is_integer_signed()
        .expect("casts always happen between numerical types")
        || dst_type
            .is_integer_signed()
            .expect("casts always happen between numerical types");
    let is_felt = match src_type {
        CoreTypeConcrete::Felt252(_) => true,
        _ => false,
    };
    dbg!(is_signed);

    let mut src_value = entry.argument(1)?.into();

    let mut block = entry;

    let (is_in_range, result) = if src_ty == dst_ty {
        let k0 = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(0, IntegerType::new(context, 1).into()).into(),
                location,
            ))
            .result(0)?
            .into();

        (k0, src_value)
    } else {
        let result = if src_width > dst_width {
            // fix felt cast > half prime = negative
            if is_felt {
                let attr_halfprime_i252 = Attribute::parse(
                    context,
                    &format!(
                        "{} : {}",
                        metadata
                            .get::<PrimeModuloMeta<Felt>>()
                            .ok_or(ErrorImpl::MissingMetadata)?
                            .prime()
                            .shr(1),
                        src_ty
                    ),
                )
                .ok_or(ErrorImpl::ParseAttributeError)?;
                let half_prime = block
                    .append_operation(arith::constant(context, attr_halfprime_i252, location))
                    .result(0)?
                    .into();

                let is_felt_neg = block
                    .append_operation(arith::cmpi(
                        context,
                        CmpiPredicate::Ugt,
                        src_value,
                        half_prime,
                        location,
                    ))
                    .result(0)?
                    .into();

                let is_neg_block = helper.append_block(Block::new(&[]));
                let is_not_neg_block = helper.append_block(Block::new(&[]));
                let final_block = helper.append_block(Block::new(&[(dst_ty, location)]));

                block.append_operation(cf::cond_br(
                    context,
                    is_felt_neg,
                    is_neg_block,
                    is_not_neg_block,
                    &[],
                    &[],
                    location,
                ));

                let prime = block
                    .append_operation(arith::constant(
                        context,
                        Attribute::parse(
                            context,
                            &format!(
                                "{} : {}",
                                metadata
                                    .get::<PrimeModuloMeta<Felt>>()
                                    .ok_or(ErrorImpl::MissingMetadata)?
                                    .prime(),
                                src_ty
                            ),
                        )
                        .ok_or(ErrorImpl::ParseAttributeError)?,
                        location,
                    ))
                    .result(0)?
                    .into();

                src_value = is_neg_block
                    .append_operation(arith::subi(prime, src_value, location))
                    .result(0)?
                    .into();
                src_value = is_neg_block
                    .append_operation(arith::trunci(src_value, dst_ty, location))
                    .result(0)?
                    .into();
                let k1 = is_neg_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(1, src_value.r#type()).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let kneg1 = is_neg_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(-1, src_value.r#type()).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                src_value = is_neg_block
                    .append_operation(arith::addi(src_value, k1, location))
                    .result(0)?
                    .into();
                src_value = is_neg_block
                    .append_operation(arith::xori(src_value, kneg1, location))
                    .result(0)?
                    .into();

                is_neg_block.append_operation(cf::br(final_block, &[src_value], location))

                block = final_block;
            } else {
                block
                    .append_operation(arith::trunci(src_value, dst_ty, location))
                    .result(0)?
                    .into()
            }
        } else if is_signed {
            block
                .append_operation(arith::extsi(src_value, dst_ty, location))
                .result(0)?
                .into()
        } else {
            block
                .append_operation(arith::extui(src_value, dst_ty, location))
                .result(0)?
                .into()
        };

        let (compare_value, compare_ty) = if src_width > dst_width {
            (src_value, src_ty)
        } else {
            (result, dst_ty)
        };

        let max_value = block
            .append_operation(arith::constant(
                context,
                Attribute::parse(
                    context,
                    &format!(
                        "{}: {}",
                        info.to_range
                            .intersection(&info.from_range)
                            .expect("should always intersect")
                            .upper
                            - 1,
                        compare_ty
                    ),
                )
                .expect("downcast: failed to make max value attribute"),
                location,
            ))
            .result(0)?
            .into();

        let min_value = block
            .append_operation(arith::constant(
                context,
                Attribute::parse(
                    context,
                    &format!(
                        "{}: {}",
                        info.to_range
                            .intersection(&info.from_range)
                            .expect("should always intersect")
                            .lower
                            - 1,
                        compare_ty
                    ),
                )
                .expect("downcast: failed to make min value attribute"),
                location,
            ))
            .result(0)?
            .into();

        let is_in_range_upper = block
            .append_operation(arith::cmpi(
                context,
                if is_signed {
                    CmpiPredicate::Sle
                } else {
                    CmpiPredicate::Ule
                },
                compare_value,
                max_value,
                location,
            ))
            .result(0)?
            .into();

        let is_in_range_lower = block
            .append_operation(arith::cmpi(
                context,
                if is_signed {
                    CmpiPredicate::Sge
                } else {
                    CmpiPredicate::Uge
                },
                compare_value,
                min_value,
                location,
            ))
            .result(0)?
            .into();

        let is_in_range = block
            .append_operation(arith::andi(is_in_range_upper, is_in_range_lower, location))
            .result(0)?
            .into();

        (is_in_range, result)
    };

    block.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, result], &[range_check]],
        location,
    ));

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
    let src_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let dst_ty = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;

    let src_width = src_ty
        .integer_width()
        .expect("casts always happen between numerical types");
    let dst_width = dst_ty
        .integer_width()
        .expect("casts always happen between numerical types");
    assert!(src_width <= dst_width);

    let is_signed = src_ty
        .is_integer_signed()
        .expect("casts always happen between numerical types")
        || dst_ty
            .is_integer_signed()
            .expect("casts always happen between numerical types");

    let result = if src_width == dst_width {
        entry.argument(0)?.into()
    } else if is_signed {
        entry
            .append_operation(arith::extsi(
                entry.argument(0)?.into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
                location,
            ))
            .result(0)?
            .into()
    } else {
        entry
            .append_operation(arith::extui(
                entry.argument(0)?.into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
                location,
            ))
            .result(0)?
            .into()
    };

    entry.append_operation(helper.br(0, &[result], location));
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
                        u8::MAX
                    ]),
                    JitValue::Bytes31([
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
                        u8::MAX,
                        u8::MAX
                    ]),
                    JitValue::Bytes31([
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
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX
                    ]),
                    JitValue::Bytes31([
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
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX
                    ]),
                    JitValue::Bytes31([
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
                        u8::MAX
                    ]),
                    JitValue::Bytes31([u8::MAX; 31]),
                ),
            ),
        );
    }
}
