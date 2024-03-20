//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    error::libfuncs::{ErrorImpl, Result},
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, CastType, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{r#type::IntegerType, Attribute, Block, Location, ValueLike},
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

    let src_ty = src_type.build(context, helper, registry, metadata, &info.from_ty)?;
    let dst_ty = dst_type.build(context, helper, registry, metadata, &info.to_ty)?;

    let location = Location::name(
        context,
        &format!("downcast<{:?}, {:?}>", src_ty, dst_ty),
        location,
    );

    let mut value: melior::ir::Value = entry.argument(1)?.into();

    let block = entry;

    let (is_in_range, result) = if info.from_range.is_full_felt252_range() {
        let range_size = info.to_range.size();
        let minus_range_lower = info.to_range.lower.clone();

        // https://github.com/starkware-libs/cairo/blob/v2.5.4/crates/cairo-lang-sierra-to-casm/src/invocations/casts.rs
        //  cargo r --bin cairo-native-test -- -s program.cairo

        let compare_ty = IntegerType::new(
            context,
            (minus_range_lower.bits().max(range_size.bits()) + 1)
                .max(252)
                .try_into()
                .unwrap(),
        )
        .into();

        if compare_ty != src_ty {
            value = block
                .append_operation(arith::extui(value, compare_ty, location))
                .result(0)?
                .into();
        }

        let const_minus_range_lower = block
            .append_operation(arith::constant(
                context,
                Attribute::parse(context, &format!("{} : {}", minus_range_lower, compare_ty))
                    .unwrap(),
                location,
            ))
            .result(0)?
            .into();

        let const_range_size = block
            .append_operation(arith::constant(
                context,
                Attribute::parse(context, &format!("{} : {}", range_size, compare_ty)).unwrap(),
                location,
            ))
            .result(0)?
            .into();

        let canonical_value = block
            .append_operation(arith::addi(value, const_minus_range_lower, location))
            .result(0)?
            .into();

        let in_range = block
            .append_operation(arith::cmpi(
                context,
                CmpiPredicate::Slt,
                canonical_value,
                const_range_size,
                location,
            ))
            .result(0)?
            .into();

        let trunc_value = if canonical_value.r#type() != dst_ty {
            block
                .append_operation(arith::trunci(canonical_value, dst_ty, location))
                .result(0)?
                .into()
        } else {
            canonical_value
        };

        (in_range, trunc_value)
    } else {
        match info.cast_type() {
            CastType {
                overflow_above: false,
                overflow_below: false,
            } => {
                todo!("cast no overflow")
            }
            CastType {
                overflow_above: true,
                overflow_below: false,
            } => {
                todo!("cast above overflow")
            }
            CastType {
                overflow_above: false,
                overflow_below: true,
            } => {
                todo!("cast below overflow")
            }
            CastType {
                overflow_above: true,
                overflow_below: true,
            } => {
                todo!("cast both overflow")
            }
        }
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
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let dst_ty = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
    let src_type = src_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let dst_type = dst_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let location = Location::name(
        context,
        &format!("upcast<{:?}, {:?}>", src_type, dst_type),
        location,
    );

    let src_width = src_ty.integer_width().ok_or_else(|| {
        ErrorImpl::SierraAssert("casts always happen between numerical types".to_string())
    })?;
    let dst_width = dst_ty.integer_width().ok_or_else(|| {
        ErrorImpl::SierraAssert("casts always happen between numerical types".to_string())
    })?;
    assert!(src_width <= dst_width);

    let block = entry;

    let result = if src_width == dst_width {
        block.argument(0)?.into()
    } else {
        block
            .append_operation(arith::extui(
                entry.argument(0)?.into(),
                IntegerType::new(context, dst_width.try_into()?).into(),
                location,
            ))
            .result(0)?
            .into()
    };

    block.append_operation(helper.br(0, &[result], location));
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
