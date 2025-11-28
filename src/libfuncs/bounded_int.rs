//! # Bounded int libfuncs

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Result},
    execution_result::RANGE_CHECK_BUILTIN_SIZE,
    metadata::MetadataStorage,
    native_assert,
    types::TypeBuilder,
    utils::RangeExt,
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::{
            BoundedIntConcreteLibfunc, BoundedIntConstrainConcreteLibfunc,
            BoundedIntDivRemAlgorithm, BoundedIntDivRemConcreteLibfunc,
            BoundedIntTrimConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        utils::Range,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    helpers::{ArithBlockExt, BuiltinBlockExt},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value, ValueLike},
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
    selector: &BoundedIntConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoundedIntConcreteLibfunc::Add(info) => {
            build_add(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Sub(info) => {
            build_sub(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Mul(info) => {
            build_mul(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::DivRem(info) => {
            build_div_rem(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Constrain(info) => {
            build_constrain(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::TrimMin(info) | BoundedIntConcreteLibfunc::TrimMax(info) => {
            build_trim(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::WrapNonZero(info) => {
            build_wrap_non_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_add<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.offset_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.offset_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.offset_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible.
    let lhs_offset = if lhs_ty.is_bounded_int(registry)? {
        &lhs_range.lower - &compute_range.lower
    } else {
        lhs_range.lower
    };
    let lhs_value = if lhs_offset != BigInt::ZERO {
        let lhs_offset = entry.const_int_from_type(context, location, lhs_offset, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };

    let rhs_offset = if rhs_ty.is_bounded_int(registry)? {
        &rhs_range.lower - &compute_range.lower
    } else {
        rhs_range.lower
    };
    let rhs_value = if rhs_offset != BigInt::ZERO {
        let rhs_offset = entry.const_int_from_type(context, location, rhs_offset, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.addi(lhs_value, rhs_value, location)?;

    // Offset and truncate the result to the output type.
    let res_offset = &dst_range.lower - &compute_range.lower;
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.offset_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Generate MLIR operations for the `bounded_int_sub` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_sub<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.offset_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.offset_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.offset_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible.
    let lhs_offset = if lhs_ty.is_bounded_int(registry)? {
        &lhs_range.lower - &compute_range.lower
    } else {
        lhs_range.lower
    };
    let lhs_value = if lhs_offset != BigInt::ZERO {
        let lhs_offset = entry.const_int_from_type(context, location, lhs_offset, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };

    let rhs_offset = if rhs_ty.is_bounded_int(registry)? {
        &rhs_range.lower - &compute_range.lower
    } else {
        rhs_range.lower
    };
    let rhs_value = if rhs_offset != BigInt::ZERO {
        let rhs_offset = entry.const_int_from_type(context, location, rhs_offset, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.append_op_result(arith::subi(lhs_value, rhs_value, location))?;

    // Offset and truncate the result to the output type.
    let res_offset = dst_range.lower.clone();
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.offset_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Generate MLIR operations for the `bounded_int_mul` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .min(&BigInt::ZERO)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.zero_based_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.zero_based_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.zero_based_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible with the operation.
    let lhs_value = if lhs_ty.is_bounded_int(registry)? && lhs_range.lower != BigInt::ZERO {
        let lhs_offset =
            entry.const_int_from_type(context, location, lhs_range.lower, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };
    let rhs_value = if rhs_ty.is_bounded_int(registry)? && rhs_range.lower != BigInt::ZERO {
        let rhs_offset =
            entry.const_int_from_type(context, location, rhs_range.lower, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.muli(lhs_value, rhs_value, location)?;

    // Offset and truncate the result to the output type.
    let res_offset = dst_range.lower.clone();
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Builds the `bounded_int_div_rem` libfunc, which divides a non negative
/// integer by a positive integer (non zero), returning the quotient and
/// the remainder as bounded ints.
///
/// # Signature
///
/// ```cairo
/// extern fn bounded_int_div_rem<Lhs, Rhs, impl H: DivRemHelper<Lhs, Rhs>>(
///     lhs: Lhs, rhs: NonZero<Rhs>,
/// ) -> (H::DivT, H::RemT) implicits(RangeCheck) nopanic;
/// ```
///
/// The input arguments can be both regular integers or bounded ints.
fn build_div_rem<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntDivRemConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(1)?;
    let rhs_value = entry.arg(2)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.param_signatures()[1].ty)?;
    let rhs_ty = registry.get_type(&info.param_signatures()[2].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let div_range = registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)?
        .integer_range(registry)?;
    let rem_range = registry
        .get_type(&info.branch_signatures()[0].vars[2].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    let div_rem_algorithm = BoundedIntDivRemAlgorithm::try_new(&lhs_range, &rhs_range)
        .to_native_assert_error(&format!(
            "div_rem of ranges: lhs = {:#?} and rhs= {:#?} is not supported yet",
            &lhs_range, &rhs_range
        ))?;

    // Calculate the computation range.
    let compute_range = Range {
        lower: BigInt::ZERO,
        upper: (&lhs_range.upper).max(&rhs_range.upper).clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.zero_based_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.zero_based_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.zero_based_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible with the operation.
    let lhs_value = if lhs_ty.is_bounded_int(registry)? && lhs_range.lower != BigInt::ZERO {
        let lhs_offset =
            entry.const_int_from_type(context, location, lhs_range.lower, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };
    let rhs_value = if rhs_ty.is_bounded_int(registry)? && rhs_range.lower != BigInt::ZERO {
        let rhs_offset =
            entry.const_int_from_type(context, location, rhs_range.lower, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let div_value = entry.append_op_result(arith::divui(lhs_value, rhs_value, location))?;
    let rem_value = entry.append_op_result(arith::remui(lhs_value, rhs_value, location))?;

    // Offset result to the output type.
    let div_value = if div_range.lower.clone() != BigInt::ZERO {
        let div_offset =
            entry.const_int_from_type(context, location, div_range.lower.clone(), compute_ty)?;
        entry.append_op_result(arith::subi(div_value, div_offset, location))?
    } else {
        div_value
    };

    native_assert!(
        rem_range.lower == BigInt::ZERO,
        "The remainder range lower bound should be zero"
    );

    // Truncate to the output type
    let div_value = if div_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            div_value,
            IntegerType::new(context, div_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        div_value
    };
    let rem_value = if rem_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            rem_value,
            IntegerType::new(context, rem_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        rem_value
    };

    // Increase range check builtin by 3, regardless of `div_rem_algorithm`:
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/int/bounded.rs#L100
    let range_check = match div_rem_algorithm {
        BoundedIntDivRemAlgorithm::KnownSmallRhs => crate::libfuncs::increment_builtin_counter_by(
            context,
            entry,
            location,
            entry.arg(0)?,
            3 * RANGE_CHECK_BUILTIN_SIZE,
        )?,
        BoundedIntDivRemAlgorithm::KnownSmallQuotient { .. }
        | BoundedIntDivRemAlgorithm::KnownSmallLhs { .. } => {
            // If `div_rem_algorithm` is `KnownSmallQuotient` or `KnownSmallLhs`, increase range check builtin by 1.
            //
            // Case KnownSmallQuotient: https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/int/bounded.rs#L129
            // Case KnownSmallLhs: https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/int/bounded.rs#L157
            crate::libfuncs::increment_builtin_counter_by(
                context,
                entry,
                location,
                entry.arg(0)?,
                4 * RANGE_CHECK_BUILTIN_SIZE,
            )?
        }
    };

    helper.br(entry, 0, &[range_check, div_value, rem_value], location)
}

/// Generate MLIR operations for the `bounded_int_constrain` libfunc.
fn build_constrain<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntConstrainConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let src_value: Value = entry.arg(1)?;

    let src_ty = registry.get_type(&info.param_signatures()[1].ty)?;
    let src_range = src_ty.integer_range(registry)?;

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_range.zero_based_bit_width()
    };

    let lower_range = registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)?
        .integer_range(registry)?;
    let upper_range = registry
        .get_type(&info.branch_signatures()[1].vars[1].ty)?
        .integer_range(registry)?;

    let boundary = if src_ty.is_bounded_int(registry)? {
        entry.const_int_from_type(
            context,
            location,
            info.boundary.clone() - src_range.lower.clone(),
            src_value.r#type(),
        )?
    } else {
        entry.const_int_from_type(context, location, info.boundary.clone(), src_value.r#type())?
    };

    let is_lower = entry.cmpi(
        context,
        if src_range.lower.sign() == Sign::Minus {
            CmpiPredicate::Slt
        } else {
            CmpiPredicate::Ult
        },
        src_value,
        boundary,
        location,
    )?;

    let lower_block = helper.append_block(Block::new(&[]));
    let upper_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_lower,
        lower_block,
        upper_block,
        &[],
        &[],
        location,
    ));

    {
        let res_value = if src_range.lower != lower_range.lower {
            let lower_offset = &lower_range.lower - &src_range.lower;
            let lower_offset = lower_block.const_int_from_type(
                context,
                location,
                lower_offset,
                src_value.r#type(),
            )?;
            lower_block.append_op_result(arith::subi(src_value, lower_offset, location))?
        } else {
            src_value
        };

        let res_value = if src_width > lower_range.offset_bit_width() {
            lower_block.trunci(
                res_value,
                IntegerType::new(context, lower_range.offset_bit_width()).into(),
                location,
            )?
        } else {
            res_value
        };

        helper.br(lower_block, 0, &[range_check, res_value], location)?;
    }

    {
        let res_value = if src_range.lower != upper_range.lower {
            let upper_offset = &upper_range.lower - &src_range.lower;
            let upper_offset = upper_block.const_int_from_type(
                context,
                location,
                upper_offset,
                src_value.r#type(),
            )?;
            upper_block.append_op_result(arith::subi(src_value, upper_offset, location))?
        } else {
            src_value
        };

        let res_value = if src_width > upper_range.offset_bit_width() {
            upper_block.trunci(
                res_value,
                IntegerType::new(context, upper_range.offset_bit_width()).into(),
                location,
            )?
        } else {
            res_value
        };

        helper.br(upper_block, 1, &[range_check, res_value], location)?;
    }

    Ok(())
}

/// Makes a downcast of a type `T` to `BoundedInt<T::MIN, T::MAX - 1>`
/// or `BoundedInt<T::MIN + 1, T::MAX>` where `T` can be any type of signed
/// or unsigned integer.
///
/// ```cairo
/// extern fn bounded_int_trim<T, const TRIMMED_VALUE: felt252, impl H: TrimHelper<T, TRIMMED_VALUE>>(
///     value: T,
/// ) -> core::internal::OptionRev<H::Target> nopanic;
/// ```
fn build_trim<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntTrimConcreteLibfunc,
) -> Result<()> {
    let value: Value = entry.arg(0)?;
    let trimmed_value = entry.const_int_from_type(
        context,
        location,
        info.trimmed_value.clone(),
        value.r#type(),
    )?;
    let trim_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let is_invalid = entry.cmpi(context, CmpiPredicate::Eq, value, trimmed_value, location)?;
    let int_range = trim_type.integer_range(registry)?;

    // There is no need to truncate the value type since we're only receiving power-of-two integers
    // and constraining their range a single value from either the lower or upper limit. However,
    // since we're returning a `BoundedInt` we need to offset its internal representation
    // accordingly.
    let value = if info.trimmed_value == BigInt::ZERO || int_range.lower < BigInt::ZERO {
        let offset = entry.const_int_from_type(
            context,
            location,
            &info.trimmed_value + 1,
            value.r#type(),
        )?;
        entry.append_op_result(arith::subi(value, offset, location))?
    } else {
        value
    };

    helper.cond_br(
        context,
        entry,
        is_invalid,
        [0, 1],
        [&[], &[value]],
        location,
    )
}

/// Generate MLIR operations for the `bounded_int_is_zero` libfunc.
fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_value: Value = entry.arg(0)?;

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let src_range = src_ty.integer_range(registry)?;

    native_assert!(
        src_range.lower <= BigInt::ZERO && BigInt::ZERO < src_range.upper,
        "value can never be zero"
    );

    let k0 = if src_ty.is_bounded_int(registry)? {
        // We can do the substraction since the lower bound of the bounded int will
        // always be less or equal than 0.
        entry.const_int_from_type(context, location, 0 - src_range.lower, src_value.r#type())?
    } else {
        entry.const_int_from_type(context, location, 0, src_value.r#type())?
    };
    let src_is_zero = entry.cmpi(context, CmpiPredicate::Eq, src_value, k0, location)?;

    helper.cond_br(
        context,
        entry,
        src_is_zero,
        [0, 1],
        [&[], &[src_value]],
        location,
    )
}

/// Generate MLIR operations for the `bounded_int_wrap_non_zero` libfunc.
fn build_wrap_non_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_range = registry
        .get_type(&info.signature.param_signatures[0].ty)?
        .integer_range(registry)?;

    native_assert!(
        src_range.lower > BigInt::ZERO || BigInt::ZERO >= src_range.upper,
        "value must not be zero"
    );

    super::build_noop::<1, false>(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &info.signature.param_signatures,
    )
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::{extensions::utils::Range, program::Program};
    use cairo_vm::Felt252;
    use lazy_static::lazy_static;
    use num_bigint::BigInt;
    use test_case::test_case;

    use crate::{
        context::NativeContext,
        execution_result::ExecutionResult,
        executor::JitNativeExecutor,
        jit_enum, jit_struct, load_cairo,
        utils::testing::{run_program, run_program_assert_output},
        OptLevel, Value,
    };

    lazy_static! {
        static ref TEST_MUL_PROGRAM: (String, Program) = load_cairo! {
                        #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt, MulHelper, mul, UnitInt};

            impl MulHelperBI_m128x127_BI_m128x127 of MulHelper<BoundedInt<-128, 127>, BoundedInt<-128, 127>> {
                type Result = BoundedInt<-16256, 16384>;
            }

            impl MulHelperBI_0x128_BI_0x128 of MulHelper<BoundedInt<0, 128>, BoundedInt<0, 128>> {
                type Result = BoundedInt<0, 16384>;
            }

            impl MulHelperBI_1x31_BI_1x1 of MulHelper<BoundedInt<1, 31>, BoundedInt<1, 1>> {
                type Result = BoundedInt<1, 31>;
            }

            impl MulHelperBI_m1x31_BI_m1xm1 of MulHelper<BoundedInt<-1, 31>, BoundedInt<-1, -1>> {
                type Result = BoundedInt<-31, 1>;
            }

            impl MulHelperBI_31x31_BI_1x1 of MulHelper<BoundedInt<31, 31>, BoundedInt<1, 1>> {
                type Result = BoundedInt<31, 31>;
            }

            impl MulHelperBI_m10x0_BI_0x100 of MulHelper<BoundedInt<-100, 0>, BoundedInt<0, 100>> {
                type Result = BoundedInt<-10000, 0>;
            }

            impl MulHelperBI_1x1_BI_1x1 of MulHelper<BoundedInt<1, 1>, BoundedInt<1, 1>> {
                type Result = BoundedInt<1, 1>;
            }

            impl MulHelperBI_m5x5_UI_2 of MulHelper<BoundedInt<-5, 5>, UnitInt<2>> {
                type Result = BoundedInt<-10, 10>;
            }

            fn bi_m128x127_times_bi_m128x127(a: felt252, b: felt252) -> BoundedInt<-16256, 16384> {
                let a: BoundedInt<-128, 127> = a.try_into().unwrap();
                let b: BoundedInt<-128, 127> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_0x128_times_bi_0x128(a: felt252, b: felt252) -> BoundedInt<0, 16384> {
                let a: BoundedInt<0, 128> = a.try_into().unwrap();
                let b: BoundedInt<0, 128> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_1x31_times_bi_1x1(a: felt252, b: felt252) -> BoundedInt<1, 31> {
                let a: BoundedInt<1, 31> = a.try_into().unwrap();
                let b: BoundedInt<1, 1> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_m1x31_times_bi_m1xm1(a: felt252, b: felt252) -> BoundedInt<-31, 1> {
                let a: BoundedInt<-1, 31> = a.try_into().unwrap();
                let b: BoundedInt<-1, -1> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_31x31_times_bi_1x1(a: felt252, b: felt252) -> BoundedInt<31, 31> {
                let a: BoundedInt<31, 31> = a.try_into().unwrap();
                let b: BoundedInt<1, 1> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_m100x0_times_bi_0x100(a: felt252, b: felt252) -> BoundedInt<-10000, 0> {
                let a: BoundedInt<-100, 0> = a.try_into().unwrap();
                let b: BoundedInt<0, 100> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_1x1_times_bi_1x1(a: felt252, b: felt252) -> BoundedInt<1, 1> {
                let a: BoundedInt<1, 1> = a.try_into().unwrap();
                let b: BoundedInt<1, 1> = b.try_into().unwrap();

                mul(a,b)
            }

            fn bi_m5x5_times_ui_2(a: felt252, b: felt252) -> BoundedInt<-10, 10> {
                let a: BoundedInt<-5, 5> = a.try_into().unwrap();
                let b: UnitInt<2> = b.try_into().unwrap();

                mul(a,b)
            }
        };
    }

    #[test_case("bi_m128x127_times_bi_m128x127", -128, -128, 16384)]
    #[test_case("bi_0x128_times_bi_0x128", 126, 128, 16128)]
    #[test_case("bi_1x31_times_bi_1x1", 31, 1, 31)]
    #[test_case("bi_m1x31_times_bi_m1xm1", 31, -1, -31)]
    #[test_case("bi_31x31_times_bi_1x1", 31, 1, 31)]
    #[test_case("bi_m100x0_times_bi_0x100", -100, 100, -10000)]
    #[test_case("bi_1x1_times_bi_1x1", 1, 1, 1)]
    #[test_case("bi_m5x5_times_ui_2", -3, 2, -6)]
    fn test_mul(entry_point: &str, lhs: i32, rhs: i32, expected_result: i32) {
        let result = run_program(
            &TEST_MUL_PROGRAM,
            entry_point,
            &[
                Value::Felt252(Felt252::from(lhs)),
                Value::Felt252(Felt252::from(rhs)),
            ],
        )
        .return_value;
        if let Value::Enum { value, .. } = result {
            if let Value::Struct { fields, .. } = *value {
                assert!(
                    matches!(fields[0], Value::BoundedInt { value, .. } if value == Felt252::from(expected_result))
                )
            } else {
                panic!("Test returned an unexpected value");
            }
        } else {
            panic!("Test returned value was not an Enum as expected");
        }
    }

    #[test]
    fn test_trim_some_pos_i8() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<-128, 126> {
                let num = match bounded_int::trim_max::<i8>(1) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(1_u8));
    }

    #[test]
    fn test_trim_some_neg_i8() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<-127, 127> {
                let num = match bounded_int::trim_min::<i8>(1) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 1,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(1_u8));
    }

    #[test]
    fn test_trim_some_u32() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<0, 4294967294> {
                let num = match bounded_int::trim_max::<u32>(0xfffffffe) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(0xfffffffe_u32));
    }

    #[test]
    fn test_trim_none() {
        let (_, program) = load_cairo!(
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt};
            use core::internal::OptionRev;

            fn main() -> BoundedInt<-32767, 32767> {
                let num = match bounded_int::trim_min::<i16>(-0x8000) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(0));
    }

    fn assert_bool_output(result: Value, expected_tag: usize) {
        if let Value::Enum { tag, value, .. } = result {
            assert_eq!(tag, 0);
            if let Value::Struct { fields, .. } = *value {
                if let Value::Enum { tag, .. } = fields[0] {
                    assert_eq!(tag, expected_tag)
                }
            }
        }
    }

    #[test]
    fn test_is_zero() {
        let program = load_cairo! {
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt, is_zero};
            use core::zeroable::IsZeroResult;

            fn run_test_1(a: felt252) -> bool {
                let bi: BoundedInt<0, 5> = a.try_into().unwrap();
                match is_zero(bi) {
                    IsZeroResult::Zero => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }

            fn run_test_2(a: felt252) -> bool {
                let bi: BoundedInt<-5, 5> = a.try_into().unwrap();
                match is_zero(bi) {
                    IsZeroResult::Zero => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };

        let result =
            run_program(&program, "run_test_1", &[Value::Felt252(Felt252::from(0))]).return_value;
        assert_bool_output(result, 1);

        let result =
            run_program(&program, "run_test_1", &[Value::Felt252(Felt252::from(5))]).return_value;
        assert_bool_output(result, 0);

        let result =
            run_program(&program, "run_test_2", &[Value::Felt252(Felt252::from(0))]).return_value;
        assert_bool_output(result, 1);

        let result =
            run_program(&program, "run_test_2", &[Value::Felt252(Felt252::from(-5))]).return_value;
        assert_bool_output(result, 0);
    }

    fn assert_constrain_output(result: Value, expected_bi: Value) {
        if let Value::Enum { tag, value, .. } = result {
            assert_eq!(tag, 0);
            if let Value::Struct { fields, .. } = *value {
                assert_eq!(expected_bi, fields[0]);
            }
        }
    }

    #[test]
    fn test_constrain() {
        let program = load_cairo! {
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt, ConstrainHelper, constrain};

            fn run_test_1(a: i8) -> BoundedInt<-128, -1> {
                match constrain::<i8, 0>(a) {
                    Ok(lt0) => lt0,
                    Err(_gt0) => panic!(),
                }
            }

            fn run_test_2(a: i8) -> BoundedInt<0, 127> {
                match constrain::<i8, 0>(a) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }

            impl ConstrainTest1 of ConstrainHelper<BoundedInt<0, 15>, 5> {
                type LowT = BoundedInt<0, 4>;
                type HighT = BoundedInt<5, 15>;
            }

            fn run_test_3(a: felt252) -> BoundedInt<0, 4> {
                let a_bi: BoundedInt<0, 15> = a.try_into().unwrap();
                match constrain::<_, 5>(a_bi) {
                    Ok(lt0) => lt0,
                    Err(_gt0) => panic!(),
                }
            }

            fn run_test_4(a: felt252) -> BoundedInt<5, 15> {
                let a_bi: BoundedInt<0, 15> = a.try_into().unwrap();
                match constrain::<_, 5>(a_bi) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }

            impl ConstrainTest2 of ConstrainHelper<BoundedInt<-10, 10>, 0> {
                type LowT = BoundedInt<-10, -1>;
                type HighT = BoundedInt<0, 10>;
            }

            fn run_test_5(a: felt252) -> BoundedInt<-10, -1> {
                let a_bi: BoundedInt<-10, 10> = a.try_into().unwrap();
                match constrain::<_, 0>(a_bi) {
                    Ok(lt0) => lt0,
                    Err(_gt0) => panic!(),
                }
            }

            fn run_test_6(a: felt252) -> BoundedInt<0, 10> {
                let a_bi: BoundedInt<-10, 10> = a.try_into().unwrap();
                match constrain::<_, 0>(a_bi) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }

            impl ConstrainTest3 of ConstrainHelper<BoundedInt<1, 61>, 31> {
                type LowT = BoundedInt<1, 30>;
                type HighT = BoundedInt<31, 61>;
            }

            fn run_test_7(a: felt252) -> BoundedInt<1, 30> {
                let a_bi: BoundedInt<1, 61> = a.try_into().unwrap();
                match constrain::<_, 31>(a_bi) {
                    Ok(lt0) => lt0,
                    Err(_gt0) => panic!(),
                }
            }

            fn run_test_8(a: felt252) -> BoundedInt<31, 61> {
                let a_bi: BoundedInt<1, 61> = a.try_into().unwrap();
                match constrain::<_, 31>(a_bi) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }

            impl ConstrainTest4 of ConstrainHelper<BoundedInt<-200, -100>, -150> {
                type LowT = BoundedInt<-200, -151>;
                type HighT = BoundedInt<-150, -100>;
            }

            fn run_test_9(a: felt252) -> BoundedInt<-200, -151> {
                let a_bi: BoundedInt<-200, -100> = a.try_into().unwrap();
                match constrain::<_, -150>(a_bi) {
                    Ok(lt0) => lt0,
                    Err(_gt0) => panic!(),
                }
            }

            fn run_test_10(a: felt252) -> BoundedInt<-150, -100> {
                let a_bi: BoundedInt<-200, -100> = a.try_into().unwrap();
                match constrain::<_, -150>(a_bi) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }

            impl ConstrainTest5 of ConstrainHelper<BoundedInt<30, 100>, 100> {
                type LowT = BoundedInt<30, 99>;
                type HighT = BoundedInt<100, 100>;
            }

            fn run_test_11(a: felt252) -> BoundedInt<100, 100> {
                let a_bi: BoundedInt<30, 100> = a.try_into().unwrap();
                match constrain::<_, 100>(a_bi) {
                    Ok(_lt0) => panic!(),
                    Err(gt0) => gt0,
                }
            }
        };

        let result = run_program(&program, "run_test_1", &[Value::Sint8(-1)]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(-1),
                range: Range {
                    lower: BigInt::from(-128),
                    upper: BigInt::from(0),
                },
            },
        );

        let result = run_program(&program, "run_test_2", &[Value::Sint8(1)]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(1),
                range: Range {
                    lower: BigInt::from(0),
                    upper: BigInt::from(128),
                },
            },
        );

        let result = run_program(&program, "run_test_2", &[Value::Sint8(0)]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(0),
                range: Range {
                    lower: BigInt::from(0),
                    upper: BigInt::from(128),
                },
            },
        );

        let result =
            run_program(&program, "run_test_3", &[Value::Felt252(Felt252::from(0))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(0),
                range: Range {
                    lower: BigInt::from(0),
                    upper: BigInt::from(5),
                },
            },
        );

        let result =
            run_program(&program, "run_test_4", &[Value::Felt252(Felt252::from(15))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(15),
                range: Range {
                    lower: BigInt::from(5),
                    upper: BigInt::from(16),
                },
            },
        );

        let result =
            run_program(&program, "run_test_5", &[Value::Felt252(Felt252::from(-5))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(-5),
                range: Range {
                    lower: BigInt::from(-10),
                    upper: BigInt::from(0),
                },
            },
        );

        let result =
            run_program(&program, "run_test_6", &[Value::Felt252(Felt252::from(5))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(5),
                range: Range {
                    lower: BigInt::from(0),
                    upper: BigInt::from(11),
                },
            },
        );

        let result =
            run_program(&program, "run_test_7", &[Value::Felt252(Felt252::from(30))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(30),
                range: Range {
                    lower: BigInt::from(1),
                    upper: BigInt::from(31),
                },
            },
        );

        let result =
            run_program(&program, "run_test_8", &[Value::Felt252(Felt252::from(31))]).return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(31),
                range: Range {
                    lower: BigInt::from(31),
                    upper: BigInt::from(62),
                },
            },
        );

        let result = run_program(
            &program,
            "run_test_9",
            &[Value::Felt252(Felt252::from(-200))],
        )
        .return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(-200),
                range: Range {
                    lower: BigInt::from(-200),
                    upper: BigInt::from(-150),
                },
            },
        );

        let result = run_program(
            &program,
            "run_test_10",
            &[Value::Felt252(Felt252::from(-150))],
        )
        .return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(-150),
                range: Range {
                    lower: BigInt::from(-150),
                    upper: BigInt::from(-99),
                },
            },
        );

        let result = run_program(
            &program,
            "run_test_11",
            &[Value::Felt252(Felt252::from(100))],
        )
        .return_value;
        assert_constrain_output(
            result,
            Value::BoundedInt {
                value: Felt252::from(100),
                range: Range {
                    lower: BigInt::from(100),
                    upper: BigInt::from(101),
                },
            },
        );
    }

    lazy_static! {
        static ref TEST_DIV_REM_PROGRAM: (String, Program) = load_cairo! {
            #[feature("bounded-int-utils")]
            use core::internal::bounded_int::{self, BoundedInt, div_rem, DivRemHelper};
            use core::internal::OptionRev;
            extern fn bounded_int_wrap_non_zero<T>(v: T) -> NonZero<T> nopanic;


            impl Helper_u8_u8 of DivRemHelper<u8, u8> {
                type DivT = BoundedInt<0, 255>;
                type RemT = BoundedInt<0, 254>;
            }
            fn test_u8(a: felt252, b: felt252) -> (felt252, felt252) {
                let a_int: u8 = a.try_into().unwrap();
                let b_int: u8 = b.try_into().unwrap();
                let b_nz: NonZero<u8> = b_int.try_into().unwrap();
                let (q, r) = div_rem(a_int, b_nz);
                return (q.into(), r.into());
            }

            impl Helper_10_100_10_40 of DivRemHelper<BoundedInt<10, 100>, BoundedInt<10, 40>> {
                type DivT = BoundedInt<0, 10>;
                type RemT = BoundedInt<0, 39>;
            }
            fn test_10_100_10_40(a: felt252, b: felt252) -> (felt252, felt252) {
                let a_int: BoundedInt<10, 100> = a.try_into().unwrap();
                let b_int: BoundedInt<10, 40> = b.try_into().unwrap();
                let (q, r) = div_rem(a_int, bounded_int_wrap_non_zero(b_int));
                return (q.into(), r.into());
            }

            impl Helper_50_100_20_40 of DivRemHelper<BoundedInt<50, 100>, BoundedInt<20, 40>> {
                type DivT = BoundedInt<1, 5>;
                type RemT = BoundedInt<0, 39>;
            }
            fn test_50_100_20_40(a: felt252, b: felt252) -> (felt252, felt252) {
                let a_int: BoundedInt<50, 100> = a.try_into().unwrap();
                let b_int: BoundedInt<20, 40> = b.try_into().unwrap();
                let (q, r) = div_rem(a_int, bounded_int_wrap_non_zero(b_int));
                return (q.into(), r.into());
            }
        };
    }

    #[test_case("test_u8", 100, 30, 3, 10)]
    #[test_case("test_10_100_10_40", 100, 30, 3, 10)]
    #[test_case("test_50_100_20_40", 100, 30, 3, 10)]
    fn test_div_rem(entry_point: &str, a: i32, b: i32, expected_q: u32, expected_r: u32) {
        let arguments = &[Felt252::from(a).into(), Felt252::from(b).into()];
        let expected_result = jit_enum!(
            0,
            jit_struct!(jit_struct!(
                Felt252::from(expected_q).into(),
                Felt252::from(expected_r).into(),
            ))
        );
        run_program_assert_output(
            &TEST_DIV_REM_PROGRAM,
            entry_point,
            arguments,
            expected_result,
        );
    }
}
