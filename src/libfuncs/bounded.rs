//! # Bounded int libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt,
    error::{Error, Result, SierraAssertError},
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::{
            BoundedIntConcreteLibfunc, BoundedIntConstrainConcreteLibfunc,
            BoundedIntDivRemConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, ConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{Block, Location, Value, ValueLike},
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
    selector: &BoundedIntConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoundedIntConcreteLibfunc::Add(info) => {
            build_bounded_int_add(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Sub(info) => {
            build_bounded_int_sub(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Mul(info) => {
            build_bounded_int_mul(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::DivRem(info) => {
            build_bounded_int_divrem(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Constrain(info) => {
            build_bounded_int_constrain(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::IsZero(info) => {
            build_bounded_int_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::WrapNonZero(info) => build_bounded_int_wrap_non_zero(
            context, registry, entry, location, helper, metadata, info,
        ),
    }
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_add<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let mut lhs = entry.argument(0)?.into();
    let mut rhs = entry.argument(1)?.into();

    let lhs_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let rhs_type = registry.get_type(&info.param_signatures()[1].ty)?;

    let lhs_is_signed = lhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;
    let rhs_is_signed = rhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[0][0])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;

    if lhs_type.integer_width() < dst_type.integer_width() {
        if lhs_is_signed {
            lhs = entry.append_op_result(arith::extsi(lhs, dst_ty, location))?;
        } else {
            lhs = entry.append_op_result(arith::extui(lhs, dst_ty, location))?;
        }
    }

    if rhs_type.integer_width() < dst_type.integer_width() {
        if rhs_is_signed {
            rhs = entry.append_op_result(arith::extsi(rhs, dst_ty, location))?;
        } else {
            rhs = entry.append_op_result(arith::extui(rhs, dst_ty, location))?;
        }
    }

    let value = entry.append_op_result(arith::addi(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_sub<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // They have felt-width already, and the libfunc is non branching, so it cant fail.
    let mut lhs = entry.argument(0)?.into();
    let mut rhs = entry.argument(1)?.into();

    let lhs_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let rhs_type = registry.get_type(&info.param_signatures()[1].ty)?;

    let lhs_is_signed = lhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;
    let rhs_is_signed = rhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[0][0])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;

    if lhs_type.integer_width() < dst_type.integer_width() {
        if lhs_is_signed {
            lhs = entry.append_op_result(arith::extsi(lhs, dst_ty, location))?;
        } else {
            lhs = entry.append_op_result(arith::extui(lhs, dst_ty, location))?;
        }
    }

    if rhs_type.integer_width() < dst_type.integer_width() {
        if rhs_is_signed {
            rhs = entry.append_op_result(arith::extsi(rhs, dst_ty, location))?;
        } else {
            rhs = entry.append_op_result(arith::extui(rhs, dst_ty, location))?;
        }
    }

    let value = entry.append_op_result(arith::subi(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // They have felt-width already, and the libfunc is non branching, so it cant fail.
    let mut lhs = entry.argument(0)?.into();
    let mut rhs = entry.argument(1)?.into();

    let lhs_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let rhs_type = registry.get_type(&info.param_signatures()[1].ty)?;

    let lhs_is_signed = lhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;
    let rhs_is_signed = rhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[0][0])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;

    if lhs_type.integer_width() < dst_type.integer_width() {
        if lhs_is_signed {
            lhs = entry.append_op_result(arith::extsi(lhs, dst_ty, location))?;
        } else {
            lhs = entry.append_op_result(arith::extui(lhs, dst_ty, location))?;
        }
    }

    if rhs_type.integer_width() < dst_type.integer_width() {
        if rhs_is_signed {
            rhs = entry.append_op_result(arith::extsi(rhs, dst_ty, location))?;
        } else {
            rhs = entry.append_op_result(arith::extui(rhs, dst_ty, location))?;
        }
    }

    let value = entry.append_op_result(arith::muli(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_divrem` libfunc.
/// Libfunc for dividing two non negative BoundedInts and getting the quotient and remainder.
pub fn build_bounded_int_divrem<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &BoundedIntDivRemConcreteLibfunc,
) -> Result<()> {
    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let mut lhs = entry.argument(1)?.into();
    let mut rhs = entry.argument(2)?.into();

    let lhs_type = registry.get_type(&info.param_signatures()[1].ty)?;
    let rhs_type = registry.get_type(&info.param_signatures()[2].ty)?;

    let lhs_is_signed = lhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let rhs_is_signed = rhs_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[0][1])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][1],
    )?;

    if lhs_type.integer_width() < dst_type.integer_width() {
        if lhs_is_signed {
            lhs = entry.append_op_result(arith::extsi(lhs, dst_ty, location))?;
        } else {
            lhs = entry.append_op_result(arith::extui(lhs, dst_ty, location))?;
        }
    }

    if rhs_type.integer_width() < dst_type.integer_width() {
        if rhs_is_signed {
            rhs = entry.append_op_result(arith::extsi(rhs, dst_ty, location))?;
        } else {
            rhs = entry.append_op_result(arith::extui(rhs, dst_ty, location))?;
        }
    }

    let result_div = entry.append_op_result(arith::divui(lhs, rhs, location))?;
    let result_rem = entry.append_op_result(arith::remui(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[range_check, result_div, result_rem], location));
    Ok(())
}

/// Generate MLIR operations for the `u64_is_zero` libfunc.
pub fn build_bounded_int_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let mut value: Value = entry.argument(0)?.into();

    let value_type = registry.get_type(&info.param_signatures()[0].ty)?;

    let value_is_signed = value_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[1][0])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[1][0],
    )?;

    if value_type.integer_width() < dst_type.integer_width() {
        if value_is_signed {
            value = entry.append_op_result(arith::extsi(value, dst_ty, location))?;
        } else {
            value = entry.append_op_result(arith::extui(value, dst_ty, location))?;
        }
    }

    let const_0 = entry.const_int_from_type(context, location, 0, value.r#type())?;

    let condition = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        value,
        const_0,
        location,
    ))?;

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[value]], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_wrap_non_zero` libfunc.
pub fn build_bounded_int_wrap_non_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let mut value: Value = entry.argument(0)?.into();

    let value_type = registry.get_type(&info.param_signatures()[0].ty)?;

    let value_is_signed = value_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    let dst_type = registry.get_type(&info.output_types()[0][0])?;
    let dst_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;

    if value_type.integer_width() < dst_type.integer_width() {
        if value_is_signed {
            value = entry.append_op_result(arith::extsi(value, dst_ty, location))?;
        } else {
            value = entry.append_op_result(arith::extui(value, dst_ty, location))?;
        }
    }

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_constrain` libfunc.
pub fn build_bounded_int_constrain<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntConstrainConcreteLibfunc,
) -> Result<()> {
    let range_check: Value = entry.argument(0)?.into();
    let mut value: Value = entry.argument(1)?.into();

    let value_type = registry.get_type(&info.param_signatures()[1].ty)?;

    let const_boundary = entry.const_int(context, location, info.boundary.clone(), 252)?;

    let is_signed = value_type
        .is_integer_signed(registry)
        .ok_or_else(|| Error::SierraAssert(SierraAssertError::Cast))?;

    if value_type.integer_width() < Some(252) {
        if is_signed {
            value =
                entry.append_op_result(arith::extsi(value, const_boundary.r#type(), location))?;
        } else {
            value =
                entry.append_op_result(arith::extui(value, const_boundary.r#type(), location))?;
        }
    }

    let condition = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ult,
        value,
        const_boundary,
        location,
    ))?;

    entry.append_operation(helper.cond_br(
        context,
        condition,
        [0, 1],
        [&[range_check, value], &[range_check, value]],
        location,
    ));

    Ok(())
}
