//! # Bounded int libfuncs

use super::LibfuncHelper;
use crate::{block_ext::BlockExt, error::Result, metadata::MetadataStorage};
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
            build_bounded_int_divrem(context, registry, entry, location, helper, info)
        }
        BoundedIntConcreteLibfunc::Constrain(info) => {
            build_bounded_int_constrain(context, registry, entry, location, helper, info)
        }
        BoundedIntConcreteLibfunc::IsZero(info) => {
            build_bounded_int_is_zero(context, registry, entry, location, helper, info)
        }
        BoundedIntConcreteLibfunc::WrapNonZero(info) => {
            build_bounded_int_wrap_non_zero(context, registry, entry, location, helper, info)
        }
    }
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_add<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // They have felt-width already, and the libfunc is non branching, so it cant fail.
    let lhs = entry.argument(0)?.into();
    let rhs = entry.argument(1)?.into();

    let value = entry.append_op_result(arith::addi(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_sub<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // They have felt-width already, and the libfunc is non branching, so it cant fail.
    let lhs = entry.argument(0)?.into();
    let rhs = entry.argument(1)?.into();

    let value = entry.append_op_result(arith::subi(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bounded_int_mul<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // They have felt-width already, and the libfunc is non branching, so it cant fail.
    let lhs = entry.argument(0)?.into();
    let rhs = entry.argument(1)?.into();

    let value = entry.append_op_result(arith::muli(lhs, rhs, location))?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_divrem` libfunc.
/// Libfunc for dividing two non negative BoundedInts and getting the quotient and remainder.
pub fn build_bounded_int_divrem<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &BoundedIntDivRemConcreteLibfunc,
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

/// Generate MLIR operations for the `u64_is_zero` libfunc.
pub fn build_bounded_int_is_zero<'ctx, 'this>(
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

/// Generate MLIR operations for the `bounded_int_wrap_non_zero` libfunc.
pub fn build_bounded_int_wrap_non_zero<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let arg0: Value = entry.argument(0)?.into();

    entry.append_operation(helper.br(0, &[arg0], location));

    Ok(())
}

/// Generate MLIR operations for the `bounded_int_constrain` libfunc.
pub fn build_bounded_int_constrain<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &BoundedIntConstrainConcreteLibfunc,
) -> Result<()> {
    let range_check: Value = entry.argument(0)?.into();
    let value: Value = entry.argument(1)?.into();

    let const_boundary = entry.const_int(context, location, info.boundary.clone(), 252)?;

    let condition = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ult,
        value,
        const_boundary,
        location,
    ))?;

    let p = registry.get_type(&info.param_signatures()[1].ty)?;
    let x = registry.get_type(&info.output_types()[0][1])?;
    let y = registry.get_type(&info.output_types()[0][1])?;
    dbg!(&p.info().long_id.to_string());
    dbg!(&x.info().long_id.to_string());
    dbg!(&y.info().long_id.to_string());

    entry.append_operation(helper.cond_br(
        context,
        condition,
        [0, 1],
        [&[range_check, value], &[range_check, value]],
        location,
    ));

    Ok(())
}
