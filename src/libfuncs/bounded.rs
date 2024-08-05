//! # Bounded int libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        boolean::BoolConcreteLibfunc,
        bounded_int::{BoundedIntConcreteLibfunc, BoundedIntDivRemConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, Location, Value},
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
        BoundedIntConcreteLibfunc::Constrain(_) => todo!(),
        BoundedIntConcreteLibfunc::IsZero(_) => todo!(),
        BoundedIntConcreteLibfunc::WrapNonZero(_) => todo!(),
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
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
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
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
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
