//! # Branch alignment libfunc
//!
//! Natively compiled code doesn't need branch alignment because it has no notion of segments.
//! Because of this, this libfunc is a no-op.

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        coupon::CouponConcreteLibfunc,
        function_call::SignatureAndFunctionConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `coupon` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CouponConcreteLibfunc,
) -> Result<()> {
    match selector {
        CouponConcreteLibfunc::Buy(info) => {
            build_buy(context, registry, entry, location, helper, metadata, info)
        }
        CouponConcreteLibfunc::Refund(info) => {
            build_refund(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `coupon` libfunc.
pub fn build_buy<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    // In the future if the gas cost is required, this is how to get it.
    // let gas = metadata.get::<GasMetadata>().ok_or(Error::MissingMetadata)?;
    // let gas_cost = gas.initial_required_gas(&info.function.id);
    let ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let coupon = entry.append_op_result(llvm::undef(ty, location))?;

    entry.append_operation(helper.br(0, &[coupon], location));

    Ok(())
}

/// Generate MLIR operations for the `coupon` libfunc.
pub fn build_refund<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    // In the future if the gas cost is required, this is how to get it.
    // let gas = metadata.get::<GasMetadata>().ok_or(Error::MissingMetadata)?;
    // let gas_cost = gas.initial_required_gas(&info.function.id);

    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}
