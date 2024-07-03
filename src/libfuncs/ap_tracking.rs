//! # `AP` tracking libfuncs
//!
//! Natively compiled code doesn't need `AP` tracking because it has no notion of the `AP` pointer.
//! Because of this, all `AP`-related libfuncs are no-ops.

use super::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        ap_tracking::ApTrackingConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
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
    selector: &ApTrackingConcreteLibfunc,
) -> Result<()> {
    match selector {
        ApTrackingConcreteLibfunc::Revoke(info) => {
            build_revoke(context, registry, entry, location, helper, metadata, info)
        }
        ApTrackingConcreteLibfunc::Enable(info) => {
            build_enable(context, registry, entry, location, helper, metadata, info)
        }
        ApTrackingConcreteLibfunc::Disable(info) => {
            build_disable(context, registry, entry, location, helper, metadata, info)
        }
    }
}

// PLT: maybe just inline these.
/// Generate MLIR operations for the `enable_ap_tracking` libfunc.
pub fn build_enable<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}

/// Generate MLIR operations for the `disable_ap_tracking` libfunc.
pub fn build_disable<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}

/// Generate MLIR operations for the `revoke_ap_tracking.` libfunc.
pub fn build_revoke<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}
// PLT: ACK
