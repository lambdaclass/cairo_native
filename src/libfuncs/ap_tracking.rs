////! # `AP` tracking libfuncs
//! # `AP` tracking libfuncs
////!
//!
////! Natively compiled code doesn't need `AP` tracking because it has no notion of the `AP` pointer.
//! Natively compiled code doesn't need `AP` tracking because it has no notion of the `AP` pointer.
////! Because of this, all `AP`-related libfuncs are no-ops.
//! Because of this, all `AP`-related libfuncs are no-ops.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        ap_tracking::ApTrackingConcreteLibfunc,
        ap_tracking::ApTrackingConcreteLibfunc,
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    ir::{Block, Location},
    ir::{Block, Location},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &ApTrackingConcreteLibfunc,
    selector: &ApTrackingConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        ApTrackingConcreteLibfunc::Revoke(info) => {
        ApTrackingConcreteLibfunc::Revoke(info) => {
//            build_revoke(context, registry, entry, location, helper, metadata, info)
            build_revoke(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ApTrackingConcreteLibfunc::Enable(info) => {
        ApTrackingConcreteLibfunc::Enable(info) => {
//            build_enable(context, registry, entry, location, helper, metadata, info)
            build_enable(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        ApTrackingConcreteLibfunc::Disable(info) => {
        ApTrackingConcreteLibfunc::Disable(info) => {
//            build_disable(context, registry, entry, location, helper, metadata, info)
            build_disable(context, registry, entry, location, helper, metadata, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `enable_ap_tracking` libfunc.
/// Generate MLIR operations for the `enable_ap_tracking` libfunc.
//pub fn build_enable<'ctx, 'this>(
pub fn build_enable<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `disable_ap_tracking` libfunc.
/// Generate MLIR operations for the `disable_ap_tracking` libfunc.
//pub fn build_disable<'ctx, 'this>(
pub fn build_disable<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `revoke_ap_tracking.` libfunc.
/// Generate MLIR operations for the `revoke_ap_tracking.` libfunc.
//pub fn build_revoke<'ctx, 'this>(
pub fn build_revoke<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
