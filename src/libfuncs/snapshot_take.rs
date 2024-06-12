////! # Snapshot taking libfuncs
//! # Snapshot taking libfuncs
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::{snapshot_clones::SnapshotClonesMeta, MetadataStorage},
    metadata::{snapshot_clones::SnapshotClonesMeta, MetadataStorage},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
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

///// Generate MLIR operations for the `snapshot_take` libfunc.
/// Generate MLIR operations for the `snapshot_take` libfunc.
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    // Handle non-trivially-copyable types (ex. arrays) by invoking their override or just copy the
    // Handle non-trivially-copyable types (ex. arrays) by invoking their override or just copy the
//    // original value otherwise.
    // original value otherwise.
//    let original_value = entry.argument(0)?.into();
    let original_value = entry.argument(0)?.into();
//    let (entry, cloned_value) = match metadata
    let (entry, cloned_value) = match metadata
//        .get_mut::<SnapshotClonesMeta>()
        .get_mut::<SnapshotClonesMeta>()
//        .and_then(|meta| meta.wrap_invoke(&info.signature.param_signatures[0].ty))
        .and_then(|meta| meta.wrap_invoke(&info.signature.param_signatures[0].ty))
//    {
    {
//        Some(invoke_fn) => invoke_fn(
        Some(invoke_fn) => invoke_fn(
//            context,
            context,
//            registry,
            registry,
//            entry,
            entry,
//            location,
            location,
//            helper,
            helper,
//            metadata,
            metadata,
//            original_value,
            original_value,
//        )?,
        )?,
//        None => (entry, original_value),
        None => (entry, original_value),
//    };
    };
//

//    entry.append_operation(helper.br(0, &[original_value, cloned_value], location));
    entry.append_operation(helper.br(0, &[original_value, cloned_value], location));
//

//    Ok(())
    Ok(())
//}
}
