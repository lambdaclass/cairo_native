//! # Snapshot taking libfuncs

use super::LibfuncHelper;
use crate::{
    error::builders::Result,
    metadata::{snapshot_clones::SnapshotClonesMeta, MetadataStorage},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `snapshot_take` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Handle non-trivially-copyable types (ex. arrays) by invoking their override or just copy the
    // original value otherwise.
    let original_value = entry.argument(0)?.into();
    let cloned_value = match metadata
        .get_mut::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.signature.param_signatures[0].ty))
    {
        Some(invoke_fn) => invoke_fn(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            original_value,
        )?,
        None => original_value,
    };

    entry.append_operation(helper.br(0, &[original_value, cloned_value], location));

    Ok(())
}
