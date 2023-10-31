//! # Snapshot taking libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::{snapshot_clones::SnapshotClonesMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{lib_func::SignatureOnlyConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `snapshot_take` libfunc.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: 'static + GenericType,
    TLibfunc: 'static + GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // TODO: Should this act like dup like it does now? or are there special requirements. So far it seems to work.
    // TODO: Handle non-trivially-copyable types (ex. arrays) and maybe update docs.

    let cloned_value = match metadata
        .get_mut::<SnapshotClonesMeta<TType, TLibfunc>>()
        .and_then(|meta| meta.wrap_invoke(&info.signature.param_signatures[0].ty))
    {
        Some(invoke_fn) => invoke_fn(context, registry, entry, location, helper, metadata)?,
        None => entry.argument(0)?.into(),
    };

    entry.append_operation(helper.br(0, &[entry.argument(0)?.into(), cloned_value], location));

    Ok(())
}
