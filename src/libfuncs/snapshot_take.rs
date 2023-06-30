//! # Snapshot taking libfuncs
//!
//! TODO

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
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
    _context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // TODO: Should this act like dup like it does now? or are there special requirements. So far it seems to work.
    // TODO: Handle non-trivially-copyable types (ex. arrays) and maybe update docs.

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));

    Ok(())
}
