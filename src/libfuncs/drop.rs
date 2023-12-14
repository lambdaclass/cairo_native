//! # `AP` tracking libfuncs
//!
//! Most types are trivial and don't need dropping (or rather, they will be dropped automatically
//! by MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need manual dropping.

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

/// Generate MLIR operations for the `drop` libfunc.
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
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    // TODO: Implement drop for arrays.

    let ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    ty.build_drop(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}
