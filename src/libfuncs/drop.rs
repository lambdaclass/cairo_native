//! # `AP` tracking libfuncs
//!
//! Most types are trivial and don't need dropping (or rather, they will be dropped automatically
//! by MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need manual dropping.

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{drop_overrides::DropOverridesMeta, MetadataStorage},
    utils::ProgramRegistryExt,
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

/// Generate MLIR operations for the `drop` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    if let Some(drop_overrides_meta) = metadata.get::<DropOverridesMeta>() {
        drop_overrides_meta.invoke_override(
            context,
            entry,
            location,
            &info.signature.param_signatures[0].ty,
            entry.argument(0)?.into(),
        )?;
    }

    entry.append_operation(helper.br(0, &[], location));
    Ok(())
}
