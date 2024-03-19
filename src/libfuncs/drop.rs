//! # `AP` tracking libfuncs
//!
//! Most types are trivial and don't need dropping (or rather, they will be dropped automatically
//! by MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need manual dropping.

use super::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage, types::TypeBuilder};
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
    // Note: Complex types implement drop within the type itself (in `build_drop`).

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
