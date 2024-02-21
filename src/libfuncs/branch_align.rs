//! # Branch alignment libfunc
//!
//! Natively compiled code doesn't need branch alignment because it has no notion of segments.
//! Because of this, this libfunc is a no-op.

use super::LibfuncHelper;
use crate::{error::builders::Result, metadata::MetadataStorage};
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

/// Generate MLIR operations for the `branch_align` libfunc.
pub fn build<'ctx, 'this>(
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
