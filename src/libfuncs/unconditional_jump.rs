//! # Unconditional jump libfunc

use super::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage};
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

/// Generate MLIR operations for the `jump` libfunc.
pub fn build<'ctx>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[], location));

    Ok(())
}
// PLT: is this implemented?
// PLT: ACK
