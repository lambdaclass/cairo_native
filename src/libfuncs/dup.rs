//! # State value duplication libfunc
//!
//! Most types are trivial and don't need any clone (or rather, they will be cloned automatically by
//! MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need special handling.

use super::LibfuncHelper;
use crate::{error::libfuncs::Result, metadata::MetadataStorage};
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

/// Generate MLIR operations for the `dup` libfunc.
pub fn build<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Note: All non-trivially-copyable are automatically handled by the cairo compiler (to Sierra).
    //   In other words, this function will only be called for copyable types.
    //
    //   Proof: The following code will fail in Cairo with an unsupported generic argument:
    //   `dup(ArrayTrait::<u8>::new())`.

    entry.append_operation(helper.br(
        0,
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));

    Ok(())
}
