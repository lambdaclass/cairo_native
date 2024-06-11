//! # State value duplication libfunc
//!
//! Most types are trivial and don't need any clone (or rather, they will be cloned automatically by
//! MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need special handling.

use super::LibfuncHelper;
use crate::{
    error::Result,
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

/// Generate MLIR operations for the `dup` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Note: The Cairo compiler will avoid using `dup` for some non-trivially-copyable types, but
    //   not all of them. For example, it'll not generate a clone implementation for `Box<T>`.
    //   That's why we need to check for clone implementations within the compiler.

    match metadata
        .get::<SnapshotClonesMeta>()
        .and_then(|meta| meta.wrap_invoke(&info.signature.param_signatures[0].ty))
    {
        Some(clone_fn) => {
            let (entry, cloned_value) = clone_fn(
                context,
                registry,
                entry,
                location,
                helper,
                metadata,
                entry.argument(0)?.into(),
            )?;

            entry.append_operation(helper.br(
                0,
                &[entry.argument(0)?.into(), cloned_value],
                location,
            ));
        }
        None => {
            entry.append_operation(helper.br(
                0,
                &[entry.argument(0)?.into(), entry.argument(0)?.into()],
                location,
            ));
        }
    }

    Ok(())
}
