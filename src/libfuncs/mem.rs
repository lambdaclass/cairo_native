//! # Memory-related libfuncs
//!
//! Natively compiled code doesn't need this kind of memory tracking because it has no notion of the
//! segments. Because of this, all of the memory-related libfuncs here are no-ops.

use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        mem::MemConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &MemConcreteLibfunc,
) -> Result<()> {
    match selector {
        MemConcreteLibfunc::StoreTemp(info) => {
            build_store_temp(context, registry, entry, location, helper, info)
        }
        MemConcreteLibfunc::StoreLocal(info) => {
            build_store_local(context, registry, entry, location, helper, metadata, info)
        }
        MemConcreteLibfunc::FinalizeLocals(info) => {
            build_finalize_locals(context, registry, entry, location, helper, info)
        }
        MemConcreteLibfunc::AllocLocal(info) => {
            build_alloc_local(context, registry, entry, location, helper, metadata, info)
        }
        MemConcreteLibfunc::Rename(info) => {
            build_rename(context, registry, entry, location, helper, info)
        }
    }
}

/// Generate MLIR operations for the `store_local` libfunc.
pub fn build_store_local<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[entry.argument(1)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `alloc_local` libfunc.
pub fn build_alloc_local<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let target_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let op = entry.append_operation(llvm::undef(target_type, location));
    let value_undef = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[value_undef], location));

    Ok(())
}

/// Generate MLIR operations for the `finalize_locals` libfunc.
pub fn build_finalize_locals<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[], location));
    Ok(())
}

/// Generate MLIR operations for the `store_temp` libfunc.
pub fn build_store_temp<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `rename` libfunc.
pub fn build_rename<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));

    Ok(())
}
