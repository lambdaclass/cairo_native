////! # Memory-related libfuncs
//! # Memory-related libfuncs
////!
//!
////! Natively compiled code doesn't need this kind of memory tracking because it has no notion of the
//! Natively compiled code doesn't need this kind of memory tracking because it has no notion of the
////! segments. Because of this, all of the memory-related libfuncs here are no-ops.
//! segments. Because of this, all of the memory-related libfuncs here are no-ops.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{
use crate::{
//    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
//        mem::MemConcreteLibfunc,
        mem::MemConcreteLibfunc,
//        ConcreteLibfunc,
        ConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::llvm,
    dialect::llvm,
//    ir::{Block, Location},
    ir::{Block, Location},
//    Context,
    Context,
//};
};
//

///// Select and call the correct libfunc builder function from the selector.
/// Select and call the correct libfunc builder function from the selector.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    selector: &MemConcreteLibfunc,
    selector: &MemConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    match selector {
    match selector {
//        MemConcreteLibfunc::StoreTemp(info) => {
        MemConcreteLibfunc::StoreTemp(info) => {
//            build_store_temp(context, registry, entry, location, helper, info)
            build_store_temp(context, registry, entry, location, helper, info)
//        }
        }
//        MemConcreteLibfunc::StoreLocal(info) => {
        MemConcreteLibfunc::StoreLocal(info) => {
//            build_store_local(context, registry, entry, location, helper, metadata, info)
            build_store_local(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        MemConcreteLibfunc::FinalizeLocals(info) => {
        MemConcreteLibfunc::FinalizeLocals(info) => {
//            build_finalize_locals(context, registry, entry, location, helper, info)
            build_finalize_locals(context, registry, entry, location, helper, info)
//        }
        }
//        MemConcreteLibfunc::AllocLocal(info) => {
        MemConcreteLibfunc::AllocLocal(info) => {
//            build_alloc_local(context, registry, entry, location, helper, metadata, info)
            build_alloc_local(context, registry, entry, location, helper, metadata, info)
//        }
        }
//        MemConcreteLibfunc::Rename(info) => {
        MemConcreteLibfunc::Rename(info) => {
//            build_rename(context, registry, entry, location, helper, info)
            build_rename(context, registry, entry, location, helper, info)
//        }
        }
//    }
    }
//}
}
//

///// Generate MLIR operations for the `store_local` libfunc.
/// Generate MLIR operations for the `store_local` libfunc.
//pub fn build_store_local<'ctx, 'this>(
pub fn build_store_local<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureAndTypeConcreteLibfunc,
    _info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[entry.argument(1)?.into()], location));
    entry.append_operation(helper.br(0, &[entry.argument(1)?.into()], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `alloc_local` libfunc.
/// Generate MLIR operations for the `alloc_local` libfunc.
//pub fn build_alloc_local<'ctx, 'this>(
pub fn build_alloc_local<'ctx, 'this>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: &SignatureAndTypeConcreteLibfunc,
    info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    let target_type = registry.build_type(
    let target_type = registry.build_type(
//        context,
        context,
//        helper,
        helper,
//        registry,
        registry,
//        metadata,
        metadata,
//        &info.branch_signatures()[0].vars[0].ty,
        &info.branch_signatures()[0].vars[0].ty,
//    )?;
    )?;
//

//    let value_undef = entry.append_op_result(llvm::undef(target_type, location))?;
    let value_undef = entry.append_op_result(llvm::undef(target_type, location))?;
//

//    entry.append_operation(helper.br(0, &[value_undef], location));
    entry.append_operation(helper.br(0, &[value_undef], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `finalize_locals` libfunc.
/// Generate MLIR operations for the `finalize_locals` libfunc.
//pub fn build_finalize_locals<'ctx, 'this>(
pub fn build_finalize_locals<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `store_temp` libfunc.
/// Generate MLIR operations for the `store_temp` libfunc.
//pub fn build_store_temp<'ctx, 'this>(
pub fn build_store_temp<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureAndTypeConcreteLibfunc,
    _info: &SignatureAndTypeConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
//

//    Ok(())
    Ok(())
//}
}
//

///// Generate MLIR operations for the `rename` libfunc.
/// Generate MLIR operations for the `rename` libfunc.
//pub fn build_rename<'ctx, 'this>(
pub fn build_rename<'ctx, 'this>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &'this Block<'ctx>,
    entry: &'this Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, 'this>,
    helper: &LibfuncHelper<'ctx, 'this>,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
    entry.append_operation(helper.br(0, &[entry.argument(0)?.into()], location));
//

//    Ok(())
    Ok(())
//}
}
