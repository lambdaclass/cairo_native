////! # `AP` tracking libfuncs
//! # `AP` tracking libfuncs
////!
//!
////! Most types are trivial and don't need dropping (or rather, they will be dropped automatically
//! Most types are trivial and don't need dropping (or rather, they will be dropped automatically
////! by MLIR). For those types, this libfunc is a no-op.
//! by MLIR). For those types, this libfunc is a no-op.
////!
//!
////! However, types like an array need manual dropping.
//! However, types like an array need manual dropping.
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{error::Result, metadata::MetadataStorage, types::TypeBuilder};
use crate::{error::Result, metadata::MetadataStorage, types::TypeBuilder};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        lib_func::SignatureOnlyConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    ir::{Block, Location},
    ir::{Block, Location},
//    Context,
    Context,
//};
};
//

///// Generate MLIR operations for the `drop` libfunc.
/// Generate MLIR operations for the `drop` libfunc.
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
//    info: &SignatureOnlyConcreteLibfunc,
    info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    // Note: Complex types implement drop within the type itself (in `build_drop`).
    // Note: Complex types implement drop within the type itself (in `build_drop`).
//

//    let ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
//    ty.build_drop(
    ty.build_drop(
//        context,
        context,
//        registry,
        registry,
//        entry,
        entry,
//        location,
        location,
//        helper,
        helper,
//        metadata,
        metadata,
//        &info.signature.param_signatures[0].ty,
        &info.signature.param_signatures[0].ty,
//    )?;
    )?;
//

//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
