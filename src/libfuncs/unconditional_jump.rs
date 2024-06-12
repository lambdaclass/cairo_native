////! # Unconditional jump libfunc
//! # Unconditional jump libfunc
//

//use super::LibfuncHelper;
use super::LibfuncHelper;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
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

///// Generate MLIR operations for the `jump` libfunc.
/// Generate MLIR operations for the `jump` libfunc.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    _context: &'ctx Context,
    _context: &'ctx Context,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    entry: &Block<'ctx>,
    entry: &Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    helper: &LibfuncHelper<'ctx, '_>,
    helper: &LibfuncHelper<'ctx, '_>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: &SignatureOnlyConcreteLibfunc,
    _info: &SignatureOnlyConcreteLibfunc,
//) -> Result<()> {
) -> Result<()> {
//    entry.append_operation(helper.br(0, &[], location));
    entry.append_operation(helper.br(0, &[], location));
//

//    Ok(())
    Ok(())
//}
}
