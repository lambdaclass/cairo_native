////! # Non-zero unwrapping libfuncs
//! # Non-zero unwrapping libfuncs
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

///// Generate MLIR operations for the `unwrap_non_zero` libfunc.
/// Generate MLIR operations for the `unwrap_non_zero` libfunc.
//pub fn build<'ctx, 'this>(
pub fn build<'ctx, 'this>(
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
