////! # State value duplication libfunc
//! # State value duplication libfunc
////!
//!
////! Most types are trivial and don't need any clone (or rather, they will be cloned automatically by
//! Most types are trivial and don't need any clone (or rather, they will be cloned automatically by
////! MLIR). For those types, this libfunc is a no-op.
//! MLIR). For those types, this libfunc is a no-op.
////!
//!
////! However, types like an array need special handling.
//! However, types like an array need special handling.
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

///// Generate MLIR operations for the `dup` libfunc.
/// Generate MLIR operations for the `dup` libfunc.
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
//    // Note: All non-trivially-copyable are automatically handled by the cairo compiler (to Sierra).
    // Note: All non-trivially-copyable are automatically handled by the cairo compiler (to Sierra).
//    //   In other words, this function will only be called for copyable types.
    //   In other words, this function will only be called for copyable types.
//    //
    //
//    //   Proof: The following code will fail in Cairo with an unsupported generic argument:
    //   Proof: The following code will fail in Cairo with an unsupported generic argument:
//    //   `dup(ArrayTrait::<u8>::new())`.
    //   `dup(ArrayTrait::<u8>::new())`.
//

//    entry.append_operation(helper.br(
    entry.append_operation(helper.br(
//        0,
        0,
//        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
//        location,
        location,
//    ));
    ));
//

//    Ok(())
    Ok(())
//}
}
