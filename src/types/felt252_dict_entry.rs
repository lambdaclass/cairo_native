//! # `Felt` dictionary entry type
//!
//! The entry type returning when getting a value from a dictionary.
//!
//! It is represented as the following struct:
//!
//! | Index | Type           | Description                                  |
//! | ----- | -------------- | -------------------------------------------- |
//! |   0   | `!llvm.ptr`    | Pointer to the dictionary (Rust).            |
//! |   1   | `!llvm.ptr`    | Pointer to the entry's value pointer (Rust). |
//!

use super::WithSelf;
use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    // TODO: Custom drop (dup?) for dict entries.
    Ok(llvm::r#type::r#struct(
        context,
        &[
            llvm::r#type::pointer(context, 0), // dict ptr
            llvm::r#type::pointer(context, 0), // value ptr
        ],
        false,
    ))
}
