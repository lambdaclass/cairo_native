//! # `Felt` dictionary entry type
//!
//! The entry type returning when getting a value from a dictionary.
//!
//! It is represented as the following struct:
//!
//! | Index | Type           | Description                      |
//! | ----- | -------------- | -------------------------------- |
//! |   0   | `i252`         | The entry key.                   |
//! |   1   | `!llvm.ptr`    | Pointer to the entry value.      |
//! |   2   | `!llvm.ptr`    | Pointer to the dictionary (rust) |

use super::WithSelf;
use crate::{error::builders::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
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
    Ok(llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(), // entry key
            llvm::r#type::opaque_pointer(context), // value ptr
            llvm::r#type::opaque_pointer(context), // dict ptr
        ],
        false,
    ))
}
