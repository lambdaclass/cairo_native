//! # `Felt` dictionary type
//!
//! A key value storage for values whose type implement Copy. The key is always a felt.
//!
//! This type is represented as a pointer to a heap allocated Rust hashmap, interacted through the runtime functions to
//! insert and get elements.

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
    Ok(llvm::r#type::opaque_pointer(context))
}
