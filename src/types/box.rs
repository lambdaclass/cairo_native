//! # Box type
//!
//! The type box for a given type `T`.
//!
//! ## Layout
//!
//! It's just a pointer to the heap-allocated data. The pointer cannot be null, it must always have
//! a value. For null-compatible boxes, check out [nullables](crate::types::nullable).

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
