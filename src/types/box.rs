//! # Box type
//!
//! The type box for a given type `T`.
//!
//! ## Layout
//!
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! to the following:
//!
//! ```
//! #[repr(transparent)]
//! pub struct Box<T>(pub T);
//! ```

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
    dialect::llvm::r#type::opaque_pointer,
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
    Ok(opaque_pointer(context))
}
