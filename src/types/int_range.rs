//! # Int range of type T
//!
//! A range [x, y) where x <= y
//!
//! ## Layout
//!
//! A struct with 2 fields of type T
//!
//! ```
//! #[repr(transparent)]
//! pub struct NonZero<T>(pub T);
//! ```

use super::WithSelf;
use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    // TODO: Can its inner type require dup or drop? probably not since they are integers
    let inner = registry.build_type(context, module, registry, metadata, &info.ty)?;

    Ok(melior::dialect::llvm::r#type::r#struct(
        context,
        &[inner, inner],
        false,
    ))
}
