//! # Unsigned 8-bit integer type

use super::WithSelf;
use crate::{error::builders::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
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
    _info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    Ok(IntegerType::new(context, 8).into())
}
