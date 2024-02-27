//! # `BoundedInt` type
//!
//! A `BoundedInt` is a int with a lower and high bound.

use crate::{error::types::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::BoundedIntConcreteType,
        core::{CoreLibfunc, CoreType},
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

use super::WithSelf;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<BoundedIntConcreteType>,
) -> Result<Type<'ctx>> {
    let bits = info.range.lower.bits().max(info.range.upper.bits()) + 1; // sign bit
    Ok(IntegerType::new(
        context,
        bits.try_into().expect("bits should always fit a u32"),
    )
    .into())
}
