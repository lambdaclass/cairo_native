//! # `BoundedInt` type
//!
//! A `BoundedInt` is a int with a lower and high bound.

use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::BoundedIntConcreteType,
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    Context,
};

use super::WithSelf;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<BoundedIntConcreteType>,
) -> Result<Type<'ctx>> {
    // todo: possible optimization, we may be able to use less bits depending on the possible values within the range.

    let info = WithSelf {
        self_ty: info.self_ty,
        inner: &InfoOnlyConcreteType {
            info: info.info.clone(),
        },
    };

    super::felt252::build(context, module, registry, metadata, info)
}
