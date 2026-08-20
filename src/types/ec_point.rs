//! # Elliptic curve point type
//!
//! An `EcPoint` is stored as the affine pair `(x, y)`. The point at infinity is
//! encoded as `(0, 0)`, which is unambiguous because no point on the STARK curve
//! has `y = 0`.

use super::WithSelf;
use crate::{error::Result, metadata::MetadataStorage};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoOnlyConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

/// Number of `felt252`s in the native representation of an `EcPoint`.
pub const NUM_FELTS: usize = 2;

/// The MLIR type of an `EcPoint`: `!llvm.struct<(i252, i252)>`.
pub fn ec_point_ty(context: &Context) -> Type<'_> {
    let felt252_ty = IntegerType::new(context, 252).into();
    llvm::r#type::r#struct(context, &[felt252_ty; NUM_FELTS], false)
}

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
    Ok(ec_point_ty(context))
}
