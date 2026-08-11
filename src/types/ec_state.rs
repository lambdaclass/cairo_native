//! # Elliptic curve state type
//!
//! An `EcState` accumulates a sum of curve points. It is stored in **projective**
//! coordinates `[X : Y : Z]`, representing the affine point `(X/Z, Y/Z)`.
//!
//! Projective coordinates let `ec_state_add` accumulate without ever dividing:
//! the modular inversion that converts back to affine happens once, in
//! `ec_state_try_finalize_nz`, instead of once per addition.
//!
//! Invariants, all of which the rest of the codebase depends on:
//!
//! - **`Z == 0` is the point at infinity**, and is the only test for it.
//!   [`starknet_types_core::curve::ProjectivePoint::is_identity`] must *not* be
//!   used: it is exact equality against `[0, 1, 0]` and so misses `[0, 5, 0]`.
//! - **`ec_state_init` emits the canonical identity `[0, 1, 0]`.**
//! - **Values in memory are not canonical.** `[X : Y : Z]` and `[λX : λY : λZ]`
//!   are the same point, and arithmetic freely produces either. Nothing may
//!   compare two `EcState`s bitwise, or use one as a dictionary key. This is
//!   safe today because Sierra has no `EcState` equality libfunc, `dup` is a
//!   plain SSA copy, and the coordinates are unobservable from Cairo.
//! - **The public [`crate::Value::EcState`] stays affine**, with `(0, 0)` for the
//!   point at infinity. Conversion happens only in `Value::to_ptr` /
//!   `Value::from_ptr` and in the argument encoder, all of which route through
//!   [`to_projective`].

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
use starknet_types_core::felt::Felt;

/// Number of `felt252`s in the native representation of an `EcState`.
pub const NUM_FELTS: usize = 3;

/// The MLIR type of an `EcState`: `!llvm.struct<(i252, i252, i252)>`.
pub fn ec_state_ty(context: &Context) -> Type<'_> {
    let felt252_ty = IntegerType::new(context, 252).into();
    llvm::r#type::r#struct(context, &[felt252_ty; NUM_FELTS], false)
}

/// Convert the affine `(x, y)` of a public [`crate::Value::EcState`] into the
/// projective triple stored natively.
///
/// The affine `(0, 0)` sentinel becomes the canonical identity `[0, 1, 0]`; every
/// other point becomes `[x, y, 1]`.
pub fn to_projective(x: Felt, y: Felt) -> [Felt; NUM_FELTS] {
    if x == Felt::ZERO && y == Felt::ZERO {
        [Felt::ZERO, Felt::ONE, Felt::ZERO]
    } else {
        [x, y, Felt::ONE]
    }
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
    Ok(ec_state_ty(context))
}
