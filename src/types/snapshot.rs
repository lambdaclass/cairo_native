////! # Snapshot type
//! # Snapshot type
////!
//!
////! The type snapshot for a given type `T`.
//! The type snapshot for a given type `T`.
////!
//!
////! ## Layout
//! ## Layout
////!
//!
////! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
//! Its layout is that of whatever it wraps. In other words, if it was Rust it would be equivalent
////! to the following:
//! to the following:
////!
//!
////! ```
//! ```
////! #[repr(transparent)]
//! #[repr(transparent)]
////! pub struct Snapshot<T>(pub T);
//! pub struct Snapshot<T>(pub T);
////! ```
//! ```
//

//use super::{TypeBuilder, WithSelf};
use super::{TypeBuilder, WithSelf};
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
//    utils::ProgramRegistryExt,
    utils::ProgramRegistryExt,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoAndTypeConcreteType,
        types::InfoAndTypeConcreteType,
//    },
    },
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    ir::{Module, Type},
    ir::{Module, Type},
//    Context,
    Context,
//};
};
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<InfoAndTypeConcreteType>,
    info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // This type is like a `Cow<T>` that clones whenever the original type is modified to keep the
    // This type is like a `Cow<T>` that clones whenever the original type is modified to keep the
//    // original data. Since implementing that is complicated we can just clone the entire value for
    // original data. Since implementing that is complicated we can just clone the entire value for
//    // now.
    // now.
//    match metadata.get_mut::<EnumSnapshotVariantsMeta>() {
    match metadata.get_mut::<EnumSnapshotVariantsMeta>() {
//        Some(x) => x,
        Some(x) => x,
//        None => metadata
        None => metadata
//            .insert(EnumSnapshotVariantsMeta::default())
            .insert(EnumSnapshotVariantsMeta::default())
//            .expect("should not fail because we checked there is no metadata beforehand"),
            .expect("should not fail because we checked there is no metadata beforehand"),
//    }
    }
//    .set_mapping(info.self_ty, registry.get_type(&info.ty)?.variants());
    .set_mapping(info.self_ty, registry.get_type(&info.ty)?.variants());
//

//    registry.build_type(context, module, registry, metadata, &info.ty)
    registry.build_type(context, module, registry, metadata, &info.ty)
//}
}
