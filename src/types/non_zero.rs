////! # Non-zero type
//! # Non-zero type
////!
//!
////! The non-zero generic type guarantees that whatever value it has is not zero.
//! The non-zero generic type guarantees that whatever value it has is not zero.
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
////! pub struct NonZero<T>(pub T);
//! pub struct NonZero<T>(pub T);
////! ```
//! ```
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
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
//    registry.build_type(context, module, registry, metadata, &info.ty)
    registry.build_type(context, module, registry, metadata, &info.ty)
//}
}
