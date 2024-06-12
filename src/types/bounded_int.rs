////! # `BoundedInt` type
//! # `BoundedInt` type
////!
//!
////! A `BoundedInt` is a int with a lower and high bound.
//! A `BoundedInt` is a int with a lower and high bound.
//

//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        bounded_int::BoundedIntConcreteType,
        bounded_int::BoundedIntConcreteType,
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        types::InfoOnlyConcreteType,
        types::InfoOnlyConcreteType,
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

//use super::WithSelf;
use super::WithSelf;
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
//    info: WithSelf<BoundedIntConcreteType>,
    info: WithSelf<BoundedIntConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    // todo: possible optimization, we may be able to use less bits depending on the possible values within the range.
    // todo: possible optimization, we may be able to use less bits depending on the possible values within the range.
//

//    let info = WithSelf {
    let info = WithSelf {
//        self_ty: info.self_ty,
        self_ty: info.self_ty,
//        inner: &InfoOnlyConcreteType {
        inner: &InfoOnlyConcreteType {
//            info: info.info.clone(),
            info: info.info.clone(),
//        },
        },
//    };
    };
//

//    super::felt252::build(context, module, registry, metadata, info)
    super::felt252::build(context, module, registry, metadata, info)
//}
}
