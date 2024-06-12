////! # Elliptic curve state type
//! # Elliptic curve state type
//

//use super::WithSelf;
use super::WithSelf;
//use crate::{error::Result, metadata::MetadataStorage};
use crate::{error::Result, metadata::MetadataStorage};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
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
//    dialect::llvm,
    dialect::llvm,
//    ir::{r#type::IntegerType, Module, Type},
    ir::{r#type::IntegerType, Module, Type},
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
//    _module: &Module<'ctx>,
    _module: &Module<'ctx>,
//    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    _metadata: &mut MetadataStorage,
    _metadata: &mut MetadataStorage,
//    _info: WithSelf<InfoOnlyConcreteType>,
    _info: WithSelf<InfoOnlyConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    let felt252_ty = IntegerType::new(context, 252).into();
    let felt252_ty = IntegerType::new(context, 252).into();
//

//    Ok(llvm::r#type::r#struct(
    Ok(llvm::r#type::r#struct(
//        context,
        context,
//        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
//        false,
        false,
//    ))
    ))
//}
}
