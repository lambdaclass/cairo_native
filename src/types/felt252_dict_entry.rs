////! # `Felt` dictionary entry type
//! # `Felt` dictionary entry type
////!
//!
////! The entry type returning when getting a value from a dictionary.
//! The entry type returning when getting a value from a dictionary.
////!
//!
////! It is represented as the following struct:
//! It is represented as the following struct:
////!
//!
////! | Index | Type           | Description                      |
//! | Index | Type           | Description                      |
////! | ----- | -------------- | -------------------------------- |
//! | ----- | -------------- | -------------------------------- |
////! |   0   | `i252`         | The entry key.                   |
//! |   0   | `i252`         | The entry key.                   |
////! |   1   | `!llvm.ptr`    | Pointer to the entry value.      |
//! |   1   | `!llvm.ptr`    | Pointer to the entry value.      |
////! |   2   | `!llvm.ptr`    | Pointer to the dictionary (rust) |
//! |   2   | `!llvm.ptr`    | Pointer to the dictionary (rust) |
////!
//!
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
//    _info: WithSelf<InfoAndTypeConcreteType>,
    _info: WithSelf<InfoAndTypeConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    Ok(llvm::r#type::r#struct(
    Ok(llvm::r#type::r#struct(
//        context,
        context,
//        &[
        &[
//            IntegerType::new(context, 252).into(), // entry key
            IntegerType::new(context, 252).into(), // entry key
//            llvm::r#type::pointer(context, 0),     // value ptr
            llvm::r#type::pointer(context, 0),     // value ptr
//            llvm::r#type::pointer(context, 0),     // dict ptr
            llvm::r#type::pointer(context, 0),     // dict ptr
//        ],
        ],
//        false,
        false,
//    ))
    ))
//}
}
