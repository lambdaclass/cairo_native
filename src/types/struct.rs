////! # Struct type
//! # Struct type
////!
//!
////! A struct is just a fixed collection of values that may have different types, which are known at
//! A struct is just a fixed collection of values that may have different types, which are known at
////! compile-time. Its fields are properly aligned and respect the declaration's field ordering.
//! compile-time. Its fields are properly aligned and respect the declaration's field ordering.
////!
//!
////! For example, the following struct would have a layout as described in the table below:
//! For example, the following struct would have a layout as described in the table below:
////!
//!
////! ```cairo
//! ```cairo
////! struct MyStruct {
//! struct MyStruct {
////!     U8: u8,
//!     U8: u8,
////!     U16: u16,
//!     U16: u16,
////!     U32: u32,
//!     U32: u32,
////!     U64: u64,
//!     U64: u64,
////!     Felt: Felt,
//!     Felt: Felt,
////! }
//! }
////! ```
//! ```
////!
//!
////! | Index | Type   | ABI (in Rust types) | Alignment | Size |
//! | Index | Type   | ABI (in Rust types) | Alignment | Size |
////! | ----- | ------ | ------------------- | --------- | ---- |
//! | ----- | ------ | ------------------- | --------- | ---- |
////! |   0   | `i8`   | `u8`                |         1 |    1 |
//! |   0   | `i8`   | `u8`                |         1 |    1 |
////! |  N/A  | N/A    | `[u8; 1]`           |         1 |    1 |
//! |  N/A  | N/A    | `[u8; 1]`           |         1 |    1 |
////! |   1   | `i16`  | `u16`               |         2 |    2 |
//! |   1   | `i16`  | `u16`               |         2 |    2 |
////! |  N/A  | N/A    | `[u8; 2]`           |         1 |    2 |
//! |  N/A  | N/A    | `[u8; 2]`           |         1 |    2 |
////! |   2   | `i32`  | `u32`               |         4 |    4 |
//! |   2   | `i32`  | `u32`               |         4 |    4 |
////! |  N/A  | N/A    | `[u8; 4]`           |         1 |    4 |
//! |  N/A  | N/A    | `[u8; 4]`           |         1 |    4 |
////! |   3   | `i64`  | `u64`               |         8 |    8 |
//! |   3   | `i64`  | `u64`               |         8 |    8 |
////! |   4   | `i252` | `[u64; 4]`          |         8 |    8 |
//! |   4   | `i252` | `[u64; 4]`          |         8 |    8 |
////!
//!
////! As inferred in the table above, the struct will have 8-byte alignment and a size of 30 bytes.
//! As inferred in the table above, the struct will have 8-byte alignment and a size of 30 bytes.
////! Since this way of generating structs is equivalent to the one used in C and C++, the same
//! Since this way of generating structs is equivalent to the one used in C and C++, the same
////! effects apply. For example, if we invert the order of the fields the ABI will change but we
//! effects apply. For example, if we invert the order of the fields the ABI will change but we
////! won't waste a single byte in padding; unless we're creating an array, in which case we'd waste
//! won't waste a single byte in padding; unless we're creating an array, in which case we'd waste
////! only a single byte per element.
//! only a single byte per element.
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
//        structure::StructConcreteType,
        structure::StructConcreteType,
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
//    info: WithSelf<StructConcreteType>,
    info: WithSelf<StructConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    let fields: Vec<_> = info
    let fields: Vec<_> = info
//        .members
        .members
//        .iter()
        .iter()
//        .map(|field| registry.build_type(context, module, registry, metadata, field))
        .map(|field| registry.build_type(context, module, registry, metadata, field))
//        .collect::<Result<_>>()?;
        .collect::<Result<_>>()?;
//    let struct_ty = llvm::r#type::r#struct(context, &fields, false);
    let struct_ty = llvm::r#type::r#struct(context, &fields, false);
//

//    Ok(struct_ty)
    Ok(struct_ty)
//}
}
