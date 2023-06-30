//! # Struct type
//!
//! A struct is just a fixed collection of values that may have different types, which are known at
//! compile-time. Its fields are properly aligned and respect the declaration's field ordering.
//!
//! For example, the following struct would have a layout as described in the table below:
//!
//! ```cairo
//! struct MyStruct {
//!     U8: u8,
//!     U16: u16,
//!     U32: u32,
//!     U64: u64,
//!     Felt: felt252,
//! }
//! ```
//!
//! | Index | Type   | ABI (in Rust types) | Alignment | Size |
//! | ----- | ------ | ------------------- | --------- | ---- |
//! |   0   | `i8`   | `u8`                |         1 |    1 |
//! |  N/A  | N/A    | `[u8; 1]`           |         1 |    1 |
//! |   1   | `i16`  | `u16`               |         2 |    2 |
//! |  N/A  | N/A    | `[u8; 2]`           |         1 |    2 |
//! |   2   | `i32`  | `u32`               |         4 |    4 |
//! |  N/A  | N/A    | `[u8; 4]`           |         1 |    4 |
//! |   3   | `i64`  | `u64`               |         8 |    8 |
//! |   4   | `i252` | `[u64; 4]`          |         8 |    8 |
//!
//! As inferred in the table above, the struct will have 8-byte alignment and a size of 30 bytes.
//! Since this way of generating structs is equivalent to the one used in C and C++, the same
//! effects apply. For example, if we invert the order of the fields the ABI will change but we
//! won't waste a single byte in padding; unless we're creating an array, in which case we'd waste
//! only a single byte per element.

use super::TypeBuilder;
use crate::{
    error::types::{Error, Result},
    metadata::MetadataStorage,
};
use cairo_lang_sierra::{
    extensions::{structure::StructConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: &StructConcreteType,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<Error = Error>,
{
    let fields: Vec<_> = info
        .members
        .iter()
        .map(|field| {
            Result::Ok(
                registry
                    .get_type(field)?
                    .build(context, module, registry, metadata)?,
            )
        })
        .try_collect()?;
    let struct_ty = llvm::r#type::r#struct(context, &fields, false);

    Ok(struct_ty)
}
