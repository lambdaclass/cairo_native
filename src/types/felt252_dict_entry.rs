//! # `Felt` dictionary entry type
//!
//! The entry type returning when getting a value from a dictionary.
//!
//! It is represented as the following struct:
//!
//! | Index | Type           | Description                      |
//! | ----- | -------------- | -----------------------------    |
//! |   0   | `i252>`        | The entry key.                   |
//! |   1   | `!llvm.ptr`    | Pointer to the entry value.      |
//! |   2   | `!llvm.ptr`    | Pointer to the dictionary (rust) |
//!

use super::{TypeBuilder, WithSelf};
use crate::{
    error::types::{Error, Result},
    metadata::MetadataStorage,
};
use cairo_lang_sierra::{
    extensions::{types::InfoAndTypeConcreteType, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    _metadata: &mut MetadataStorage,
    _info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    Ok(llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(), // entry key
            llvm::r#type::opaque_pointer(context), // value ptr
            llvm::r#type::opaque_pointer(context), // dict ptr
        ],
        false,
    ))
}
