//! # Memory allocation external bindings
//!
//! This metadata ensures that the bindings to the C function `memmove` exist in the current
//! compilation context.

use melior::{
    dialect::{func, llvm},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Identifier, Location, Module, Operation, Region, Value,
    },
    Context,
};
use std::marker::PhantomData;

/// Memory allocation `memmove` metadata.
#[derive(Debug)]
pub struct MemmoveBindingsMeta {
    phantom: PhantomData<()>,
}

impl MemmoveBindingsMeta {
    /// Register the bindings to the `memmove` C function and return the metadata.
    pub fn new(context: &Context, module: &Module) -> Self {
        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "memmove"),
            TypeAttribute::new(
                FunctionType::new(
                    context,
                    &[
                        llvm::r#type::opaque_pointer(context), // dst
                        llvm::r#type::opaque_pointer(context), // src
                        IntegerType::new(context, 64).into(),  // len
                    ],
                    &[llvm::r#type::opaque_pointer(context)],
                )
                .into(),
            ),
            Region::new(),
            &[(
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            )],
            Location::unknown(context),
        ));

        Self {
            phantom: PhantomData,
        }
    }

    /// Calls the `memmove` function, returns a op with 1 result: an opaque pointer.
    pub fn memmove<'c, 'a>(
        context: &'c Context,
        dst_ptr: Value<'c, 'a>,
        src_ptr: Value<'c, 'a>,
        len: Value<'c, 'a>,
        location: Location<'c>,
    ) -> Operation<'c> {
        func::call(
            context,
            FlatSymbolRefAttribute::new(context, "memmove"),
            &[dst_ptr, src_ptr, len],
            &[llvm::r#type::opaque_pointer(context)],
            location,
        )
    }
}
