//! # Memory allocation external bindings
//!
//! This metadata ensures that the bindings to the C function `realloc` exist in the current
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

/// Memory allocation `realloc` metadata.
#[derive(Debug)]
pub struct ReallocBindingsMeta {
    phantom: PhantomData<()>,
}

impl ReallocBindingsMeta {
    /// Register the bindings to the `realloc` C function and return the metadata.
    pub fn new(context: &Context, module: &Module) -> Self {
        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "realloc"),
            TypeAttribute::new(
                FunctionType::new(
                    context,
                    &[
                        llvm::r#type::pointer(context, 0),
                        IntegerType::new(context, 64).into(),
                    ],
                    &[llvm::r#type::pointer(context, 0)],
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
        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "free"),
            TypeAttribute::new(
                FunctionType::new(context, &[llvm::r#type::pointer(context, 0)], &[]).into(),
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

    /// Calls the `realloc` function, returns a op with 1 result: an opaque pointer.
    pub fn realloc<'c, 'a>(
        context: &'c Context,
        ptr: Value<'c, 'a>,
        len: Value<'c, 'a>,
        location: Location<'c>,
    ) -> Operation<'c> {
        func::call(
            context,
            FlatSymbolRefAttribute::new(context, "realloc"),
            &[ptr, len],
            &[llvm::r#type::pointer(context, 0)],
            location,
        )
    }

    /// Calls the `free` function.
    pub fn free<'c>(
        context: &'c Context,
        ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        func::call(
            context,
            FlatSymbolRefAttribute::new(context, "free"),
            &[ptr],
            &[],
            location,
        )
    }
}
// PLT: could abstract away function creation as mentioned in `DebugImpl`.
// PLT: ACK
