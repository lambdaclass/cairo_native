//! # Memory allocation external bindings
//!
//! This metadata ensures that the bindings to the C function `realloc` exist in the current
//! compilation context.

use melior::{
    dialect::llvm,
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Identifier, Location, Module, Operation, Region, Value,
    },
    Context,
};

/// Memory allocation `realloc` metadata.
#[derive(Debug)]
pub struct ReallocBindingsMeta;

impl ReallocBindingsMeta {
    /// Register the bindings to the `realloc` C function and return the metadata.
    pub fn new(context: &Context, module: &Module) -> Self {
        module.body().append_operation(llvm::func(
            context,
            StringAttribute::new(context, "realloc"),
            TypeAttribute::new(llvm::r#type::function(
                llvm::r#type::pointer(context, 0),
                &[
                    llvm::r#type::pointer(context, 0),
                    IntegerType::new(context, 64).into(),
                ],
                false,
            )),
            Region::new(),
            &[(
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            )],
            Location::unknown(context),
        ));
        module.body().append_operation(llvm::func(
            context,
            StringAttribute::new(context, "free"),
            TypeAttribute::new(llvm::r#type::function(
                llvm::r#type::void(context),
                &[llvm::r#type::pointer(context, 0)],
                false,
            )),
            Region::new(),
            &[(
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            )],
            Location::unknown(context),
        ));

        Self
    }

    /// Calls the `realloc` function, returns a op with 1 result: an opaque pointer.
    pub fn realloc<'c, 'a>(
        context: &'c Context,
        ptr: Value<'c, 'a>,
        len: Value<'c, 'a>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.call", location)
            .add_attributes(&[(
                Identifier::new(context, "callee"),
                FlatSymbolRefAttribute::new(context, "realloc").into(),
            )])
            .add_operands(&[ptr, len])
            .add_results(&[llvm::r#type::pointer(context, 0)])
            .build()
            .unwrap()
    }

    /// Calls the `free` function.
    pub fn free<'c>(
        context: &'c Context,
        ptr: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.call", location)
            .add_attributes(&[(
                Identifier::new(context, "callee"),
                FlatSymbolRefAttribute::new(context, "free").into(),
            )])
            .add_operands(&[ptr])
            .build()
            .unwrap()
    }
}
