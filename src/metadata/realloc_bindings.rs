////! # Memory allocation external bindings
//! # Memory allocation external bindings
////!
//!
////! This metadata ensures that the bindings to the C function `realloc` exist in the current
//! This metadata ensures that the bindings to the C function `realloc` exist in the current
////! compilation context.
//! compilation context.
//

//use melior::{
use melior::{
//    dialect::{func, llvm},
    dialect::{func, llvm},
//    ir::{
    ir::{
//        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
//        r#type::{FunctionType, IntegerType},
        r#type::{FunctionType, IntegerType},
//        Identifier, Location, Module, Operation, Region, Value,
        Identifier, Location, Module, Operation, Region, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use std::marker::PhantomData;
use std::marker::PhantomData;
//

///// Memory allocation `realloc` metadata.
/// Memory allocation `realloc` metadata.
//#[derive(Debug)]
#[derive(Debug)]
//pub struct ReallocBindingsMeta {
pub struct ReallocBindingsMeta {
//    phantom: PhantomData<()>,
    phantom: PhantomData<()>,
//}
}
//

//impl ReallocBindingsMeta {
impl ReallocBindingsMeta {
//    /// Register the bindings to the `realloc` C function and return the metadata.
    /// Register the bindings to the `realloc` C function and return the metadata.
//    pub fn new(context: &Context, module: &Module) -> Self {
    pub fn new(context: &Context, module: &Module) -> Self {
//        module.body().append_operation(func::func(
        module.body().append_operation(func::func(
//            context,
            context,
//            StringAttribute::new(context, "realloc"),
            StringAttribute::new(context, "realloc"),
//            TypeAttribute::new(
            TypeAttribute::new(
//                FunctionType::new(
                FunctionType::new(
//                    context,
                    context,
//                    &[
                    &[
//                        llvm::r#type::pointer(context, 0),
                        llvm::r#type::pointer(context, 0),
//                        IntegerType::new(context, 64).into(),
                        IntegerType::new(context, 64).into(),
//                    ],
                    ],
//                    &[llvm::r#type::pointer(context, 0)],
                    &[llvm::r#type::pointer(context, 0)],
//                )
                )
//                .into(),
                .into(),
//            ),
            ),
//            Region::new(),
            Region::new(),
//            &[(
            &[(
//                Identifier::new(context, "sym_visibility"),
                Identifier::new(context, "sym_visibility"),
//                StringAttribute::new(context, "private").into(),
                StringAttribute::new(context, "private").into(),
//            )],
            )],
//            Location::unknown(context),
            Location::unknown(context),
//        ));
        ));
//        module.body().append_operation(func::func(
        module.body().append_operation(func::func(
//            context,
            context,
//            StringAttribute::new(context, "free"),
            StringAttribute::new(context, "free"),
//            TypeAttribute::new(
            TypeAttribute::new(
//                FunctionType::new(context, &[llvm::r#type::pointer(context, 0)], &[]).into(),
                FunctionType::new(context, &[llvm::r#type::pointer(context, 0)], &[]).into(),
//            ),
            ),
//            Region::new(),
            Region::new(),
//            &[(
            &[(
//                Identifier::new(context, "sym_visibility"),
                Identifier::new(context, "sym_visibility"),
//                StringAttribute::new(context, "private").into(),
                StringAttribute::new(context, "private").into(),
//            )],
            )],
//            Location::unknown(context),
            Location::unknown(context),
//        ));
        ));
//

//        Self {
        Self {
//            phantom: PhantomData,
            phantom: PhantomData,
//        }
        }
//    }
    }
//

//    /// Calls the `realloc` function, returns a op with 1 result: an opaque pointer.
    /// Calls the `realloc` function, returns a op with 1 result: an opaque pointer.
//    pub fn realloc<'c, 'a>(
    pub fn realloc<'c, 'a>(
//        context: &'c Context,
        context: &'c Context,
//        ptr: Value<'c, 'a>,
        ptr: Value<'c, 'a>,
//        len: Value<'c, 'a>,
        len: Value<'c, 'a>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Operation<'c> {
    ) -> Operation<'c> {
//        func::call(
        func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "realloc"),
            FlatSymbolRefAttribute::new(context, "realloc"),
//            &[ptr, len],
            &[ptr, len],
//            &[llvm::r#type::pointer(context, 0)],
            &[llvm::r#type::pointer(context, 0)],
//            location,
            location,
//        )
        )
//    }
    }
//

//    /// Calls the `free` function.
    /// Calls the `free` function.
//    pub fn free<'c>(
    pub fn free<'c>(
//        context: &'c Context,
        context: &'c Context,
//        ptr: Value<'c, '_>,
        ptr: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Operation<'c> {
    ) -> Operation<'c> {
//        func::call(
        func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "free"),
            FlatSymbolRefAttribute::new(context, "free"),
//            &[ptr],
            &[ptr],
//            &[],
            &[],
//            location,
            location,
//        )
        )
//    }
    }
//}
}
