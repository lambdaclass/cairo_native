//! # Debug utilities
//!
//! A collection of utilities to debug values in MLIR in execution.
//!
//! ## Example
//!
//! ```
//! # use cairo_lang_sierra::{
//! #     extensions::{
//! #         lib_func::SignatureAndTypeConcreteLibfunc,
//! #         GenericType,
//! #         GenericLibfunc,
//! #     },
//! #     program_registry::ProgramRegistry,
//! # };
//! # use cairo_native::{
//! #     error::{
//! #         libfuncs::{Error, Result},
//! #         CoreTypeBuilderError,
//! #     },
//! #     libfuncs::{LibfuncBuilder, LibfuncHelper},
//! #     metadata::{debug_utils::DebugUtils, MetadataStorage},
//! #     types::TypeBuilder,
//! #     utils::ProgramRegistryExt,
//! # };
//! # use melior::{
//! #     dialect::llvm,
//! #     ir::{
//! #         attribute::DenseI64ArrayAttribute,
//! #         r#type::IntegerType,
//! #         Block,
//! #         Location,
//! #     },
//! #     Context,
//! # };
//!
//! pub fn build_array_len<'ctx, 'this, TType, TLibfunc>(
//!     context: &'ctx Context,
//!     registry: &ProgramRegistry<TType, TLibfunc>,
//!     entry: &'this Block<'ctx>,
//!     location: Location<'ctx>,
//!     helper: &LibfuncHelper<'ctx, 'this>,
//!     metadata: &mut MetadataStorage,
//!     info: &SignatureAndTypeConcreteLibfunc,
//! ) -> Result<()>
//! where
//!     TType: GenericType,
//!     TLibfunc: GenericLibfunc,
//!     <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
//!     <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
//! {
//!     let array_val = entry.argument(0)?.into();
//!     let elem_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
//!
//!     #[cfg(feature = "with-debug-utils")]
//!     {
//!         let array_ptr = entry
//!             .append_operation(llvm::extract_value(
//!                 context,
//!                 array_val,
//!                 DenseI64ArrayAttribute::new(context, &[0]),
//!                 elem_ty,
//!                 location,
//!             ))
//!             .result(0)?
//!             .into();
//!
//!         metadata.get_mut::<DebugUtils>()
//!             .unwrap()
//!             .print_pointer(context, helper, entry, array_ptr, location)?;
//!     }
//!
//!     let array_len = entry
//!         .append_operation(llvm::extract_value(
//!             context,
//!             array_val,
//!             DenseI64ArrayAttribute::new(context, &[1]),
//!             IntegerType::new(context, 32).into(),
//!             location,
//!         ))
//!         .result(0)?
//!         .into();
//!
//!     entry.append_operation(helper.br(0, &[array_len], location));
//!     Ok(())
//! }
//! ```

#![cfg(feature = "with-debug-utils")]

use crate::error::libfuncs::Result;
use melior::{
    dialect::{func, llvm},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, Region, Value,
    },
    Context, ExecutionEngine,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum DebugBinding {
    BreakpointMarker,
    PrintI1,
    PrintI8,
    PrintPointer,
}

#[derive(Debug, Default)]
pub struct DebugUtils {
    active_map: HashSet<DebugBinding>,
}

impl DebugUtils {
    pub fn breakpoint_marker<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::BreakpointMarker) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__breakpoint_marker"),
                TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__breakpoint_marker"),
            &[],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn print_pointer<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::PrintPointer) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_pointer"),
                TypeAttribute::new(
                    FunctionType::new(context, &[llvm::r#type::opaque_pointer(context)], &[])
                        .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__print_pointer"),
            &[value],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn print_i1<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::PrintI1) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_i1"),
                TypeAttribute::new(
                    FunctionType::new(context, &[IntegerType::new(context, 1).into()], &[]).into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__print_i1"),
            &[value],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn print_i8<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::PrintI8) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_i8"),
                TypeAttribute::new(
                    FunctionType::new(context, &[IntegerType::new(context, 8).into()], &[]).into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__print_i8"),
            &[value],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn register_impls(&self, engine: &ExecutionEngine) {
        if self.active_map.contains(&DebugBinding::BreakpointMarker) {
            unsafe {
                engine.register_symbol(
                    "__debug__breakpoint_marker",
                    breakpoint_marker_impl as *const fn() -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::PrintI1) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_i1",
                    print_i1_impl as *const fn(bool) -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::PrintI8) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_i8",
                    print_i8_impl as *const fn(u8) -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::PrintPointer) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_pointer",
                    print_pointer_impl as *const fn(*const ()) -> () as *mut (),
                );
            }
        }
    }
}

extern "C" fn breakpoint_marker_impl() {
    println!("[DEBUG] Breakpoint marker.");
}

extern "C" fn print_i1_impl(value: bool) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_i8_impl(value: u8) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_pointer_impl(value: *const ()) {
    println!("[DEBUG] {value:018x?}");
}
