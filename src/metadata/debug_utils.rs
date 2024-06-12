////! # Debug utilities
//! # Debug utilities
////!
//!
////! A collection of utilities to debug values in MLIR in execution.
//! A collection of utilities to debug values in MLIR in execution.
////!
//!
////! ## Example
//! ## Example
////!
//!
////! ```
//! ```
////! # use cairo_lang_sierra::{
//! # use cairo_lang_sierra::{
////! #     extensions::{
//! #     extensions::{
////! #         core::{CoreLibfunc, CoreType},
//! #         core::{CoreLibfunc, CoreType},
////! #         lib_func::SignatureAndTypeConcreteLibfunc,
//! #         lib_func::SignatureAndTypeConcreteLibfunc,
////! #         GenericType,
//! #         GenericType,
////! #         GenericLibfunc,
//! #         GenericLibfunc,
////! #     },
//! #     },
////! #     program_registry::ProgramRegistry,
//! #     program_registry::ProgramRegistry,
////! # };
//! # };
////! # use cairo_native::{
//! # use cairo_native::{
////! #     error::{
//! #     error::{
////! #         Error, Result,
//! #         Error, Result,
////! #     },
//! #     },
////! #     libfuncs::{LibfuncBuilder, LibfuncHelper},
//! #     libfuncs::{LibfuncBuilder, LibfuncHelper},
////! #     metadata::{debug_utils::DebugUtils, MetadataStorage},
//! #     metadata::{debug_utils::DebugUtils, MetadataStorage},
////! #     types::TypeBuilder,
//! #     types::TypeBuilder,
////! #     utils::ProgramRegistryExt,
//! #     utils::ProgramRegistryExt,
////! # };
//! # };
////! # use melior::{
//! # use melior::{
////! #     dialect::llvm,
//! #     dialect::llvm,
////! #     ir::{
//! #     ir::{
////! #         attribute::DenseI64ArrayAttribute,
//! #         attribute::DenseI64ArrayAttribute,
////! #         r#type::IntegerType,
//! #         r#type::IntegerType,
////! #         Block,
//! #         Block,
////! #         Location,
//! #         Location,
////! #     },
//! #     },
////! #     Context,
//! #     Context,
////! # };
//! # };
////!
//!
////! pub fn build_array_len<'ctx, 'this>(
//! pub fn build_array_len<'ctx, 'this>(
////!     context: &'ctx Context,
//!     context: &'ctx Context,
////!     registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//!     registry: &ProgramRegistry<CoreType, CoreLibfunc>,
////!     entry: &'this Block<'ctx>,
//!     entry: &'this Block<'ctx>,
////!     location: Location<'ctx>,
//!     location: Location<'ctx>,
////!     helper: &LibfuncHelper<'ctx, 'this>,
//!     helper: &LibfuncHelper<'ctx, 'this>,
////!     metadata: &mut MetadataStorage,
//!     metadata: &mut MetadataStorage,
////!     info: &SignatureAndTypeConcreteLibfunc,
//!     info: &SignatureAndTypeConcreteLibfunc,
////! ) -> Result<()>
//! ) -> Result<()>
////! {
//! {
////!     let array_val = entry.argument(0)?.into();
//!     let array_val = entry.argument(0)?.into();
////!     let elem_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
//!     let elem_ty = registry.build_type(context, helper, registry, metadata, &info.ty)?;
////!
//!
////!     #[cfg(feature = "with-debug-utils")]
//!     #[cfg(feature = "with-debug-utils")]
////!     {
//!     {
////!         let array_ptr = entry
//!         let array_ptr = entry
////!             .append_operation(llvm::extract_value(
//!             .append_operation(llvm::extract_value(
////!                 context,
//!                 context,
////!                 array_val,
//!                 array_val,
////!                 DenseI64ArrayAttribute::new(context, &[0]),
//!                 DenseI64ArrayAttribute::new(context, &[0]),
////!                 elem_ty,
//!                 elem_ty,
////!                 location,
//!                 location,
////!             ))
//!             ))
////!             .result(0)?
//!             .result(0)?
////!             .into();
//!             .into();
////!
//!
////!         metadata.get_mut::<DebugUtils>()
//!         metadata.get_mut::<DebugUtils>()
////!             .unwrap()
//!             .unwrap()
////!             .print_pointer(context, helper, entry, array_ptr, location)?;
//!             .print_pointer(context, helper, entry, array_ptr, location)?;
////!     }
//!     }
////!
//!
////!     let array_len = entry
//!     let array_len = entry
////!         .append_operation(llvm::extract_value(
//!         .append_operation(llvm::extract_value(
////!             context,
//!             context,
////!             array_val,
//!             array_val,
////!             DenseI64ArrayAttribute::new(context, &[1]),
//!             DenseI64ArrayAttribute::new(context, &[1]),
////!             IntegerType::new(context, 32).into(),
//!             IntegerType::new(context, 32).into(),
////!             location,
//!             location,
////!         ))
//!         ))
////!         .result(0)?
//!         .result(0)?
////!         .into();
//!         .into();
////!
//!
////!     entry.append_operation(helper.br(0, &[array_len], location));
//!     entry.append_operation(helper.br(0, &[array_len], location));
////!     Ok(())
//!     Ok(())
////! }
//! }
////! ```
//! ```
//

//#![cfg(feature = "with-debug-utils")]
#![cfg(feature = "with-debug-utils")]
//

//use crate::{block_ext::BlockExt, error::Result};
use crate::{block_ext::BlockExt, error::Result};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith, func,
        arith, func,
//        llvm::{self, r#type::pointer},
        llvm::{self, r#type::pointer},
//        ods,
        ods,
//    },
    },
//    ir::{
    ir::{
//        attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
//        operation::OperationBuilder,
        operation::OperationBuilder,
//        r#type::{FunctionType, IntegerType},
        r#type::{FunctionType, IntegerType},
//        Block, Identifier, Location, Module, Region, Value,
        Block, Identifier, Location, Module, Region, Value,
//    },
    },
//    Context, ExecutionEngine,
    Context, ExecutionEngine,
//};
};
//use num_bigint::BigUint;
use num_bigint::BigUint;
//use std::collections::HashSet;
use std::collections::HashSet;
//

//#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
//enum DebugBinding {
enum DebugBinding {
//    BreakpointMarker,
    BreakpointMarker,
//    DebugPrint,
    DebugPrint,
//    PrintI1,
    PrintI1,
//    PrintI8,
    PrintI8,
//    PrintI32,
    PrintI32,
//    PrintI64,
    PrintI64,
//    PrintI128,
    PrintI128,
//    PrintPointer,
    PrintPointer,
//    PrintFelt252,
    PrintFelt252,
//    DumpMemRegion,
    DumpMemRegion,
//}
}
//

//#[derive(Debug, Default)]
#[derive(Debug, Default)]
//pub struct DebugUtils {
pub struct DebugUtils {
//    active_map: HashSet<DebugBinding>,
    active_map: HashSet<DebugBinding>,
//}
}
//

//impl DebugUtils {
impl DebugUtils {
//    pub fn breakpoint_marker<'c, 'a>(
    pub fn breakpoint_marker<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::BreakpointMarker) {
        if self.active_map.insert(DebugBinding::BreakpointMarker) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__breakpoint_marker"),
                StringAttribute::new(context, "__debug__breakpoint_marker"),
//                TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
                TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__breakpoint_marker"),
            FlatSymbolRefAttribute::new(context, "__debug__breakpoint_marker"),
//            &[],
            &[],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    /// Prints the given &str.
    /// Prints the given &str.
//    pub fn debug_print<'c, 'a>(
    pub fn debug_print<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        message: &str,
        message: &str,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::DebugPrint) {
        if self.active_map.insert(DebugBinding::DebugPrint) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__debug_print_impl"),
                StringAttribute::new(context, "__debug__debug_print_impl"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[pointer(context, 0), IntegerType::new(context, 64).into()],
                        &[pointer(context, 0), IntegerType::new(context, 64).into()],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        let ty = llvm::r#type::array(
        let ty = llvm::r#type::array(
//            IntegerType::new(context, 8).into(),
            IntegerType::new(context, 8).into(),
//            message.len().try_into().unwrap(),
            message.len().try_into().unwrap(),
//        );
        );
//

//        let ptr = block.alloca1(context, location, ty, None)?;
        let ptr = block.alloca1(context, location, ty, None)?;
//

//        let msg = block
        let msg = block
//            .append_operation(
            .append_operation(
//                ods::llvm::mlir_constant(
                ods::llvm::mlir_constant(
//                    context,
                    context,
//                    llvm::r#type::array(
                    llvm::r#type::array(
//                        IntegerType::new(context, 8).into(),
                        IntegerType::new(context, 8).into(),
//                        message.len().try_into().unwrap(),
                        message.len().try_into().unwrap(),
//                    ),
                    ),
//                    StringAttribute::new(context, message).into(),
                    StringAttribute::new(context, message).into(),
//                    location,
                    location,
//                )
                )
//                .into(),
                .into(),
//            )
            )
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        block.append_operation(ods::llvm::store(context, msg, ptr, location).into());
        block.append_operation(ods::llvm::store(context, msg, ptr, location).into());
//        let len = block
        let len = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(
                IntegerAttribute::new(
//                    IntegerType::new(context, 64).into(),
                    IntegerType::new(context, 64).into(),
//                    message.len().try_into().unwrap(),
                    message.len().try_into().unwrap(),
//                )
                )
//                .into(),
                .into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__debug_print_impl"),
            FlatSymbolRefAttribute::new(context, "__debug__debug_print_impl"),
//            &[ptr, len],
            &[ptr, len],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn debug_breakpoint_trap<'c, 'a>(
    pub fn debug_breakpoint_trap<'c, 'a>(
//        &mut self,
        &mut self,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        block.append_operation(OperationBuilder::new("llvm.intr.debugtrap", location).build()?);
        block.append_operation(OperationBuilder::new("llvm.intr.debugtrap", location).build()?);
//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_pointer<'c, 'a>(
    pub fn print_pointer<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintPointer) {
        if self.active_map.insert(DebugBinding::PrintPointer) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_pointer"),
                StringAttribute::new(context, "__debug__print_pointer"),
//                TypeAttribute::new(FunctionType::new(context, &[pointer(context, 0)], &[]).into()),
                TypeAttribute::new(FunctionType::new(context, &[pointer(context, 0)], &[]).into()),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_pointer"),
            FlatSymbolRefAttribute::new(context, "__debug__print_pointer"),
//            &[value],
            &[value],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_i1<'c, 'a>(
    pub fn print_i1<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintI1) {
        if self.active_map.insert(DebugBinding::PrintI1) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_i1"),
                StringAttribute::new(context, "__debug__print_i1"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[IntegerType::new(context, 1).into()], &[]).into(),
                    FunctionType::new(context, &[IntegerType::new(context, 1).into()], &[]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_i1"),
            FlatSymbolRefAttribute::new(context, "__debug__print_i1"),
//            &[value],
            &[value],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_felt252<'c, 'a>(
    pub fn print_felt252<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintFelt252) {
        if self.active_map.insert(DebugBinding::PrintFelt252) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_felt252"),
                StringAttribute::new(context, "__debug__print_felt252"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        let k64 = block
        let k64 = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(IntegerType::new(context, 64).into(), 64).into(),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let l0 = block
        let l0 = block
//            .append_operation(arith::trunci(
            .append_operation(arith::trunci(
//                value,
                value,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = block
        let value = block
//            .append_operation(arith::shrui(value, k64, location))
            .append_operation(arith::shrui(value, k64, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let l1 = block
        let l1 = block
//            .append_operation(arith::trunci(
            .append_operation(arith::trunci(
//                value,
                value,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = block
        let value = block
//            .append_operation(arith::shrui(value, k64, location))
            .append_operation(arith::shrui(value, k64, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let l2 = block
        let l2 = block
//            .append_operation(arith::trunci(
            .append_operation(arith::trunci(
//                value,
                value,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value = block
        let value = block
//            .append_operation(arith::shrui(value, k64, location))
            .append_operation(arith::shrui(value, k64, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let l3 = block
        let l3 = block
//            .append_operation(arith::trunci(
            .append_operation(arith::trunci(
//                value,
                value,
//                IntegerType::new(context, 64).into(),
                IntegerType::new(context, 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_felt252"),
            FlatSymbolRefAttribute::new(context, "__debug__print_felt252"),
//            &[l0, l1, l2, l3],
            &[l0, l1, l2, l3],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_i8<'c, 'a>(
    pub fn print_i8<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintI8) {
        if self.active_map.insert(DebugBinding::PrintI8) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_i8"),
                StringAttribute::new(context, "__debug__print_i8"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[IntegerType::new(context, 8).into()], &[]).into(),
                    FunctionType::new(context, &[IntegerType::new(context, 8).into()], &[]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_i8"),
            FlatSymbolRefAttribute::new(context, "__debug__print_i8"),
//            &[value],
            &[value],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_i32<'c, 'a>(
    pub fn print_i32<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintI32) {
        if self.active_map.insert(DebugBinding::PrintI32) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_i32"),
                StringAttribute::new(context, "__debug__print_i32"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[IntegerType::new(context, 32).into()], &[]).into(),
                    FunctionType::new(context, &[IntegerType::new(context, 32).into()], &[]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_i32"),
            FlatSymbolRefAttribute::new(context, "__debug__print_i32"),
//            &[value],
            &[value],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_i64<'c, 'a>(
    pub fn print_i64<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintI64) {
        if self.active_map.insert(DebugBinding::PrintI64) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_i64"),
                StringAttribute::new(context, "__debug__print_i64"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(context, &[IntegerType::new(context, 64).into()], &[]).into(),
                    FunctionType::new(context, &[IntegerType::new(context, 64).into()], &[]).into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_i64"),
            FlatSymbolRefAttribute::new(context, "__debug__print_i64"),
//            &[value],
            &[value],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn print_i128<'c, 'a>(
    pub fn print_i128<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        value: Value<'c, '_>,
        value: Value<'c, '_>,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::PrintI128) {
        if self.active_map.insert(DebugBinding::PrintI128) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__print_i128"),
                StringAttribute::new(context, "__debug__print_i128"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                Location::unknown(context),
                Location::unknown(context),
//            ));
            ));
//        }
        }
//

//        let i64_ty = IntegerType::new(context, 64).into();
        let i64_ty = IntegerType::new(context, 64).into();
//        let k64 = block
        let k64 = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(IntegerType::new(context, 128).into(), 64).into(),
                IntegerAttribute::new(IntegerType::new(context, 128).into(), 64).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        let value_lo = block
        let value_lo = block
//            .append_operation(arith::trunci(value, i64_ty, location))
            .append_operation(arith::trunci(value, i64_ty, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value_hi = block
        let value_hi = block
//            .append_operation(arith::shrui(value, k64, location))
            .append_operation(arith::shrui(value, k64, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//        let value_hi = block
        let value_hi = block
//            .append_operation(arith::trunci(value_hi, i64_ty, location))
            .append_operation(arith::trunci(value_hi, i64_ty, location))
//            .result(0)?
            .result(0)?
//            .into();
            .into();
//

//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__print_i128"),
            FlatSymbolRefAttribute::new(context, "__debug__print_i128"),
//            &[value_lo, value_hi],
            &[value_lo, value_hi],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    /// Dump a memory region at runtime.
    /// Dump a memory region at runtime.
//    ///
    ///
//    /// Requires the pointer (at runtime) and its length in bytes (at compile-time).
    /// Requires the pointer (at runtime) and its length in bytes (at compile-time).
//    pub fn dump_mem<'c, 'a>(
    pub fn dump_mem<'c, 'a>(
//        &mut self,
        &mut self,
//        context: &'c Context,
        context: &'c Context,
//        module: &Module,
        module: &Module,
//        block: &'a Block<'c>,
        block: &'a Block<'c>,
//        ptr: Value<'c, '_>,
        ptr: Value<'c, '_>,
//        len: usize,
        len: usize,
//        location: Location<'c>,
        location: Location<'c>,
//    ) -> Result<()>
    ) -> Result<()>
//    where
    where
//        'c: 'a,
        'c: 'a,
//    {
    {
//        if self.active_map.insert(DebugBinding::DumpMemRegion) {
        if self.active_map.insert(DebugBinding::DumpMemRegion) {
//            module.body().append_operation(func::func(
            module.body().append_operation(func::func(
//                context,
                context,
//                StringAttribute::new(context, "__debug__dump_mem"),
                StringAttribute::new(context, "__debug__dump_mem"),
//                TypeAttribute::new(
                TypeAttribute::new(
//                    FunctionType::new(
                    FunctionType::new(
//                        context,
                        context,
//                        &[
                        &[
//                            llvm::r#type::pointer(context, 0),
                            llvm::r#type::pointer(context, 0),
//                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
//                        ],
                        ],
//                        &[],
                        &[],
//                    )
                    )
//                    .into(),
                    .into(),
//                ),
                ),
//                Region::new(),
                Region::new(),
//                &[(
                &[(
//                    Identifier::new(context, "sym_visibility"),
                    Identifier::new(context, "sym_visibility"),
//                    StringAttribute::new(context, "private").into(),
                    StringAttribute::new(context, "private").into(),
//                )],
                )],
//                location,
                location,
//            ));
            ));
//        }
        }
//

//        let len = block.const_int(context, location, len, 64)?;
        let len = block.const_int(context, location, len, 64)?;
//        block.append_operation(func::call(
        block.append_operation(func::call(
//            context,
            context,
//            FlatSymbolRefAttribute::new(context, "__debug__dump_mem"),
            FlatSymbolRefAttribute::new(context, "__debug__dump_mem"),
//            &[ptr, len],
            &[ptr, len],
//            &[],
            &[],
//            location,
            location,
//        ));
        ));
//

//        Ok(())
        Ok(())
//    }
    }
//

//    pub fn register_impls(&self, engine: &ExecutionEngine) {
    pub fn register_impls(&self, engine: &ExecutionEngine) {
//        if self.active_map.contains(&DebugBinding::BreakpointMarker) {
        if self.active_map.contains(&DebugBinding::BreakpointMarker) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__breakpoint_marker",
                    "__debug__breakpoint_marker",
//                    breakpoint_marker_impl as *const fn() -> () as *mut (),
                    breakpoint_marker_impl as *const fn() -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::DebugPrint) {
        if self.active_map.contains(&DebugBinding::DebugPrint) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__debug_print_impl",
                    "__debug__debug_print_impl",
//                    debug_print_impl as *const fn(*const std::ffi::c_char) -> () as *mut (),
                    debug_print_impl as *const fn(*const std::ffi::c_char) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintI1) {
        if self.active_map.contains(&DebugBinding::PrintI1) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_i1",
                    "__debug__print_i1",
//                    print_i1_impl as *const fn(bool) -> () as *mut (),
                    print_i1_impl as *const fn(bool) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintI8) {
        if self.active_map.contains(&DebugBinding::PrintI8) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_i8",
                    "__debug__print_i8",
//                    print_i8_impl as *const fn(u8) -> () as *mut (),
                    print_i8_impl as *const fn(u8) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintI32) {
        if self.active_map.contains(&DebugBinding::PrintI32) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_i32",
                    "__debug__print_i32",
//                    print_i32_impl as *const fn(u8) -> () as *mut (),
                    print_i32_impl as *const fn(u8) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintI64) {
        if self.active_map.contains(&DebugBinding::PrintI64) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_i64",
                    "__debug__print_i64",
//                    print_i64_impl as *const fn(u8) -> () as *mut (),
                    print_i64_impl as *const fn(u8) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintI128) {
        if self.active_map.contains(&DebugBinding::PrintI128) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_i128",
                    "__debug__print_i128",
//                    print_i128_impl as *const fn(u64, u64) -> () as *mut (),
                    print_i128_impl as *const fn(u64, u64) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintPointer) {
        if self.active_map.contains(&DebugBinding::PrintPointer) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_pointer",
                    "__debug__print_pointer",
//                    print_pointer_impl as *const fn(*const ()) -> () as *mut (),
                    print_pointer_impl as *const fn(*const ()) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::PrintFelt252) {
        if self.active_map.contains(&DebugBinding::PrintFelt252) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__print_felt252",
                    "__debug__print_felt252",
//                    print_felt252 as *const fn(u64, u64, u64, u64) -> () as *mut (),
                    print_felt252 as *const fn(u64, u64, u64, u64) -> () as *mut (),
//                );
                );
//            }
            }
//        }
        }
//

//        if self.active_map.contains(&DebugBinding::DumpMemRegion) {
        if self.active_map.contains(&DebugBinding::DumpMemRegion) {
//            unsafe {
            unsafe {
//                engine.register_symbol(
                engine.register_symbol(
//                    "__debug__dump_mem",
                    "__debug__dump_mem",
//                    dump_mem_impl as *const fn(*const (), u64) as *mut (),
                    dump_mem_impl as *const fn(*const (), u64) as *mut (),
//                );
                );
//            }
            }
//        }
        }
//    }
    }
//}
}
//

//extern "C" fn breakpoint_marker_impl() {
extern "C" fn breakpoint_marker_impl() {
//    println!("[DEBUG] Breakpoint marker.");
    println!("[DEBUG] Breakpoint marker.");
//}
}
//

//extern "C" fn debug_print_impl(message: *const std::ffi::c_char, len: u64) {
extern "C" fn debug_print_impl(message: *const std::ffi::c_char, len: u64) {
//    // llvm constant strings are not zero terminated
    // llvm constant strings are not zero terminated
//    let slice = unsafe { std::slice::from_raw_parts(message as *const u8, len as usize) };
    let slice = unsafe { std::slice::from_raw_parts(message as *const u8, len as usize) };
//    let message = std::str::from_utf8(slice);
    let message = std::str::from_utf8(slice);
//

//    if let Ok(message) = message {
    if let Ok(message) = message {
//        println!("[DEBUG] Message: {}", message);
        println!("[DEBUG] Message: {}", message);
//    } else {
    } else {
//        println!("[DEBUG] Message: {:?}", message);
        println!("[DEBUG] Message: {:?}", message);
//    }
    }
//}
}
//

//extern "C" fn print_i1_impl(value: bool) {
extern "C" fn print_i1_impl(value: bool) {
//    println!("[DEBUG] {value}");
    println!("[DEBUG] {value}");
//}
}
//

//extern "C" fn print_i8_impl(value: u8) {
extern "C" fn print_i8_impl(value: u8) {
//    println!("[DEBUG] {value}");
    println!("[DEBUG] {value}");
//}
}
//

//extern "C" fn print_i32_impl(value: u32) {
extern "C" fn print_i32_impl(value: u32) {
//    println!("[DEBUG] {value}");
    println!("[DEBUG] {value}");
//}
}
//

//extern "C" fn print_i64_impl(value: u64) {
extern "C" fn print_i64_impl(value: u64) {
//    println!("[DEBUG] {value}");
    println!("[DEBUG] {value}");
//}
}
//

//extern "C" fn print_i128_impl(value_lo: u64, value_hi: u64) {
extern "C" fn print_i128_impl(value_lo: u64, value_hi: u64) {
//    let value = ((value_hi as u128) << 64) | value_lo as u128;
    let value = ((value_hi as u128) << 64) | value_lo as u128;
//    println!("[DEBUG] {value}");
    println!("[DEBUG] {value}");
//}
}
//

//extern "C" fn print_pointer_impl(value: *const ()) {
extern "C" fn print_pointer_impl(value: *const ()) {
//    println!("[DEBUG] {value:018x?}");
    println!("[DEBUG] {value:018x?}");
//}
}
//

//unsafe extern "C" fn dump_mem_impl(ptr: *const (), len: u64) {
unsafe extern "C" fn dump_mem_impl(ptr: *const (), len: u64) {
//    println!("[DEBUG] Memory dump at {ptr:?}:");
    println!("[DEBUG] Memory dump at {ptr:?}:");
//    for chunk in (0..len).step_by(8) {
    for chunk in (0..len).step_by(8) {
//        print!("  {ptr:?}:");
        print!("  {ptr:?}:");
//        for offset in chunk..chunk + 8 {
        for offset in chunk..chunk + 8 {
//            print!(" {:02x}", ptr.byte_add(offset as usize).cast::<u8>().read());
            print!(" {:02x}", ptr.byte_add(offset as usize).cast::<u8>().read());
//        }
        }
//        println!();
        println!();
//    }
    }
//}
}
//

//extern "C" fn print_felt252(l0: u64, l1: u64, l2: u64, l3: u64) {
extern "C" fn print_felt252(l0: u64, l1: u64, l2: u64, l3: u64) {
//    println!(
    println!(
//        "[DEBUG] {}",
        "[DEBUG] {}",
//        BigUint::from_bytes_le(
        BigUint::from_bytes_le(
//            &l0.to_le_bytes()
            &l0.to_le_bytes()
//                .into_iter()
                .into_iter()
//                .chain(l1.to_le_bytes())
                .chain(l1.to_le_bytes())
//                .chain(l2.to_le_bytes())
                .chain(l2.to_le_bytes())
//                .chain(l3.to_le_bytes())
                .chain(l3.to_le_bytes())
//                .collect::<Vec<_>>(),
                .collect::<Vec<_>>(),
//        ),
        ),
//    );
    );
//}
}
