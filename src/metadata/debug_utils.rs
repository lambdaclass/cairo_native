//! # Debug utilities
//!
//! A collection of utilities to debug values in MLIR in execution.
//!
//! ## Example
//!
//! ```
//! # use cairo_lang_sierra::{
//! #     extensions::{
//! #         core::{CoreLibfunc, CoreType},
//! #         lib_func::SignatureAndTypeConcreteLibfunc,
//! #         GenericType,
//! #         GenericLibfunc,
//! #     },
//! #     program_registry::ProgramRegistry,
//! # };
//! # use cairo_native::{
//! #     error::{
//! #         Error, Result,
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
//! pub fn build_array_len<'ctx, 'this>(
//!     context: &'ctx Context,
//!     registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//!     entry: &'this Block<'ctx>,
//!     location: Location<'ctx>,
//!     helper: &LibfuncHelper<'ctx, 'this>,
//!     metadata: &mut MetadataStorage,
//!     info: &SignatureAndTypeConcreteLibfunc,
//! ) -> Result<()>
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

use crate::{block_ext::BlockExt, error::Result};
use melior::{
    dialect::{
        arith, func,
        llvm::{self, r#type::opaque_pointer},
        ods,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, Region, Value,
    },
    Context, ExecutionEngine,
};
use num_bigint::BigUint;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum DebugBinding {
    BreakpointMarker,
    DebugPrint,
    PrintI1,
    PrintI8,
    PrintI32,
    PrintI64,
    PrintI128,
    PrintPointer,
    PrintFelt252,
    DumpMemRegion,
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

    /// Prints the given &str.
    pub fn debug_print<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        message: &str,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::DebugPrint) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__debug_print_impl"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            opaque_pointer(context),
                            IntegerType::new(context, 64).into(),
                        ],
                        &[],
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
        }

        let ty = llvm::r#type::array(
            IntegerType::new(context, 8).into(),
            message.len().try_into().unwrap(),
        );

        let k1 = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
                location,
            ))
            .result(0)?
            .into();

        let ptr = block
            .append_operation(
                {
                    let mut op = ods::llvm::alloca(context, opaque_pointer(context), k1, location);
                    op.set_elem_type(TypeAttribute::new(ty));
                    op
                }
                .into(),
            )
            .result(0)?
            .into();

        let msg = block
            .append_operation(
                ods::llvm::mlir_constant(
                    context,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        message.len().try_into().unwrap(),
                    ),
                    StringAttribute::new(context, message).into(),
                    location,
                )
                .into(),
            )
            .result(0)?
            .into();
        block.append_operation(ods::llvm::store(context, msg, ptr, location).into());
        let len = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(
                    IntegerType::new(context, 64).into(),
                    message.len().try_into().unwrap(),
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__debug_print_impl"),
            &[ptr, len],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn debug_breakpoint_trap<'c, 'a>(
        &mut self,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        block.append_operation(OperationBuilder::new("llvm.intr.debugtrap", location).build()?);
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

    pub fn print_felt252<'c, 'a>(
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
        if self.active_map.insert(DebugBinding::PrintFelt252) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_felt252"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
                        ],
                        &[],
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
        }

        let k64 = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 64).into(),
                location,
            ))
            .result(0)?
            .into();

        let l0 = block
            .append_operation(arith::trunci(
                value,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let value = block
            .append_operation(arith::shrui(value, k64, location))
            .result(0)?
            .into();
        let l1 = block
            .append_operation(arith::trunci(
                value,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let value = block
            .append_operation(arith::shrui(value, k64, location))
            .result(0)?
            .into();
        let l2 = block
            .append_operation(arith::trunci(
                value,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();
        let value = block
            .append_operation(arith::shrui(value, k64, location))
            .result(0)?
            .into();
        let l3 = block
            .append_operation(arith::trunci(
                value,
                IntegerType::new(context, 64).into(),
                location,
            ))
            .result(0)?
            .into();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__print_felt252"),
            &[l0, l1, l2, l3],
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

    pub fn print_i32<'c, 'a>(
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
        if self.active_map.insert(DebugBinding::PrintI32) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_i32"),
                TypeAttribute::new(
                    FunctionType::new(context, &[IntegerType::new(context, 32).into()], &[]).into(),
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
            FlatSymbolRefAttribute::new(context, "__debug__print_i32"),
            &[value],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn print_i64<'c, 'a>(
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
        if self.active_map.insert(DebugBinding::PrintI64) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_i64"),
                TypeAttribute::new(
                    FunctionType::new(context, &[IntegerType::new(context, 64).into()], &[]).into(),
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
            FlatSymbolRefAttribute::new(context, "__debug__print_i64"),
            &[value],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn print_i128<'c, 'a>(
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
        if self.active_map.insert(DebugBinding::PrintI128) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__print_i128"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            IntegerType::new(context, 64).into(),
                            IntegerType::new(context, 64).into(),
                        ],
                        &[],
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
        }

        let i64_ty = IntegerType::new(context, 64).into();
        let k64 = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 128).into(), 64).into(),
                location,
            ))
            .result(0)?
            .into();

        let value_lo = block
            .append_operation(arith::trunci(value, i64_ty, location))
            .result(0)?
            .into();
        let value_hi = block
            .append_operation(arith::shrui(value, k64, location))
            .result(0)?
            .into();
        let value_hi = block
            .append_operation(arith::trunci(value_hi, i64_ty, location))
            .result(0)?
            .into();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__print_i128"),
            &[value_lo, value_hi],
            &[],
            location,
        ));

        Ok(())
    }

    /// Dump a memory region at runtime.
    ///
    /// Requires the pointer (at runtime) and its length in bytes (at compile-time).
    pub fn dump_mem<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        ptr: Value<'c, '_>,
        len: usize,
        location: Location<'c>,
    ) -> Result<()>
    where
        'c: 'a,
    {
        if self.active_map.insert(DebugBinding::DumpMemRegion) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__debug__dump_mem"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::opaque_pointer(context),
                            IntegerType::new(context, 64).into(),
                        ],
                        &[],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                location,
            ));
        }

        let len = block.const_int(context, location, len, 64)?;
        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__debug__dump_mem"),
            &[ptr, len],
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

        if self.active_map.contains(&DebugBinding::DebugPrint) {
            unsafe {
                engine.register_symbol(
                    "__debug__debug_print_impl",
                    debug_print_impl as *const fn(*const std::ffi::c_char) -> () as *mut (),
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

        if self.active_map.contains(&DebugBinding::PrintI32) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_i32",
                    print_i32_impl as *const fn(u8) -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::PrintI64) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_i64",
                    print_i64_impl as *const fn(u8) -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::PrintI128) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_i128",
                    print_i128_impl as *const fn(u64, u64) -> () as *mut (),
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

        if self.active_map.contains(&DebugBinding::PrintFelt252) {
            unsafe {
                engine.register_symbol(
                    "__debug__print_felt252",
                    print_felt252 as *const fn(u64, u64, u64, u64) -> () as *mut (),
                );
            }
        }

        if self.active_map.contains(&DebugBinding::DumpMemRegion) {
            unsafe {
                engine.register_symbol(
                    "__debug__dump_mem",
                    dump_mem_impl as *const fn(*const (), u64) as *mut (),
                );
            }
        }
    }
}

extern "C" fn breakpoint_marker_impl() {
    println!("[DEBUG] Breakpoint marker.");
}

extern "C" fn debug_print_impl(message: *const std::ffi::c_char, len: u64) {
    // llvm constant strings are not zero terminated
    let slice = unsafe { std::slice::from_raw_parts(message as *const u8, len as usize) };
    let message = std::str::from_utf8(slice);

    if let Ok(message) = message {
        println!("[DEBUG] Message: {}", message);
    } else {
        println!("[DEBUG] Message: {:?}", message);
    }
}

extern "C" fn print_i1_impl(value: bool) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_i8_impl(value: u8) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_i32_impl(value: u32) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_i64_impl(value: u64) {
    println!("[DEBUG] {value}");
}

extern "C" fn print_i128_impl(value_lo: u64, value_hi: u64) {
    let value = ((value_hi as u128) << 64) | value_lo as u128;
    println!("[DEBUG] {value}");
}

extern "C" fn print_pointer_impl(value: *const ()) {
    println!("[DEBUG] {value:018x?}");
}

unsafe extern "C" fn dump_mem_impl(ptr: *const (), len: u64) {
    println!("[DEBUG] Memory dump at {ptr:?}:");
    for chunk in (0..len).step_by(8) {
        print!("  {ptr:?}:");
        for offset in chunk..chunk + 8 {
            print!(" {:02x}", ptr.byte_add(offset as usize).cast::<u8>().read());
        }
        println!();
    }
}

extern "C" fn print_felt252(l0: u64, l1: u64, l2: u64, l3: u64) {
    println!(
        "[DEBUG] {}",
        BigUint::from_bytes_le(
            &l0.to_le_bytes()
                .into_iter()
                .chain(l1.to_le_bytes())
                .chain(l2.to_le_bytes())
                .chain(l3.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    );
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::context::NativeContext;
    use crate::utils;
    use std::path::Path;

    #[test]
    fn test_breakpoint_marker() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new context for the test.
        let context = Context::new();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Call the breakpoint_marker function with the context, module, block, and location.
        let res = debug_utils.breakpoint_marker(&context, &module, &block, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Create a set with the expected active debug binding.
        let expected_active_map: HashSet<_> =
            vec![DebugBinding::BreakpointMarker].into_iter().collect();

        // Assert that the active_map of debug_utils matches the expected set.
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_debug_print() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Define the message to be printed.
        let message = "random message!";

        // Call the debug_print function with the context, module, block, message, and location.
        let res = debug_utils.debug_print(&context, &module, &block, message, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Create a set with the expected active debug binding.
        let expected_active_map: HashSet<_> = vec![DebugBinding::DebugPrint].into_iter().collect();

        // Assert that the active_map of debug_utils matches the expected set.
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_debug_breakpoint_trap() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Call the debug_breakpoint_trap function with the block and location.
        let res = debug_utils.debug_breakpoint_trap(&block, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Assert that the active_map in debug_utils is still empty
        assert!(debug_utils.active_map.is_empty());
    }

    #[test]
    fn test_print_pointer() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an integer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_pointer function with the context, module, block, value, and location.
        let res = debug_utils.print_pointer(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintPointer after calling print_pointer.
        let expected_active_map: HashSet<_> =
            vec![DebugBinding::PrintPointer].into_iter().collect();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_i1() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a boolean constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 1).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i1 function with the context, module, block, value, and location.
        let res = debug_utils.print_i1(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintI1 after calling print_i1.
        let expected_active_map = [DebugBinding::PrintI1]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_felt252() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a felt252 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 256).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_felt252 function with the context, module, block, value, and location.
        let res = debug_utils.print_felt252(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintFelt252 after calling print_felt252.
        let expected_active_map = [DebugBinding::PrintFelt252]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_i8() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an i8 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 8).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i8 function with the context, module, block, value, and location.
        let res = debug_utils.print_i8(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintI8 after calling print_i8.
        let expected_active_map = [DebugBinding::PrintI8]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_i32() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an i32 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 32).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i32 function with the context, module, block, value, and location.
        let res = debug_utils.print_i32(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintI32 after calling print_i32.
        let expected_active_map = [DebugBinding::PrintI32]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_i64() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an i64 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i64 function with the context, module, block, value, and location.
        let res = debug_utils.print_i64(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintI64 after calling print_i64.
        let expected_active_map = [DebugBinding::PrintI64]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_print_i128() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an i128 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 128).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i128 function with the context, module, block, value, and location.
        let res = debug_utils.print_i128(&context, &module, &block, value, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::PrintI128 after calling print_i128.
        let expected_active_map = [DebugBinding::PrintI128]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_dump_mem() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();

        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a pointer value (e.g., an opaque pointer) to be dumped.
        let ptr = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 128).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Define the length of the memory region to be dumped.
        let len = 100;

        // Call the dump_mem function with the context, module, block, pointer, length, and location.
        let res = debug_utils.dump_mem(&context, &module, &block, ptr, len, location);

        // Assert that the result is Ok, indicating success.
        assert!(res.is_ok());

        // Ensure that the active_map contains the DebugBinding::DumpMemRegion after calling dump_mem.
        let expected_active_map = [DebugBinding::DumpMemRegion]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(debug_utils.active_map, expected_active_map);
    }

    #[test]
    fn test_register_impl_break_point_marker() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();

        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Call the breakpoint_marker function with the context, module, block, and location.
        let _ = debug_utils.breakpoint_marker(&context, &module, &block, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);

        // Instantiate a Cairo Native MLIR context. This data structure is responsible for MLIR
        // initialization and compilation of Sierra programs into an MLIR module.
        let native_context = NativeContext::new();

        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();

        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_debug_print_impl() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Call the debug_print function with the context, module, block, and message.
        let _ = debug_utils.debug_print(&context, &module, &block, "Test message", location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_print_i1() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a boolean constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 1).into(), 1).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i1 function with the context, module, block, and value.
        let _ = debug_utils.print_i1(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_print_i8() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, an 8-bit integer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 8).into(), 42).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i8 function with the context, module, block, and value.
        let _ = debug_utils.print_i8(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_print_i32() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a 32-bit integer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 32).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i32 function with the context, module, block, and value.
        let _ = debug_utils.print_i32(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    /// Test to verify the registration of `print_i64` implementation.
    #[test]
    fn test_register_impl_print_i64() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a 64-bit integer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i64 function with the context, module, block, and value.
        let _ = debug_utils.print_i64(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    /// Test to verify the registration of `print_i128` implementation.
    #[test]
    fn test_register_impl_print_i128() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a 128-bit integer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 128).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_i128 function with the context, module, block, and value.
        let _ = debug_utils.print_i128(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_print_pointer() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a pointer constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_pointer function with the context, module, block, and value.
        let _ = debug_utils.print_pointer(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_print_felt252() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a value to be printed (for example, a felt252 constant).
        let value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the print_felt252 function with the context, module, block, and value.
        let _ = debug_utils.print_felt252(&context, &module, &block, value, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__print_felt252").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__dump_mem").is_null());
    }

    #[test]
    fn test_register_impl_dump_mem() {
        // Create a default instance of DebugUtils.
        let mut debug_utils = DebugUtils::default();
        // Create a new native context for the test.
        let native_context = NativeContext::new();
        // Get the context from the native context.
        let context = native_context.context();
        // Create an unknown location within the context.
        let location = Location::unknown(&context);
        // Create a new module with the given location.
        let module = Module::new(location);
        // Create a new block with no instructions.
        let block = Block::new(&[]);

        // Create a pointer value to be dumped (for example, a pointer constant).
        let ptr_value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 123).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();

        // Call the dump_mem function with the context, module, block, pointer value, length, and location.
        let _ = debug_utils.dump_mem(&context, &module, &block, ptr_value, 10, location);

        // Path to the Cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the Cairo program to Sierra.
        let sierra_program = utils::cairo_to_sierra(program_path);
        // Instantiate a Cairo Native MLIR context.
        let native_context = NativeContext::new();
        // Compile the Sierra program into an MLIR module.
        let native_program = native_context.compile(&sierra_program, None).unwrap();
        // Create an execution engine for the MLIR module.
        let engine = ExecutionEngine::new(&native_program.module, 2, &[], false);

        // Register debug implementations in the engine.
        debug_utils.register_impls(&engine);

        // Verify that the debug functions are properly registered in the engine.
        assert!(!engine.lookup("__debug__dump_mem").is_null());
        assert!(engine.lookup("__debug__breakpoint_marker").is_null());
        assert!(engine.lookup("__debug__debug_print_impl").is_null());
        assert!(engine.lookup("__debug__print_i1").is_null());
        assert!(engine.lookup("__debug__print_i8").is_null());
        assert!(engine.lookup("__debug__print_i32").is_null());
        assert!(engine.lookup("__debug__print_i64").is_null());
        assert!(engine.lookup("__debug__print_i128").is_null());
        assert!(engine.lookup("__debug__print_pointer").is_null());
        assert!(engine.lookup("__debug__print_felt252").is_null());
    }
}
