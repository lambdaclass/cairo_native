//! # Debug utilities
//!
//! A collection of utilities to debug values in MLIR in execution.
//!
//! ## Example
//!
//! ```rust,ignore
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
//!     let array_val = entry.arg(0)?;
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
//!             .ok_or(Error::MissingMetadata)?
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

use crate::{
    error::{Error, Result},
    utils::{get_integer_layout, BlockExt},
};
use melior::{
    dialect::{
        arith,
        llvm::{self},
        ods,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, BlockLike, Location, Module, Region, Value,
    },
    Context,
};
use num_bigint::BigUint;
use std::{collections::HashSet, ffi::c_void};

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum DebugBinding {
    BreakpointMarker,
    PrintStr,
    PrintI1,
    PrintI8,
    PrintI32,
    PrintI64,
    PrintI128,
    PrintPointer,
    PrintFelt252,
    DumpMemRegion,
}

impl DebugBinding {
    const fn symbol(self) -> &'static str {
        match self {
            DebugBinding::BreakpointMarker => "cairo_native__debug__breakpoint_marker_impl",
            DebugBinding::PrintStr => "cairo_native__debug__print_str_impl",
            DebugBinding::PrintI1 => "cairo_native__debug__print_i1_impl",
            DebugBinding::PrintI8 => "cairo_native__debug__print_i8_impl",
            DebugBinding::PrintI32 => "cairo_native__debug__print_i32_impl",
            DebugBinding::PrintI64 => "cairo_native__debug__print_i64_impl",
            DebugBinding::PrintI128 => "cairo_native__debug__print_i128_impl",
            DebugBinding::PrintPointer => "cairo_native__debug__print_pointer_impl",
            DebugBinding::PrintFelt252 => "cairo_native__debug__print_felt252_impl",
            DebugBinding::DumpMemRegion => "cairo_native__debug__dump_mem_region_impl",
        }
    }
    const fn function_ptr(self) -> *const () {
        match self {
            DebugBinding::BreakpointMarker => breakpoint_marker_impl as *const (),
            DebugBinding::PrintStr => print_str_impl as *const (),
            DebugBinding::PrintI1 => print_i1_impl as *const (),
            DebugBinding::PrintI8 => print_i8_impl as *const (),
            DebugBinding::PrintI32 => print_i32_impl as *const (),
            DebugBinding::PrintI64 => print_i64_impl as *const (),
            DebugBinding::PrintI128 => print_i128_impl as *const (),
            DebugBinding::PrintPointer => print_pointer_impl as *const (),
            DebugBinding::PrintFelt252 => print_felt252_impl as *const (),
            DebugBinding::DumpMemRegion => dump_mem_region_impl as *const (),
        }
    }
}

#[derive(Debug, Default)]
pub struct DebugUtils {
    active_map: HashSet<DebugBinding>,
}

impl DebugUtils {
    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored function.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    fn build_function<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        binding: DebugBinding,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(binding) {
            module.body().append_operation(
                ods::llvm::mlir_global(
                    context,
                    Region::new(),
                    TypeAttribute::new(llvm::r#type::pointer(context, 0)),
                    StringAttribute::new(context, binding.symbol()),
                    Attribute::parse(context, "#llvm.linkage<weak>")
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            );
        }

        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, binding.symbol()),
                location,
            )
            .into(),
        )?;

        block.load(
            context,
            location,
            global_address,
            llvm::r#type::pointer(context, 0),
        )
    }

    pub fn breakpoint_marker(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        location: Location,
    ) -> Result<()> {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            DebugBinding::BreakpointMarker,
        )?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .build()?,
        );

        Ok(())
    }

    /// Prints the given &str.
    pub fn debug_print(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        message: &str,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintStr)?;

        let ty = llvm::r#type::array(
            IntegerType::new(context, 8).into(),
            message
                .len()
                .try_into()
                .map_err(|_| Error::IntegerConversion)?,
        );

        let ptr = block.alloca1(context, location, ty, get_integer_layout(8).align())?;

        let msg = block
            .append_operation(
                ods::llvm::mlir_constant(
                    context,
                    llvm::r#type::array(
                        IntegerType::new(context, 8).into(),
                        message
                            .len()
                            .try_into()
                            .map_err(|_| Error::IntegerConversion)?,
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
                    message
                        .len()
                        .try_into()
                        .map_err(|_| Error::IntegerConversion)?,
                )
                .into(),
                location,
            ))
            .result(0)?
            .into();

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[ptr, len])
                .build()?,
        );

        Ok(())
    }

    pub fn debug_breakpoint_trap(&self, block: &Block, location: Location) -> Result<()> {
        block.append_operation(OperationBuilder::new("llvm.intr.debugtrap", location).build()?);
        Ok(())
    }

    pub fn print_pointer(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintPointer)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value])
                .build()?,
        );

        Ok(())
    }

    pub fn print_i1(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintI1)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value])
                .build()?,
        );

        Ok(())
    }

    pub fn print_felt252(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintFelt252)?;

        let k64 = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(IntegerType::new(context, 252).into(), 64).into(),
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

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[l0, l1, l2, l3])
                .build()?,
        );

        Ok(())
    }

    pub fn print_i8(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintI8)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value])
                .build()?,
        );

        Ok(())
    }

    pub fn print_i32(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintI32)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value])
                .build()?,
        );

        Ok(())
    }

    pub fn print_i64(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintI64)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value])
                .build()?,
        );

        Ok(())
    }

    pub fn print_i128(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        value: Value,
        location: Location,
    ) -> Result<()> {
        let function =
            self.build_function(context, module, block, location, DebugBinding::PrintI128)?;

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

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[value_lo, value_hi])
                .build()?,
        );

        Ok(())
    }

    /// Dump a memory region at runtime.
    ///
    /// Requires the pointer (at runtime) and its length in bytes (at compile-time).
    pub fn dump_mem(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        ptr: Value,
        len: usize,
        location: Location,
    ) -> Result<()> {
        let function = self.build_function(
            context,
            module,
            block,
            location,
            DebugBinding::DumpMemRegion,
        )?;

        let len = block.const_int(context, location, len, 64)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[ptr, len])
                .build()?,
        );

        Ok(())
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    for binding in [
        DebugBinding::BreakpointMarker,
        DebugBinding::PrintStr,
        DebugBinding::PrintI1,
        DebugBinding::PrintI8,
        DebugBinding::PrintI32,
        DebugBinding::PrintI64,
        DebugBinding::PrintI128,
        DebugBinding::PrintPointer,
        DebugBinding::PrintFelt252,
        DebugBinding::DumpMemRegion,
    ] {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}

extern "C" fn breakpoint_marker_impl() {
    println!("[DEBUG] Breakpoint marker.");
}

extern "C" fn print_str_impl(message: *const std::ffi::c_char, len: u64) {
    // llvm constant strings are not zero terminated
    let slice = unsafe { std::slice::from_raw_parts(message as *const u8, len as usize) };
    let message = std::str::from_utf8(slice);

    if let Ok(message) = message {
        println!("[DEBUG] {}", message);
    } else {
        println!("[DEBUG] {:?}", message);
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

unsafe extern "C" fn dump_mem_region_impl(ptr: *const (), len: u64) {
    println!("[DEBUG] Memory dump at {ptr:?}:");
    for chunk in (0..len).step_by(8) {
        print!("  {:?}:", ptr.byte_add(chunk as usize));
        for offset in chunk..chunk + 8 {
            print!(" {:02x}", ptr.byte_add(offset as usize).cast::<u8>().read());
        }
        println!();
    }
}

extern "C" fn print_felt252_impl(l0: u64, l1: u64, l2: u64, l3: u64) {
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
