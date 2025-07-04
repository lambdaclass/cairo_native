#![cfg(feature = "with-libfunc-counter")]
//! The libfunc counter feature is used to generate information counting how many time a libfunc has been called.
//!
//! When this feature is used, the compiler will call three important methods:
//!
//! 1. `count_libfunc`: called before every libfunc execution.
//!
//! 2. `build_array_counter`: called only once to build the array of counters (one for each libfuncs). The order of
//!    is based on the libfuncs' declaration order.
//!
//! 3. `store_array_counter`: called before finishing each entrypoint execution. It transforms the MLIR array into a
//!    Rust vector which is then stored in `LIBFUNC_COUNTER`, a static variable that registers the array of counters by
//!    execution, along with its `counter_id` (which is relative to the execution).
//!
//! In the context of Starknet contracts, we need to add support for building
//! the arrays of counters for multiple executions. To do so, we need one important element, which must be set before every contract
//! execution:
//!
//! A counter to track the ID of the current array of counter, which gets updated every time we switch to another
//! contract. Since a contract can call other contracts, we need a way of restoring the counter after every execution.
//!
//! See `cairo-native-run` for an example on how to do it.
use std::{collections::HashSet, os::raw::c_void, ptr};

use melior::{
    dialect::{llvm, memref, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{IntegerType, MemRefType},
        Attribute, Block, BlockLike, Location, Module, Region, Value,
    },
    Context,
};

use crate::{
    error::{Error, Result},
    metadata::realloc_bindings::ReallocBindingsMeta,
    utils::{get_integer_layout, layout_repeat, BlockExt, GepIndex},
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LibfuncCounterBinding {
    StoreArrayCounter,
    CounterId,
    ArrayCounter,
}

impl LibfuncCounterBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            LibfuncCounterBinding::StoreArrayCounter => "cairo_native__store_array_counter",
            LibfuncCounterBinding::CounterId => "cairo_native__counter_id",
            LibfuncCounterBinding::ArrayCounter => "cairo_native__array_counter",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            LibfuncCounterBinding::StoreArrayCounter => {
                libfunc_counter_runtime::store_array_counter as *const ()
            }
            LibfuncCounterBinding::CounterId => ptr::null(),
            LibfuncCounterBinding::ArrayCounter => ptr::null(),
        }
    }
}

#[derive(Clone, Default)]
pub struct LibfuncCounterMeta {
    active_map: HashSet<LibfuncCounterBinding>,
}

impl LibfuncCounterMeta {
    pub fn new() -> Self {
        Self {
            active_map: HashSet::new(),
        }
    }

    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored value.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    pub fn build_function<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        binding: LibfuncCounterBinding,
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

    fn build_counter_id<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(LibfuncCounterBinding::CounterId) {
            module.body().append_operation(memref::global(
                context,
                LibfuncCounterBinding::CounterId.symbol(),
                None,
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                None,
                false,
                None,
                location,
            ));
        }

        let libfunc_counter_id_ptr = block
            .append_op_result(memref::get_global(
                context,
                LibfuncCounterBinding::CounterId.symbol(),
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                location,
            ))
            .unwrap();

        block.append_op_result(memref::load(libfunc_counter_id_ptr, &[], location))
    }

    /// Indexes the array of counters and increments the counter relative
    /// to the given libfunc index
    pub fn store_array_counter(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block<'_>,
        location: Location,
        libfunc_amount: u32,
    ) -> Result<()> {
        let counter_id = self.build_counter_id(context, module, block, location)?;
        let function_ptr = self.build_function(
            context,
            module,
            block,
            location,
            LibfuncCounterBinding::StoreArrayCounter,
        )?;
        let lifuncs_amount = block.const_int(context, location, libfunc_amount, 32)?;
        // by this time, the array counter should be initialized
        let array_counter_ptr_ptr = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                location,
            )
            .into(),
        )?;
        let array_counter_ptr = block.load(context, location, array_counter_ptr_ptr, llvm::r#type::pointer(context, 0))?;
        

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function_ptr])
                .add_operands(&[counter_id, array_counter_ptr, lifuncs_amount])
                .build()?,
        );

        Ok(())
    }

    /// Build the array of counters
    fn get_array_counter<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        libfunc_amount: u32,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(LibfuncCounterBinding::ArrayCounter) {
            module.body().append_operation(
                ods::llvm::mlir_global(
                    context,
                    Region::new(),
                    TypeAttribute::new(llvm::r#type::pointer(context, 0)),
                    StringAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                    Attribute::parse(context, "#llvm.linkage<weak>")
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            );

            // Once we created the global pointer to the counters, we need to reallocate it so that it
            // can hold as many counters as libfuncs declared
            let u32_layout = get_integer_layout(32);
            let libfuncs_amount_bytes = layout_repeat(&u32_layout, libfunc_amount as usize)?
                .0
                .pad_to_align()
                .size();
            let libfuncs_amount_bytes =
                block.const_int(context, location, libfuncs_amount_bytes, 64)?;

            let array_counter_ptr_ptr = block.append_op_result(
                ods::llvm::mlir_addressof(
                    context,
                    llvm::r#type::pointer(context, 0),
                    FlatSymbolRefAttribute::new(
                        context,
                        LibfuncCounterBinding::ArrayCounter.symbol(),
                    ),
                    location,
                )
                .into(),
            )?;

            let array_counter_ptr =
                block.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

            let array_counter_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                context,
                array_counter_ptr,
                libfuncs_amount_bytes,
                location,
            )?)?;

            block.store(context, location, array_counter_ptr_ptr, array_counter_ptr)?;
        }

        let array_counter_ptr_ptr = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                location,
            )
            .into(),
        )?;

        // // return the pointer to array counter
        block.load(
            context,
            location,
            array_counter_ptr_ptr,
            llvm::r#type::pointer(context, 0),
        )
    }

    pub fn count_libfunc(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block<'_>,
        location: Location,
        libfunc_idx: usize,
        libfuncs_amount: u32,
    ) -> Result<()> {
        let u32_ty = IntegerType::new(context, 32).into();
        let k1 = block.const_int(context, location, 0, 32)?;

        let array_counter_ptr =
            self.get_array_counter(context, module, block, location, libfuncs_amount)?;
        let value_counter_ptr = block.gep(
            context,
            location,
            array_counter_ptr,
            &[GepIndex::Const(libfunc_idx as i32)],
            u32_ty,
        )?;

        let value_counter = block.load(context, location, value_counter_ptr, u32_ty)?;
        let value_incremented = block.addi(value_counter, k1, location)?;

        block.store(context, location, value_counter_ptr, value_incremented)?;

        Ok(())
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    let bindings = &[LibfuncCounterBinding::StoreArrayCounter];

    for binding in bindings {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}

pub mod libfunc_counter_runtime {
    use std::{
        collections::HashMap,
        sync::{LazyLock, Mutex},
    };

    use itertools::Itertools;
    use melior::{
        ir::{Block, Location, Module},
        Context,
    };

    use crate::{
        error::Result,
        metadata::{libfunc_counter::LibfuncCounterMeta, MetadataStorage},
    };

    /// Contains an array of vector for each execution completed
    pub static LIBFUNC_COUNTER: LazyLock<Mutex<HashMap<u64, Vec<u32>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    /// Increase the libfunc's counter given its index
    pub fn count_libfunc(
        context: &Context,
        module: &Module,
        block: &Block,
        location: Location,
        metadata: &mut MetadataStorage,
        libfunc_idx: usize,
        libfuncs_amount: u32,
    ) -> Result<()> {
        let libfunc_counter = metadata.get_or_insert_with(LibfuncCounterMeta::default);

        libfunc_counter.count_libfunc(
            context,
            module,
            block,
            location,
            libfunc_idx,
            libfuncs_amount,
        )
    }

    pub unsafe extern "C" fn store_array_counter(
        counter_id: u64,
        array_counter: *const u32,
        libfuncs_amount: u32,
    ) {
        let mut libfunc_counter = LIBFUNC_COUNTER.lock().unwrap();
        let vec = (0..libfuncs_amount)
            .map(|i| *array_counter.add(i as usize))
            .collect_vec();

        libfunc_counter.insert(counter_id, vec);
    }
}
