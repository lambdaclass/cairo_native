#![cfg(feature = "with-libfunc-counter")]
//! The libfunc counter feature is used to generate information counting how many time a libfunc has been called.
//!
//! When this feature is used, the compiler will call one main method:
//!
//! 1. `count_libfunc`: called before every libfunc execution. This method will handle the counting. Given the index
//!    of a libfunc (relative to its declaration order), it accesses the array of counters and updates the counter.
//!
//! In the context of Starknet contracts, we need to add support for building the array of counters for multiple executions.
//! To do so, we need one important element which must be set before every contract execution:
//!
//! * A counter to track the ID of the current array of counter, which gets updated every time we switch to another
//!   contract. Since a contract can call other contracts, we need a way of restoring the counter after every execution.
//!
//! * An array-of-counters guard. Every time a new entrypoint is executed, a new array of counters needs to be created.
//!   The guard keeps the last array that was used to restore it once the inner entrypoint execution has finished.
//!
//! See `cairo-native-run` for an example on how to do it.
use std::collections::HashSet;

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
    utils::{BlockExt, GepIndex},
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LibfuncCounterBinding {
    CounterId,
    GetCounterArray,
}

impl LibfuncCounterBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            LibfuncCounterBinding::CounterId => "cairo_native__counter_id",
            LibfuncCounterBinding::GetCounterArray => "cairo_native__get_counters_array",
        }
    }

    pub const fn function_ptr(self) -> *const () {
        match self {
            LibfuncCounterBinding::CounterId => std::ptr::null(),
            LibfuncCounterBinding::GetCounterArray => {
                libfunc_counter_runtime::get_counters_array as *const ()
            }
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
    /// For the function to be available, `setup_runtime` must be called before running the module.
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

    /// Returns the array of counters.
    fn get_array_counter<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        self.build_counter_id(context, module, block, location)?;

        let function_ptr = self.build_function(
            context,
            module,
            block,
            location,
            LibfuncCounterBinding::GetCounterArray,
        )?;

        block.append_op_result(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function_ptr])
                .add_results(&[llvm::r#type::pointer(context, 0)])
                .build()?,
        )
    }

    pub fn count_libfunc(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block<'_>,
        location: Location,
        libfunc_idx: usize,
    ) -> Result<()> {
        let u32_ty = IntegerType::new(context, 32).into();
        let k1 = block.const_int(context, location, 1, 32)?;

        let array_counter_ptr = self.get_array_counter(context, module, block, location)?;

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

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut libc::c_void>) {
    let bindings = &[LibfuncCounterBinding::GetCounterArray];

    for binding in bindings {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}

pub mod libfunc_counter_runtime {
    use core::slice;
    use std::{
        cell::Cell,
        collections::HashMap,
        sync::{LazyLock, Mutex},
    };

    use melior::{
        ir::{Block, Location, Module},
        Context,
    };

    use crate::{
        error::Result,
        metadata::{libfunc_counter::LibfuncCounterMeta, MetadataStorage},
        utils::{libc_free, libc_malloc},
    };

    /// Contains an array of vector for each execution completed.
    pub static LIBFUNC_COUNTER: LazyLock<Mutex<HashMap<u64, Vec<u32>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    thread_local! {
        pub(crate) static COUNTERS_ARRAY: Cell<*mut u32> = const {
            // This value will be overritten before executing the code
            Cell::new(std::ptr::null_mut())
        };
    }

    /// In the context of Starknet, a contract may call another. This means we
    /// need as many arrays of counters as call contracts are invoked during execution.
    /// This struct is used to hold the current array before calling the next contract
    /// so that it can then be restored.
    pub struct CountersArrayGuard(pub *mut u32);

    impl CountersArrayGuard {
        pub fn init(libfuncs_amount: usize) -> CountersArrayGuard {
            let u32_libfuncs_amount = libfuncs_amount * 4;
            let new_array: *mut u32 = unsafe { libc_malloc(u32_libfuncs_amount).cast() };

            // All positions in the array must be initialized with 0. Since
            // some libfuncs declared may not be called, their respective counter
            // won't be updated.
            for i in 0..libfuncs_amount {
                unsafe { *(new_array.add(i)) = 0 };
            }

            Self(COUNTERS_ARRAY.replace(new_array))
        }
    }

    impl Drop for CountersArrayGuard {
        fn drop(&mut self) {
            COUNTERS_ARRAY.set(self.0);
        }
    }

    /// Update the libfunc's counter based on its index, relative to the order of declaration.
    pub fn count_libfunc(
        context: &Context,
        module: &Module,
        block: &Block,
        location: Location,
        metadata: &mut MetadataStorage,
        libfunc_idx: usize,
    ) -> Result<()> {
        let libfunc_counter = metadata.get_or_insert_with(LibfuncCounterMeta::default);

        libfunc_counter.count_libfunc(context, module, block, location, libfunc_idx)
    }

    pub extern "C" fn get_counters_array() -> *mut u32 {
        COUNTERS_ARRAY.with(|x| x.get())
    }

    /// Converts the pointer to the counters into a Rust `Vec` and store it. Then, it frees the pointer.
    ///
    /// This method should be called at the end of an entrypoint execution.
    pub unsafe fn store_and_free_counters_array(counter_id_ptr: *mut u64, libfuncs_amount: usize) {
        let counter_array_ptr = get_counters_array();
        let counters_vec = slice::from_raw_parts(counter_array_ptr, libfuncs_amount).to_vec();

        LIBFUNC_COUNTER
            .lock()
            .unwrap()
            .insert(*counter_id_ptr, counters_vec);

        libc_free(counter_array_ptr as *mut libc::c_void);
    }
}
