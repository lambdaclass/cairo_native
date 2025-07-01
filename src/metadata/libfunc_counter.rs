#![cfg(feature = "with-libfunc-counter")]

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
    utils::{BlockExt, GepIndex},
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
            LibfuncCounterBinding::StoreArrayCounter => {
                "cairo_native__store_array_counter__push_stmt"
            }
            LibfuncCounterBinding::CounterId => "cairo_native__counter__profile_id",
            LibfuncCounterBinding::ArrayCounter => "cairo_native__array_counter__profile_id",
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

    pub fn build_counter_id<'c, 'a>(
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

    pub fn store_array_counter<'c, 'a>(
        &mut self,
        context: &Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location,
        libfunc_amount: u32
    ) -> Result<()> {
        let array_ty = llvm::r#type::array(IntegerType::new(context, 32).into(), libfunc_amount);

        let counter_id = self.build_counter_id(context, module, &block, location)?;
        let function_ptr = self.build_function(
            context,
            module,
            &block,
            location,
            LibfuncCounterBinding::StoreArrayCounter,
        )?;
        let lifuncs_amount = block.const_int(context, location, libfunc_amount, 32)?;
        // by this time, the array counter should be initialized
        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                location,
            )
            .into(),
        )?;

        let array_counter_ptr = block.load(context, location, global_address, array_ty)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function_ptr])
                .add_operands(&[counter_id, array_counter_ptr, lifuncs_amount])
                .build()?,
        );

        Ok(())
    }

    fn build_array_counter<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        libfunc_amount: u32,
    ) -> Result<()> {
        let array_ty = llvm::r#type::array(IntegerType::new(context, 32).into(), libfunc_amount);
        let k0 = block.const_int(context, location, 0, 32)?;

        module.body().append_operation(
            ods::llvm::mlir_global(
                context,
                Region::new(),
                TypeAttribute::new(array_ty),
                StringAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                Attribute::parse(context, "#llvm.linkage<weak>")
                    .ok_or(Error::ParseAttributeError)?,
                location,
            )
            .into(),
        );

        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                location,
            )
            .into(),
        )?;

        let array_counter_ptr = block.load(context, location, global_address, array_ty)?;

        block.insert_values(
            context,
            location,
            array_counter_ptr,
            &vec![k0; libfunc_amount as usize],
        )?;

        Ok(())
    }

    pub fn count_libfunc<'c, 'a>(
        &mut self,
        context: &Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location,
        libfunc_idx: usize,
        libfuncs_amount: u32,
    ) -> Result<()> {
        if self.active_map.insert(LibfuncCounterBinding::ArrayCounter) {
            self.build_array_counter(context, module, block, location, libfuncs_amount)?;
        }

        let u32_ty = IntegerType::new(context, 32).into();
        let array_ty = llvm::r#type::array(u32_ty, libfuncs_amount);
        let k1 = block.const_int(context, location, 1, 32)?;

        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, LibfuncCounterBinding::ArrayCounter.symbol()),
                location,
            )
            .into(),
        )?;

        let array_counter_ptr = block.load(context, location, global_address, array_ty)?;

        let value_counter = block.gep(
            context,
            location,
            array_counter_ptr,
            &[GepIndex::Const(libfunc_idx as i32)],
            u32_ty,
        )?;
        let value_incremented = block.addi(value_counter, k1, location)?;

        block.insert_value(
            context,
            location,
            array_counter_ptr,
            value_incremented,
            libfunc_idx,
        )?;

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

    pub static LIBFUNC_COUNTER: LazyLock<Mutex<HashMap<u64, Vec<u32>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

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
        array_counter: &[u32],
        libfuncs_amount: u32,
    ) {
        let mut libfunc_counter = LIBFUNC_COUNTER.lock().unwrap();

        libfunc_counter.insert(counter_id, array_counter.to_vec());
    }
}
