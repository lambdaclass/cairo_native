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
    metadata::libfunc_counter::libfunc_counter_runtime::CounterImpl,
    utils::BlockExt,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LibfuncCounterBinding {
    IncCounter,
    CounterId,
}

impl LibfuncCounterBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            LibfuncCounterBinding::IncCounter => "cairo_native__inc_counter__push_stmt",
            LibfuncCounterBinding::CounterId => "cairo_native__counter__profile_id",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            LibfuncCounterBinding::IncCounter => CounterImpl::count_libfunc as *const (),
            LibfuncCounterBinding::CounterId => ptr::null(),
        }
    }
}

#[derive(Clone, Default)]
pub struct LibfuncCounterMeta {
    active_map: HashSet<LibfuncCounterBinding>,
}

impl<'c, 'a> LibfuncCounterMeta {
    pub fn new() -> Self {
        Self {
            active_map: HashSet::new(),
        }
    }

    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored value.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    pub fn build_function(
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

    pub fn build_counter_id(
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

    pub fn count_libfunc(
        &mut self,
        context: &Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location,
        libfunc_idx: usize,
    ) -> Result<()> {
        let counter_id = self.build_counter_id(context, module, block, location)?;
        let libfunc_idx = block.const_int(context, location, libfunc_idx, 32)?;

        let function_ptr = self.build_function(
            context,
            module,
            block,
            location,
            LibfuncCounterBinding::IncCounter,
        )?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function_ptr])
                .add_operands(&[counter_id, libfunc_idx])
                .build()?,
        );

        Ok(())
    }

    // pub fn build_array_counter(
    //     context: &'c Context,
    //     module: &Module,
    //     block: &'a Block<'c>,
    //     location: Location<'c>,
    //     libfunc_count: u32,
    // ) {
    //     let array_ty = llvm::r#type::array(IntegerType::new(context, 32), libfunc_count);
    //     let (layout, align) = layout_repeat(get_integer_layout(32), libfunc_count)?;
    //     let array_counter_ptr = block.alloca1(context, location, array_ty, align)?;
    // }

    // pub fn count_libfunc(
    //     &self,
    //     context: &Context,
    //     registry: &Program,
    //     module: &Module,
    //     block: &'a Block<'c>,
    //     location: Location,
    //     metadata: &mut MetadataStorage,
    //     libfunc_id: ConcreteLibfuncId,
    // ) -> Result<()> {
    //     let u32_ty = IntegerType::new(context, 32);
    //     let array_ty = llvm::r#type::array(u32_ty, libfunc_count);
    //     let k1 = block.const_int(context, location, 1, 32)?;

    //     let array_counter_ptr = block.load(
    //         context,
    //         location,
    //         libfunc_counter.array_counter_ptr,
    //         array_ty,
    //     )?;
    //     let value_counter = block.gep(
    //         context,
    //         location,
    //         array_counter_ptr,
    //         GepIndex::Const(libfunc_id.id),
    //         u32_ty,
    //     )?;
    //     let value_incremented = block.addi(value_counter, k1, location)?;

    //     block.insert_value(
    //         context,
    //         location,
    //         array_counter_ptr,
    //         value_incremented,
    //         libfunc_id.id,
    //     )?;

    //     Ok(())
    // }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    let bindings = &[LibfuncCounterBinding::IncCounter];

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

    use melior::{
        ir::{BlockRef, Location, Module},
        Context,
    };

    use crate::{
        error::Result,
        metadata::{libfunc_counter::LibfuncCounterMeta, MetadataStorage},
    };

    pub static LIBFUNC_COUNTER: LazyLock<Mutex<HashMap<u64, CounterImpl>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    pub fn count_libfunc(
        context: &Context,
        module: &Module,
        block: &BlockRef,
        location: Location,
        metadata: &mut MetadataStorage,
        libfunc_idx: usize,
    ) -> Result<()> {
        let libfunc_counter = metadata.get_or_insert_with(LibfuncCounterMeta::default);

        libfunc_counter.count_libfunc(context, module, block, location, libfunc_idx)
    }

    #[derive(Default, Debug)]
    pub struct CounterImpl {
        pub array_counter: Vec<u32>,
    }

    impl CounterImpl {
        pub fn new(libfunc_amount: usize) -> Self {
            let array_counter =  vec![0u32; libfunc_amount];

            Self {
                array_counter,
            }
        }

        pub extern "C" fn count_libfunc(counter_id: u64, libfunc_idx: u32) {
            let index = libfunc_idx as usize;
            let mut libfunc_counter_map = LIBFUNC_COUNTER.lock().unwrap();
            let libfunc_counter = libfunc_counter_map.get_mut(&counter_id).unwrap();

            let counter = libfunc_counter.array_counter.get_mut(index).unwrap();

            *counter += 1;
        }
    }
}
