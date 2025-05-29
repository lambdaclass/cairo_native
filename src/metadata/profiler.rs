#![cfg(feature = "with-libfunc-profiling")]

use crate::{
    error::{Error, Result},
    utils::BlockExt,
};
use cairo_lang_sierra::program::StatementIdx;
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::{self},
        ods,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, Value,
    },
    Context,
};
use std::{
    cell::{RefCell, UnsafeCell},
    collections::HashSet,
    ffi::c_void,
    mem,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ProfilerBinding {
    CallBack,
}

impl ProfilerBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            ProfilerBinding::CallBack => "cairo_native__profiler__callback",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            ProfilerBinding::CallBack => ProfilerImpl::callback as *const (),
        }
    }
}

#[derive(Clone)]
pub struct ProfilerMeta {
    active_map: RefCell<HashSet<ProfilerBinding>>,
    _private: (),
}

impl ProfilerMeta {
    pub fn new() -> Result<Self> {
        Ok(Self {
            active_map: RefCell::new(HashSet::new()),
            _private: (),
        })
    }

    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored value.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    fn build_function<'c, 'a>(
        &self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        binding: ProfilerBinding,
    ) -> Result<Value<'c, 'a>> {
        if !self.active_map.borrow().contains(&binding) {
            self.active_map.borrow_mut().insert(binding);

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

    /// Get the timestamp
    #[cfg(target_arch = "x86_64")]
    pub fn measure_timestamp<'c, 'a>(
        &self,
        context: &'c Context,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<(Value<'c, 'a>, Value<'c, 'a>)> {
        let i32_ty = IntegerType::new(context, 32).into();
        let i64_ty = IntegerType::new(context, 64).into();
        let k32 = block.const_int_from_type(context, location, 32, i64_ty)?;

        // edx:eax := TimeStampCounter   (clock value)
        // ecx     := IA32_TSC_AUX[31:0] (core ID)
        let value = block.append_op_result(
            OperationBuilder::new("llvm.inline_asm", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "asm_string"),
                        StringAttribute::new(context, "mfence\nrdtscp\nlfence").into(),
                    ),
                    (
                        Identifier::new(context, "has_side_effects"),
                        Attribute::unit(context),
                    ),
                    (
                        Identifier::new(context, "constraints"),
                        StringAttribute::new(context, "={edx},={eax},={ecx}").into(),
                    ),
                ])
                .add_results(&[llvm::r#type::r#struct(
                    context,
                    &[i32_ty, i32_ty, i32_ty],
                    false,
                )])
                .build()?,
        )?;
        let value_hi = block.extract_value(context, location, value, i32_ty, 0)?;
        let value_lo = block.extract_value(context, location, value, i32_ty, 1)?;
        let core_idx = block.extract_value(context, location, value, i32_ty, 2)?;

        let value_hi = block.extui(value_hi, i64_ty, location)?;
        let value_lo = block.extui(value_lo, i64_ty, location)?;
        let value = block.shli(value_hi, k32, location)?;
        let value = block.append_op_result(arith::ori(value, value_lo, location))?;

        Ok((value, core_idx))
    }

    /// Get the timestamp
    #[cfg(target_arch = "aarch64")]
    pub fn measure_timestamp<'c, 'a>(
        &self,
        context: &'c Context,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<(Value<'c, 'a>, Value<'c, 'a>)> {
        let i64_ty = IntegerType::new(context, 64).into();

        let value = block.append_op_result(
            OperationBuilder::new("llvm.inline_asm", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "asm_string"),
                        StringAttribute::new(context, "isb\nmrs $0, CNTVCT_EL0\nisb").into(),
                    ),
                    (
                        Identifier::new(context, "has_side_effects"),
                        Attribute::unit(context),
                    ),
                    (
                        Identifier::new(context, "constraints"),
                        StringAttribute::new(context, "=r").into(),
                    ),
                ])
                .add_results(&[i64_ty])
                .build()?,
        )?;
        let core_idx = block.const_int(context, location, 0, 64)?;

        Ok((value, core_idx))
    }

    /// Receives two timestamps, if they were originated in the same statement index
    /// the delta time between these two is calculated. If not, then the delta time is
    /// assigned to -1. Then it pushes the frame, which is made of the statement index
    /// the delta time.
    pub fn push_frame<'c>(
        &self,
        context: &'c Context,
        module: &Module,
        block: &Block<'c>,
        statement_idx: usize,
        t0: (Value<'c, '_>, Value<'c, '_>),
        t1: (Value<'c, '_>, Value<'c, '_>),
        location: Location<'c>,
    ) -> Result<()> {
        // If core idx matches:
        //   Calculate time delta.
        //   Write statement idx and time delta.
        // If core idx does not match:
        //   Write statement idx and -1.

        let i64_ty = IntegerType::new(context, 64).into();

        let statement_idx = block.const_int_from_type(context, location, statement_idx, i64_ty)?;
        let is_same_core = block.cmpi(context, CmpiPredicate::Eq, t0.1, t1.1, location)?;

        let delta_value = block.append_op_result(arith::subi(t1.0, t0.0, location))?;
        let invalid_value = block.const_int_from_type(context, location, u64::MAX, i64_ty)?;
        let delta_value = block.append_op_result(arith::select(
            is_same_core,
            delta_value,
            invalid_value,
            location,
        ))?;

        let callback_ptr =
            self.build_function(context, module, block, location, ProfilerBinding::CallBack)?;

        block.append_operation(
            ods::llvm::call(
                context,
                &[callback_ptr, statement_idx, delta_value],
                location,
            )
            .into(),
        );
        Ok(())
    }
}

thread_local! {
    static PROFILER_IMPL: UnsafeCell<ProfilerImpl> = const { UnsafeCell::new(ProfilerImpl::new()) };
}

pub struct ProfilerImpl {
    trace: Vec<(StatementIdx, u64)>,
}

impl ProfilerImpl {
    const fn new() -> Self {
        Self { trace: Vec::new() }
    }

    pub fn take() -> Vec<(StatementIdx, u64)> {
        PROFILER_IMPL.with(|x| {
            let x = unsafe { &mut *x.get() };

            let mut trace = Vec::new();
            mem::swap(&mut x.trace, &mut trace);

            trace
        })
    }

    // Push a profiler frame
    pub extern "C" fn callback(statement_idx: u64, tick_delta: u64) {
        PROFILER_IMPL.with(|x| {
            let x = unsafe { &mut *x.get() };

            x.trace
                .push((StatementIdx(statement_idx as usize), tick_delta));
        });
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    let bindings = &[ProfilerBinding::CallBack];

    for binding in bindings {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}
