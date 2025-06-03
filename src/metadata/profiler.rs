#![cfg(feature = "with-libfunc-profiling")]

use crate::{
    error::{Error, Result},
    utils::BlockExt,
};
use cairo_lang_sierra::{
    ids::ConcreteLibfuncId,
    program::{Program, Statement, StatementIdx},
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::{self},
        memref, ods,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{IntegerType, MemRefType},
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, Value,
    },
    Context,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr,
    sync::{LazyLock, Mutex},
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ProfilerBinding {
    PushStmt,
    TraceId,
}

impl ProfilerBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            ProfilerBinding::PushStmt => "cairo_native__profiler__push_stmt",
            ProfilerBinding::TraceId => "cairo_native__profiler__trace_id",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            ProfilerBinding::PushStmt => ProfileImpl::push_stmt as *const (),
            ProfilerBinding::TraceId => ptr::null(),
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
        if self.active_map.borrow_mut().insert(binding) {
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

    pub fn build_trace_id<'c, 'a>(
        &self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        if self
            .active_map
            .borrow_mut()
            .insert(ProfilerBinding::TraceId)
        {
            module.body().append_operation(memref::global(
                context,
                ProfilerBinding::TraceId.symbol(),
                None,
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                None,
                false,
                None,
                location,
            ));
        }

        let trace_id_ptr = block
            .append_op_result(memref::get_global(
                context,
                ProfilerBinding::TraceId.symbol(),
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                location,
            ))
            .unwrap();

        block.append_op_result(memref::load(trace_id_ptr, &[], location))
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

    #[allow(clippy::too_many_arguments)]
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

        let trace_id = self.build_trace_id(context, module, block, location)?;

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
            self.build_function(context, module, block, location, ProfilerBinding::PushStmt)?;

        block.append_operation(
            ods::llvm::call(
                context,
                &[callback_ptr, trace_id, statement_idx, delta_value],
                location,
            )
            .into(),
        );
        Ok(())
    }
}

pub static LIBFUNC_PROFILE: LazyLock<Mutex<HashMap<u64, ProfileImpl>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// This represents a libfunc's profile, which has the following structure:
///
/// `Vec<(libfunc_id, (samples_number, total_execution_time, quartiles, average_execution_time, standard_deviations))>``
type ProfileInfo = Vec<(ConcreteLibfuncId, (u64, u64, [u64; 5], f64, f64))>;

pub struct ProfileImpl {
    pub trace: Vec<(StatementIdx, u64)>,
    sierra_program: Program,
}

impl ProfileImpl {
    pub fn new(sierra_program: Program) -> Self {
        Self {
            trace: Vec::new(),
            sierra_program,
        }
    }

    // Push a profiler frame
    pub extern "C" fn push_stmt(trace_id: u64, statement_idx: u64, tick_delta: u64) {
        let mut profiler = LIBFUNC_PROFILE.lock().unwrap();

        let Some(profiler) = profiler.get_mut(&trace_id) else {
            eprintln!("Could not find libfunc profiler!");
            return;
        };

        profiler
            .trace
            .push((StatementIdx(statement_idx as usize), tick_delta));
    }

    pub fn process(&self) -> ProfileInfo {
        let mut trace = HashMap::<ConcreteLibfuncId, (Vec<u64>, u64)>::new();

        for (statement_idx, tick_delta) in self.trace.iter() {
            if let Statement::Invocation(invocation) =
                &self.sierra_program.statements[statement_idx.0]
            {
                let (tick_deltas, extra_count) =
                    trace.entry(invocation.libfunc_id.clone()).or_default();

                // A tick_delta equal to u64::MAX implies it is invalid, so we don't take it
                // into account
                if *tick_delta != u64::MAX {
                    tick_deltas.push(*tick_delta);
                } else {
                    *extra_count += 1;
                }
            }
        }

        let mut trace = trace
            .into_iter()
            .map(|(libfunc_id, (mut tick_deltas, extra_count))| {
                tick_deltas.sort();

                // Drop outliers.
                {
                    let q1 = tick_deltas[tick_deltas.len() / 4];
                    let q3 = tick_deltas[3 * tick_deltas.len() / 4];
                    let iqr = q3 - q1;

                    let q1_thr = q1.saturating_sub(iqr + iqr / 2);
                    let q3_thr = q3 + (iqr + iqr / 2);

                    tick_deltas.retain(|x| *x >= q1_thr && *x <= q3_thr);
                }

                // Compute the quartiles.
                let quartiles = [
                    *tick_deltas.first().unwrap(),
                    tick_deltas[tick_deltas.len() / 4],
                    tick_deltas[tick_deltas.len() / 2],
                    tick_deltas[3 * tick_deltas.len() / 4],
                    *tick_deltas.last().unwrap(),
                ];

                // Compuite the average.
                let average =
                    tick_deltas.iter().copied().sum::<u64>() as f64 / tick_deltas.len() as f64;

                // Compute the standard deviation.
                let std_dev = {
                    let sum = tick_deltas
                        .iter()
                        .copied()
                        .map(|x| x as f64)
                        .map(|x| (x - average))
                        .map(|x| x * x)
                        .sum::<f64>();
                    sum / (tick_deltas.len() as u64 + extra_count) as f64
                };

                (
                    libfunc_id,
                    (
                        tick_deltas.len() as u64 + extra_count,
                        tick_deltas.iter().sum::<u64>()
                            + (extra_count as f64 * average).round() as u64,
                        quartiles,
                        average,
                        std_dev,
                    ),
                )
            })
            .collect::<Vec<_>>();

        // Sort libfuncs by the order in which they are declared.
        trace.sort_by_key(|(libfunc_id, _)| {
            self.sierra_program
                .libfunc_declarations
                .iter()
                .enumerate()
                .find_map(|(i, x)| (&x.id == libfunc_id).then_some(i))
                .unwrap()
        });

        trace
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    let bindings = &[ProfilerBinding::PushStmt];

    for binding in bindings {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}
