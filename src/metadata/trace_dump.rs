#![cfg(feature = "with-trace-dump")]

use crate::{error::Result, utils::BlockExt};
use cairo_lang_sierra::{
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
};
use melior::{
    dialect::{func, llvm, memref},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
        Block, Identifier, Location, Module, Region, Value,
    },
    Context, ExecutionEngine,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum TraceBinding {
    State,
    Push,
    TraceId,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TraceDumpMeta {
    bindings: HashSet<TraceBinding>,
}

impl TraceDumpMeta {
    #[allow(clippy::too_many_arguments)]
    pub fn build_state(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ty: &ConcreteTypeId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        self.build_trace_id(context, module)?;
        if self.bindings.insert(TraceBinding::State) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__trace_dump__state"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            IntegerType::new(context, 64).into(), // Trace ID.
                            IntegerType::new(context, 64).into(), // Var ID.
                            IntegerType::new(context, 64).into(), // Value type (`ConcreteTypeId`).
                            llvm::r#type::pointer(context, 0),    // Value pointer.
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

        let trace_id = block
            .append_op_result(memref::get_global(
                context,
                "TRACE_DUMP__TRACE_ID",
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                location,
            ))
            .unwrap();
        let trace_id = block.append_op_result(memref::load(trace_id, &[], location))?;
        let var_id = block.const_int(context, location, var_id.id, 64)?;
        let value_ty = block.const_int(context, location, value_ty.id, 64)?;
        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__trace_dump__state"),
            &[trace_id, var_id, value_ty, value_ptr],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn build_push(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        statement_idx: StatementIdx,
        location: Location,
    ) -> Result<()> {
        self.build_trace_id(context, module)?;
        if self.bindings.insert(TraceBinding::Push) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "cairo_native__trace_dump__push"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            IntegerType::new(context, 64).into(), // Trace ID.
                            IntegerType::new(context, 64).into(), // Statement index.
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

        let trace_id = block
            .append_op_result(memref::get_global(
                context,
                "TRACE_DUMP__TRACE_ID",
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                location,
            ))
            .unwrap();
        let trace_id = block.append_op_result(memref::load(trace_id, &[], location))?;
        let statement_idx = block.const_int(context, location, statement_idx.0, 64)?;
        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cairo_native__trace_dump__push"),
            &[trace_id, statement_idx],
            &[],
            location,
        ));

        Ok(())
    }

    fn build_trace_id(&mut self, context: &Context, module: &Module) -> Result<()> {
        if self.bindings.insert(TraceBinding::TraceId) {
            module.body().append_operation(memref::global(
                context,
                "TRACE_DUMP__TRACE_ID",
                None,
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                None,
                false,
                None,
                Location::unknown(context),
            ));
        }

        Ok(())
    }

    pub fn register_impls(&self, engine: &ExecutionEngine) {
        if self.bindings.contains(&TraceBinding::State) {
            unsafe {
                engine.register_symbol(
                    "cairo_native__trace_dump__push",
                    cairo_native_runtime::trace_dump::cairo_native__trace_dump__push as *mut (),
                );
            }
        }

        if !self.bindings.is_empty() {
            unsafe {
                engine.register_symbol(
                    "cairo_native__trace_dump__state",
                    cairo_native_runtime::trace_dump::cairo_native__trace_dump__state as *mut (),
                );
            }
        }
    }
}
