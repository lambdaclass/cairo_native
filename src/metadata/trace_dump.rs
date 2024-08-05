#![cfg(feature = "with-trace-dump")]

use crate::{block_ext::BlockExt, error::Result, starknet::ArrayAbi};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use melior::{
    dialect::{func, llvm, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, Region, Value,
    },
    Context, ExecutionEngine,
};
use sierra_emu::{ProgramTrace, StateDump};
use starknet_types_core::felt::Felt;
use std::{
    cell::RefCell,
    collections::HashSet,
    mem::swap,
    sync::{Arc, Weak},
};

pub struct InternalState<'a> {
    trace: RefCell<ProgramTrace<'a>>,
    state: RefCell<OrderedHashMap<VarId, sierra_emu::Value<'a>>>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
}

impl<'a> InternalState<'a> {
    pub fn new(registry: ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            trace: RefCell::default(),
            state: RefCell::default(),
            registry,
        }
    }

    pub fn extract(&self) -> ProgramTrace {
        self.trace.borrow().clone()
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum TraceBinding {
    StateFelt,
    StateU128,
    StateU32,
    StateU8,

    Push,
}

pub struct TraceDump<'a> {
    trace: Arc<InternalState<'a>>,
    bindings: HashSet<TraceBinding>,
}

impl<'a> TraceDump<'a> {
    pub fn new(registry: ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            trace: Arc::new(InternalState::new(registry)),
            bindings: HashSet::default(),
        }
    }
    pub fn internal_state(&self) -> Arc<InternalState<'a>> {
        self.trace.clone()
    }

    pub fn build_state_felt252(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::StateFelt) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state_felt252"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
                            IntegerType::new(context, 64).into(),
                            llvm::r#type::pointer(context, 0),
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

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state_felt252"),
            &[state, var_id, value_ptr],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn build_state_u128(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::StateU128) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state_u128"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
                            IntegerType::new(context, 64).into(),
                            llvm::r#type::pointer(context, 0),
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

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state_u128"),
            &[state, var_id, value_ptr],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn build_state_u32(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::StateU32) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state_u32"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
                            IntegerType::new(context, 64).into(),
                            llvm::r#type::pointer(context, 0),
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

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state_u32"),
            &[state, var_id, value_ptr],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn build_state_u8(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::StateU8) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state_u8"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
                            IntegerType::new(context, 64).into(),
                            llvm::r#type::pointer(context, 0),
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

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state_u8"),
            &[state, var_id, value_ptr],
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
        if self.bindings.insert(TraceBinding::Push) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__push"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
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

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let statement_idx = block.const_int(context, location, statement_idx.0, 64)?;

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__push"),
            &[state, statement_idx],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn register_impls(&self, engine: &ExecutionEngine) {
        if self.bindings.contains(&TraceBinding::StateFelt) {
            unsafe {
                engine.register_symbol(
                    "__trace__state_felt252",
                    trace_state_felt252 as *const fn(*const InternalState, u64, &[u8; 32]) -> ()
                        as *mut (),
                );
            }
        }

        if self.bindings.contains(&TraceBinding::StateU128) {
            unsafe {
                engine.register_symbol("__trace__state_u128", trace_state_u128 as *mut ());
            }
        }

        if self.bindings.contains(&TraceBinding::StateU32) {
            unsafe {
                engine.register_symbol("__trace__state_u32", trace_state_u32 as *mut ());
            }
        }

        if self.bindings.contains(&TraceBinding::StateU8) {
            unsafe {
                engine.register_symbol("__trace__state_u8", trace_state_u8 as *mut ());
            }
        }

        if !self.bindings.is_empty() {
            unsafe {
                engine.register_symbol(
                    "__trace__push",
                    trace_push as *const fn(*const InternalState) -> () as *mut (),
                );
            }
        }
    }
}

extern "C" fn trace_state_felt252(state: *const InternalState, var_id: u64, value: &[u8; 32]) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut state = state.state.borrow_mut();
        state.insert(
            VarId::new(var_id),
            sierra_emu::Value::Felt(Felt::from_bytes_le(value)),
        );
    }
}

extern "C" fn trace_state_u128(state: *const InternalState, var_id: u64, value: &[u8; 16]) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut state = state.state.borrow_mut();
        state.insert(
            VarId::new(var_id),
            sierra_emu::Value::U128(u128::from_le_bytes(*value)),
        );
    }
}

extern "C" fn trace_state_u32(state: *const InternalState, var_id: u64, value: &[u8; 4]) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut state = state.state.borrow_mut();
        state.insert(
            VarId::new(var_id),
            sierra_emu::Value::U32(u32::from_le_bytes(*value)),
        );
    }
}

extern "C" fn trace_state_u8(state: *const InternalState, var_id: u64, value: &u8) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut state = state.state.borrow_mut();
        state.insert(VarId::new(var_id), sierra_emu::Value::U8(*value));
    }
}

extern "C" fn trace_push(state: *const InternalState, statement_idx: usize) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut items = OrderedHashMap::default();
        swap(&mut items, &mut *state.state.borrow_mut());

        state
            .trace
            .borrow_mut()
            .push(StateDump::new(StatementIdx(statement_idx), items));
    }
}
