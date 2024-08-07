#![cfg(feature = "with-trace-dump")]

use crate::{
    block_ext::BlockExt,
    error::Result,
    starknet::{ArrayAbi, Felt252Abi},
    types::TypeBuilder,
    utils::next_multiple_of_usize,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        enm::EnumConcreteType,
        structure::StructConcreteType,
        types::InfoAndTypeConcreteType,
    },
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
    alloc::Layout,
    borrow::Borrow,
    cell::RefCell,
    collections::{HashMap, HashSet},
    mem::swap,
    ptr::NonNull,
    rc::Rc,
    sync::Weak,
};

pub struct InternalState {
    trace: RefCell<ProgramTrace>,
    state: RefCell<OrderedHashMap<VarId, sierra_emu::Value>>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
}

impl InternalState {
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
    State,
    Push,
}

type Type = Rc<InternalState>;

pub struct TraceDump {
    trace: Type,
    bindings: HashSet<TraceBinding>,
}

impl TraceDump {
    pub fn new(registry: ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            trace: Rc::new(InternalState::new(registry)),
            bindings: HashSet::default(),
        }
    }
    pub fn internal_state(&self) -> Rc<InternalState> {
        self.trace.clone()
    }

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
        if self.bindings.insert(TraceBinding::State) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            // internal state
                            llvm::r#type::pointer(context, 0),
                            // var id
                            IntegerType::new(context, 64).into(),
                            // value type
                            IntegerType::new(context, 64).into(),
                            // value ptr
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
                Rc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();
        let value_id = block.const_int(context, location, value_ty.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state"),
            &[state, var_id, value_id, value_ptr],
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
                Rc::downgrade(&self.trace).into_raw() as i64,
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
        if self.bindings.contains(&TraceBinding::State) {
            unsafe {
                engine.register_symbol("__trace__state", trace_state as *mut ());
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

extern "C" fn trace_state(
    state: *const InternalState,
    var_id: u64,
    value_type_id: u64,
    value_ptr: *const (),
) {
    let Some(state) = unsafe { Weak::from_raw(state) }.upgrade() else {
        return;
    };

    state.state.borrow_mut().insert(
        VarId::new(var_id),
        value_from_pointer(
            state.borrow(),
            &ConcreteTypeId::new(value_type_id),
            value_ptr,
        ),
    );
}

fn value_from_pointer(
    state: &InternalState,
    value_type_id: &ConcreteTypeId,
    value_ptr: *const (),
) -> sierra_emu::Value {
    let value_type = state.registry.get_type(value_type_id).unwrap();

    match value_type {
        CoreTypeConcrete::Felt252(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 32]>().as_ref().unwrap() };
            sierra_emu::Value::Felt(Felt::from_bytes_le(bytes))
        }
        CoreTypeConcrete::Uint8(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 1]>().as_ref().unwrap() };
            sierra_emu::Value::U8(u8::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint16(_) => todo!(),
        CoreTypeConcrete::Uint32(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 4]>().as_ref().unwrap() };
            sierra_emu::Value::U32(u32::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint64(_) => todo!(),
        CoreTypeConcrete::Uint128(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 16]>().as_ref().unwrap() };
            sierra_emu::Value::U128(u128::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => todo!(),
        CoreTypeConcrete::Sint16(_) => todo!(),
        CoreTypeConcrete::Sint32(_) => todo!(),
        CoreTypeConcrete::Sint64(_) => todo!(),
        CoreTypeConcrete::Sint128(_) => todo!(),
        CoreTypeConcrete::Array(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        })
        | CoreTypeConcrete::Span(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => {
            let member_stride = state
                .registry
                .get_type(inner_type_id)
                .unwrap()
                .layout(&state.registry)
                .unwrap()
                .pad_to_align()
                .size();

            let array = unsafe { value_ptr.cast::<ArrayAbi<()>>().as_ref().unwrap() };

            let length = (array.until - array.since) as usize;
            let start_ptr = unsafe { array.ptr.byte_add(array.since as usize * member_stride) };
            let mut data = Vec::with_capacity(length);

            for i in 0..length {
                let current_ptr = unsafe { start_ptr.byte_add(i * member_stride) };
                data.push(value_from_pointer(state, inner_type_id, current_ptr))
            }

            sierra_emu::Value::Array {
                ty: inner_type_id.clone(),
                data,
            }
        }
        CoreTypeConcrete::Struct(StructConcreteType { members, .. }) => {
            let mut layout = Layout::new::<()>();

            let mut data = Vec::with_capacity(members.len());

            for member_ty in members {
                let member = state.registry.get_type(member_ty).unwrap();
                let member_layout = member.layout(&state.registry).unwrap();

                let (new_layout, offset) = layout.extend(member_layout).unwrap();
                layout = new_layout;

                let current_ptr = unsafe { value_ptr.byte_add(offset) };

                data.push(value_from_pointer(state, member_ty, current_ptr))
            }

            sierra_emu::Value::Struct(data)
        }
        CoreTypeConcrete::Enum(EnumConcreteType { variants, .. }) => {
            let tag_layout = crate::utils::get_integer_layout(match variants.len() {
                0 => unreachable!("an enum without variants is not a valid type."),
                1 => 0,
                num_variants => (next_multiple_of_usize(num_variants.next_power_of_two(), 8) >> 3)
                    .try_into()
                    .unwrap(),
            });
            let tag_value = match variants.len() {
                0 => unreachable!("an enum without variants is not a valid type."),
                1 => 0,
                _ => match tag_layout.size() {
                    1 => *unsafe { value_ptr.cast::<u8>().as_ref().unwrap() } as usize,
                    2 => *unsafe { value_ptr.cast::<u16>().as_ref().unwrap() } as usize,
                    4 => *unsafe { value_ptr.cast::<u32>().as_ref().unwrap() } as usize,
                    8 => *unsafe { value_ptr.cast::<u64>().as_ref().unwrap() } as usize,
                    _ => unreachable!(),
                },
            };

            let payload_type_id = &variants[tag_value];
            let payload_layout = state
                .registry
                .get_type(payload_type_id)
                .unwrap()
                .layout(&state.registry)
                .unwrap();
            let (_, payload_offset) = tag_layout.extend(payload_layout).unwrap();
            let payload_ptr = unsafe { value_ptr.byte_add(payload_offset) };
            let payload = value_from_pointer(state, payload_type_id, payload_ptr);

            sierra_emu::Value::Enum {
                self_ty: value_type_id.clone(),
                index: tag_value,
                payload: Box::new(payload),
            }
        }
        CoreTypeConcrete::Felt252Dict(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        })
        | CoreTypeConcrete::SquashedFelt252Dict(InfoAndTypeConcreteType {
            ty: inner_type_id,
            ..
        }) => {
            let (dict, _) = unsafe {
                (*value_ptr.cast::<*const (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64)>())
                    .as_ref()
                    .unwrap()
            };

            let dict = build_dict(state, dict, inner_type_id);

            sierra_emu::Value::FeltDict {
                ty: inner_type_id.clone(),
                data: dict,
            }
        }
        CoreTypeConcrete::Felt252DictEntry(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => {
            let entry = unsafe { value_ptr.cast::<DictEntryAbi<()>>().as_ref().unwrap() };
            let key = Felt::from_bytes_le(&entry.key.0);
            let (dict, _) = unsafe { entry.dict_ptr.as_ref().unwrap() };

            let dict = build_dict(state, dict, inner_type_id);

            sierra_emu::Value::FeltDictEntry {
                ty: inner_type_id.clone(),
                data: dict,
                key,
            }
        }
        CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => value_from_pointer(state, inner_type_id, value_ptr),
        CoreTypeConcrete::GasBuiltin(_) => todo!(),
        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        CoreTypeConcrete::NonZero(_) => todo!(),
        CoreTypeConcrete::Nullable(_) => todo!(),
        CoreTypeConcrete::RangeCheck(_) => todo!(),
        CoreTypeConcrete::RangeCheck96(_) => todo!(),
        CoreTypeConcrete::Pedersen(_) => todo!(),
        CoreTypeConcrete::Poseidon(_) => todo!(),
        CoreTypeConcrete::StarkNet(_) => todo!(),
        CoreTypeConcrete::Bytes31(_) => todo!(),
        CoreTypeConcrete::BoundedInt(_) => todo!(),
        CoreTypeConcrete::Coupon(_) => todo!(),
        CoreTypeConcrete::Bitwise(_) => todo!(),
        CoreTypeConcrete::Box(_) => todo!(),
        CoreTypeConcrete::Circuit(_) => todo!(),
        CoreTypeConcrete::Const(_) => todo!(),
        CoreTypeConcrete::EcOp(_) => todo!(),
        CoreTypeConcrete::EcPoint(_) => todo!(),
        CoreTypeConcrete::EcState(_) => todo!(),
        CoreTypeConcrete::Uninitialized(InfoAndTypeConcreteType { ty, .. }) => {
            sierra_emu::Value::Uninitialized { ty: ty.clone() }
        }
        CoreTypeConcrete::SegmentArena(_) => sierra_emu::Value::Unit,
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

fn build_dict(
    state: &InternalState,
    dict: &HashMap<[u8; 32], NonNull<libc::c_void>>,
    inner_type_id: &ConcreteTypeId,
) -> HashMap<Felt, sierra_emu::Value> {
    let mut new_dict = HashMap::new();

    for (key, value_ptr) in dict {
        let felt = Felt::from_bytes_le(key);
        let value = value_from_pointer(state, inner_type_id, value_ptr.cast().as_ptr());

        new_dict.insert(felt, value);
    }
    new_dict
}

#[repr(C)]
#[derive(Debug)]
pub struct DictEntryAbi<T> {
    pub key: Felt252Abi,
    pub value_ptr: *const T,
    pub dict_ptr: *const (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
}
