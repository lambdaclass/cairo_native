#![cfg(feature = "with-trace-dump")]

use crate::{
    block_ext::BlockExt,
    error::Result,
    starknet::{ArrayAbi, Felt252Abi},
    types::TypeBuilder,
    utils::{next_multiple_of_usize, RangeExt},
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
use libc::c_void;
use melior::{
    dialect::{func, llvm, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, Region,
    },
    Context, ExecutionEngine,
};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::One;
use sierra_emu::{ProgramTrace, StateDump, Value};
use starknet_types_core::felt::Felt;
use std::{
    alloc::Layout,
    borrow::Borrow,
    cell::RefCell,
    collections::{HashMap, HashSet},
    mem::swap,
    ptr::NonNull,
    rc::Rc,
    slice,
    sync::Weak,
};

pub struct InternalState {
    trace: RefCell<ProgramTrace>,
    state: RefCell<OrderedHashMap<VarId, Value>>,
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
        value_ptr: melior::ir::Value,
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
    value_ptr: *mut c_void,
) {
    let Some(state) = unsafe { Weak::from_raw(state) }.upgrade() else {
        return;
    };
    let Some(value_ptr) = NonNull::new(value_ptr) else {
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
    value_ptr: NonNull<c_void>,
) -> Value {
    let value_type = state.registry.get_type(value_type_id).unwrap();

    match value_type {
        CoreTypeConcrete::Felt252(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 32]>().as_ref() };
            Value::Felt(Felt::from_bytes_le(bytes))
        }
        CoreTypeConcrete::Uint8(_) => unsafe { Value::U8(value_ptr.cast::<u8>().read()) },
        CoreTypeConcrete::Uint16(_) => todo!(),
        CoreTypeConcrete::Uint32(_) => unsafe { Value::U32(value_ptr.cast::<u32>().read()) },
        CoreTypeConcrete::Uint64(_) => todo!(),
        CoreTypeConcrete::Uint128(_) => unsafe { Value::U128(value_ptr.cast::<u128>().read()) },
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => unsafe { Value::I8(value_ptr.cast::<i8>().read()) },
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

            let array = unsafe { value_ptr.cast::<ArrayAbi<c_void>>().as_ref() };

            let length = (array.until - array.since) as usize;
            let start_ptr = unsafe { array.ptr.byte_add(array.since as usize * member_stride) };
            let mut data = Vec::with_capacity(length);

            for i in 0..length {
                let current_ptr =
                    unsafe { NonNull::new(start_ptr.byte_add(i * member_stride)).unwrap() };
                data.push(value_from_pointer(state, inner_type_id, current_ptr))
            }

            Value::Array {
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

                let current_ptr =
                    unsafe { NonNull::new(value_ptr.as_ptr().byte_add(offset)).unwrap() };

                data.push(value_from_pointer(state, member_ty, current_ptr))
            }

            Value::Struct(data)
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
                    1 => unsafe { *value_ptr.cast::<u8>().as_ref() as usize },
                    2 => unsafe { *value_ptr.cast::<u16>().as_ref() as usize },
                    4 => unsafe { *value_ptr.cast::<u32>().as_ref() as usize },
                    8 => unsafe { *value_ptr.cast::<u64>().as_ref() as usize },
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
            let payload_ptr =
                unsafe { NonNull::new(value_ptr.as_ptr().byte_add(payload_offset)).unwrap() };
            let payload = value_from_pointer(state, payload_type_id, payload_ptr);

            Value::Enum {
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
                (*value_ptr
                    .as_ptr()
                    .cast::<*const (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64)>())
                .as_ref()
                .unwrap()
            };

            let dict = build_dict(state, dict, inner_type_id);

            Value::FeltDict {
                ty: inner_type_id.clone(),
                data: dict,
            }
        }
        CoreTypeConcrete::Felt252DictEntry(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => {
            let entry = unsafe { value_ptr.cast::<DictEntryAbi<()>>().as_ref() };
            let key = Felt::from_bytes_le(&entry.key.0);
            let (dict, _) = unsafe { entry.dict_ptr.as_ref().unwrap() };

            let dict = build_dict(state, dict, inner_type_id);

            Value::FeltDictEntry {
                ty: inner_type_id.clone(),
                data: dict,
                key,
            }
        }
        CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => value_from_pointer(state, inner_type_id, value_ptr),
        CoreTypeConcrete::Box(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => {
            let inner_ptr =
                unsafe { NonNull::new(*value_ptr.as_ptr().cast::<*mut c_void>()).unwrap() };

            value_from_pointer(state, inner_type_id, inner_ptr)
        }
        CoreTypeConcrete::GasBuiltin(_) => todo!(),
        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        CoreTypeConcrete::NonZero(info) => value_from_pointer(state, &info.ty, value_ptr),
        CoreTypeConcrete::Nullable(_) => todo!(),
        CoreTypeConcrete::RangeCheck(_) => Value::Unit,
        CoreTypeConcrete::RangeCheck96(_) => todo!(),
        CoreTypeConcrete::Pedersen(_) => todo!(),
        CoreTypeConcrete::Poseidon(_) => todo!(),
        CoreTypeConcrete::StarkNet(_) => todo!(),
        CoreTypeConcrete::Bytes31(_) => todo!(),
        CoreTypeConcrete::BoundedInt(info) => {
            let mut data = BigUint::from_bytes_le(unsafe {
                slice::from_raw_parts(
                    value_ptr.cast::<u8>().as_ptr(),
                    (info.range.offset_bit_width().next_multiple_of(8) >> 3) as usize,
                )
            })
            .to_bigint()
            .unwrap();

            data &= (BigInt::one() << info.range.offset_bit_width()) - BigInt::one();
            data += &info.range.lower;

            Value::BoundedInt {
                value: data,
                range: info.range.lower.clone()..info.range.upper.clone(),
            }
        }
        CoreTypeConcrete::Coupon(_) => todo!(),
        CoreTypeConcrete::Bitwise(_) => todo!(),
        CoreTypeConcrete::Circuit(_) => todo!(),
        CoreTypeConcrete::Const(_) => todo!(),
        CoreTypeConcrete::EcOp(_) => todo!(),
        CoreTypeConcrete::EcPoint(_) => todo!(),
        CoreTypeConcrete::EcState(_) => todo!(),
        CoreTypeConcrete::Uninitialized(InfoAndTypeConcreteType { ty, .. }) => {
            Value::Uninitialized { ty: ty.clone() }
        }
        CoreTypeConcrete::SegmentArena(_) => Value::Unit,
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
) -> HashMap<Felt, Value> {
    let mut new_dict = HashMap::new();

    for (key, value_ptr) in dict {
        let felt = Felt::from_bytes_le(key);
        let value = value_from_pointer(state, inner_type_id, value_ptr.cast());

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
