//! # JIT params and return values de/serialization

//! A Rusty interface to provide parameters to JIT calls.

use crate::{
    error::jit_engine::{make_type_builder_error, ErrorImpl, RunnerError},
    types::{felt252::PRIME, TypeBuilder},
    utils::{felt252_bigint, get_integer_layout, layout_repeat, next_multiple_of_usize},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use educe::Educe;
use num_bigint::{BigInt, Sign};
use starknet_types_core::felt::Felt;
use std::{alloc::Layout, collections::HashMap, ops::Neg, ptr::NonNull};

/// A JitValue is a value that can be passed to the JIT engine as an argument or received as a result.
///
/// They map to the cairo/sierra types.
///
/// The debug_name field on some variants is `Some` when receiving a [`JitValue`] as a result.
///
/// A Boxed value or a non-null Nullable value is returned with it's inner value.
#[derive(Debug, Clone, Educe)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[educe(Eq, PartialEq)]
pub enum JitValue {
    Felt252(Felt),
    /// all elements need to be same type
    Array(Vec<Self>),
    Struct {
        fields: Vec<Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    }, // element types can differ
    Enum {
        tag: usize,
        value: Box<Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    },
    Felt252Dict {
        value: HashMap<Felt, Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    },
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
    Sint8(i8),
    Sint16(i16),
    Sint32(i32),
    Sint64(i64),
    Sint128(i128),
    EcPoint(Felt, Felt),
    EcState(Felt, Felt, Felt, Felt),
    Secp256K1Point {
        x: (u128, u128),
        y: (u128, u128),
    },
    Secp256R1Point {
        x: (u128, u128),
        y: (u128, u128),
    },
    /// Used as return value for Nullables that are null.
    Null,
}

// Conversions

impl From<Felt> for JitValue {
    fn from(value: Felt) -> Self {
        Self::Felt252(value)
    }
}

impl From<u8> for JitValue {
    fn from(value: u8) -> Self {
        Self::Uint8(value)
    }
}

impl From<u16> for JitValue {
    fn from(value: u16) -> Self {
        Self::Uint16(value)
    }
}

impl From<u32> for JitValue {
    fn from(value: u32) -> Self {
        Self::Uint32(value)
    }
}

impl From<u64> for JitValue {
    fn from(value: u64) -> Self {
        Self::Uint64(value)
    }
}

impl From<u128> for JitValue {
    fn from(value: u128) -> Self {
        Self::Uint128(value)
    }
}

impl From<i8> for JitValue {
    fn from(value: i8) -> Self {
        Self::Sint8(value)
    }
}

impl From<i16> for JitValue {
    fn from(value: i16) -> Self {
        Self::Sint16(value)
    }
}

impl From<i32> for JitValue {
    fn from(value: i32) -> Self {
        Self::Sint32(value)
    }
}

impl From<i64> for JitValue {
    fn from(value: i64) -> Self {
        Self::Sint64(value)
    }
}

impl From<i128> for JitValue {
    fn from(value: i128) -> Self {
        Self::Sint128(value)
    }
}

impl<T: Into<JitValue> + Clone> From<&[T]> for JitValue {
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().map(|x| x.clone().into()).collect())
    }
}

impl<T: Into<JitValue>> From<Vec<T>> for JitValue {
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(|x| x.into()).collect())
    }
}

impl<T: Into<JitValue>, const N: usize> From<[T; N]> for JitValue {
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(|x| x.into()).collect())
    }
}

impl JitValue {
    pub(crate) fn resolve_type<'a>(
        ty: &'a CoreTypeConcrete,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> &'a CoreTypeConcrete {
        match ty {
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty).unwrap(),
            x => x,
        }
    }

    /// Allocates the value in the given arena so it can be passed to the JIT engine.
    pub(crate) fn to_jit(
        &self,
        arena: &Bump,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_id: &ConcreteTypeId,
    ) -> Result<NonNull<()>, RunnerError> {
        let ty = registry.get_type(type_id)?;

        Ok(unsafe {
            match self {
                Self::Felt252(value) => {
                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();

                    let data = felt252_bigint(value.to_bigint());
                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
                    ptr
                }
                Self::Array(data) => {
                    if let CoreTypeConcrete::Array(info) = Self::resolve_type(ty, registry) {
                        let elem_ty = registry.get_type(&info.ty)?;
                        let elem_layout = elem_ty
                            .layout(registry)
                            .map_err(|e| make_type_builder_error(type_id)(e.into()))?
                            .pad_to_align();

                        let ptr: *mut NonNull<()> =
                            libc::malloc(elem_layout.size() * data.len()).cast();
                        let mut len: u32 = 0;
                        let cap: u32 = data.len().try_into().unwrap();

                        for elem in data {
                            let elem = elem.to_jit(arena, registry, &info.ty)?;

                            std::ptr::copy_nonoverlapping(
                                elem.cast::<u8>().as_ptr(),
                                NonNull::new(
                                    ((NonNull::new_unchecked(ptr).as_ptr() as usize)
                                        + len as usize * elem_layout.size())
                                        as *mut u8,
                                )
                                .unwrap()
                                .cast()
                                .as_ptr(),
                                elem_layout.size(),
                            );

                            len += 1;
                        }

                        let target = arena.alloc_layout(
                            Layout::new::<*mut NonNull<()>>()
                                .extend(Layout::new::<u32>())?
                                .0
                                .extend(Layout::new::<u32>())?
                                .0,
                        );

                        *target.cast::<*mut NonNull<()>>().as_mut() = ptr;

                        let (layout, offset) =
                            Layout::new::<*mut NonNull<()>>().extend(Layout::new::<u32>())?;

                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                            .unwrap()
                            .cast()
                            .as_mut() = len;

                        let (_, offset) = layout.extend(Layout::new::<u32>())?;

                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                            .unwrap()
                            .cast()
                            .as_mut() = cap;
                        target.cast()
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got an array",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Struct {
                    fields: members, ..
                } => {
                    if let CoreTypeConcrete::Struct(info) = Self::resolve_type(ty, registry) {
                        let mut layout: Option<Layout> = None;
                        let mut data = Vec::with_capacity(info.members.len());

                        let mut is_memory_allocated = false;
                        for (member_type_id, member) in info.members.iter().zip(members) {
                            let member_ty = registry.get_type(member_type_id)?;
                            let member_layout = member_ty
                                .layout(registry)
                                .map_err(make_type_builder_error(type_id))?;

                            let (new_layout, offset) = match layout {
                                Some(layout) => layout.extend(member_layout)?,
                                None => (member_layout, 0),
                            };
                            layout = Some(new_layout);

                            let member_ptr = member.to_jit(arena, registry, member_type_id)?;
                            data.push((
                                member_layout,
                                offset,
                                if member_ty.is_memory_allocated(registry) {
                                    is_memory_allocated = true;

                                    // Undo the wrapper pointer added because the member's memory
                                    // allocated flag.
                                    *member_ptr.cast::<NonNull<()>>().as_ref()
                                } else {
                                    member_ptr
                                },
                            ));
                        }

                        let ptr = arena
                            .alloc_layout(layout.unwrap_or(Layout::new::<()>()))
                            .cast();

                        for (layout, offset, member_ptr) in data {
                            std::ptr::copy_nonoverlapping(
                                member_ptr.cast::<u8>().as_ptr(),
                                NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut u8)
                                    .unwrap()
                                    .cast()
                                    .as_ptr(),
                                layout.size(),
                            );
                        }

                        if is_memory_allocated {
                            NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
                                .unwrap()
                                .cast()
                        } else {
                            ptr
                        }
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got a struct",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Enum { tag, value, .. } => {
                    if let CoreTypeConcrete::Enum(info) = Self::resolve_type(ty, registry) {
                        assert!(*tag <= info.variants.len(), "Variant index out of range.");

                        let payload_type_id = &info.variants[*tag];
                        let payload = value.to_jit(arena, registry, payload_type_id)?;

                        let (layout, tag_layout, variant_layouts) =
                            crate::types::r#enum::get_layout_for_variants(registry, &info.variants)
                                .unwrap();
                        let ptr = arena.alloc_layout(layout).cast::<()>();

                        match tag_layout.size() {
                            0 => panic!("An enum without variants cannot be instantiated."),
                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                            _ => unreachable!(),
                        }

                        std::ptr::copy_nonoverlapping(
                            payload.cast::<u8>().as_ptr(),
                            NonNull::new(
                                ((ptr.as_ptr() as usize)
                                    + tag_layout.extend(variant_layouts[*tag]).unwrap().1)
                                    as *mut u8,
                            )
                            .unwrap()
                            .cast()
                            .as_ptr(),
                            variant_layouts[*tag].size(),
                        );

                        NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
                            .unwrap()
                            .cast()
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got an enum value",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Felt252Dict { value: map, .. } => {
                    if let CoreTypeConcrete::Felt252Dict(info) = Self::resolve_type(ty, registry) {
                        let elem_ty = registry.get_type(&info.ty).unwrap();
                        let elem_layout = elem_ty.layout(registry).unwrap().pad_to_align();

                        let mut value_map = HashMap::<[u8; 32], NonNull<std::ffi::c_void>>::new();

                        // next key must be called before next_value

                        for (key, value) in map.iter() {
                            let key = key.to_bytes_le();
                            let value = value.to_jit(arena, registry, &info.ty)?;

                            let value_malloc_ptr =
                                NonNull::new(libc::malloc(elem_layout.size())).unwrap();

                            std::ptr::copy_nonoverlapping(
                                value.cast::<u8>().as_ptr(),
                                value_malloc_ptr.cast().as_ptr(),
                                elem_layout.size(),
                            );

                            value_map.insert(key, value_malloc_ptr);
                        }

                        NonNull::new_unchecked(Box::into_raw(Box::new(value_map))).cast()
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got a felt dict",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Uint8(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
                    *ptr.cast::<u8>().as_mut() = *value;

                    ptr
                }
                Self::Uint16(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u16>()).cast();
                    *ptr.cast::<u16>().as_mut() = *value;

                    ptr
                }
                Self::Uint32(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u32>()).cast();
                    *ptr.cast::<u32>().as_mut() = *value;

                    ptr
                }
                Self::Uint64(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u64>()).cast();
                    *ptr.cast::<u64>().as_mut() = *value;

                    ptr
                }
                Self::Uint128(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
                    *ptr.cast::<u128>().as_mut() = *value;

                    ptr
                }
                Self::Sint8(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i8>()).cast();
                    *ptr.cast::<i8>().as_mut() = *value;

                    ptr
                }
                Self::Sint16(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i16>()).cast();
                    *ptr.cast::<i16>().as_mut() = *value;

                    ptr
                }
                Self::Sint32(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i32>()).cast();
                    *ptr.cast::<i32>().as_mut() = *value;

                    ptr
                }
                Self::Sint64(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i64>()).cast();
                    *ptr.cast::<i64>().as_mut() = *value;

                    ptr
                }
                Self::Sint128(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i128>()).cast();
                    *ptr.cast::<i128>().as_mut() = *value;

                    ptr
                }
                Self::EcPoint(a, b) => {
                    let ptr = arena
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 2).unwrap().0)
                        .cast();

                    let a = felt252_bigint(a.to_bigint());
                    let b = felt252_bigint(b.to_bigint());
                    let data = [a, b];

                    ptr.cast::<[[u32; 8]; 2]>().as_mut().copy_from_slice(&data);

                    ptr
                }
                Self::EcState(a, b, c, d) => {
                    let ptr = arena
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 4).unwrap().0)
                        .cast();

                    let a = felt252_bigint(a.to_bigint());
                    let b = felt252_bigint(b.to_bigint());
                    let c = felt252_bigint(c.to_bigint());
                    let d = felt252_bigint(d.to_bigint());
                    let data = [a, b, c, d];

                    ptr.cast::<[[u32; 8]; 4]>().as_mut().copy_from_slice(&data);

                    ptr
                }
                Self::Secp256K1Point { .. } => todo!(),
                Self::Secp256R1Point { .. } => todo!(),
                Self::Null => {
                    unimplemented!("null is meant as return value for nullable for now")
                }
            }
        })
    }

    /// From the given pointer acquired from the JIT outputs, convert it to a [`Self`]
    pub(crate) fn from_jit(
        ptr: NonNull<()>,
        type_id: &ConcreteTypeId,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> Self {
        let ty = registry.get_type(type_id).unwrap();

        unsafe {
            match ty {
                CoreTypeConcrete::Array(info) => {
                    let elem_ty = registry.get_type(&info.ty).unwrap();

                    let elem_layout = elem_ty.layout(registry).unwrap();
                    let elem_stride = elem_layout.pad_to_align().size();

                    let ptr_layout = Layout::new::<*mut ()>();
                    let len_layout = crate::utils::get_integer_layout(32);

                    let len_value = *NonNull::new(
                        ((ptr.as_ptr() as usize) + ptr_layout.extend(len_layout).unwrap().1)
                            as *mut (),
                    )
                    .unwrap()
                    .cast::<u32>()
                    .as_ref();

                    let data_ptr = *ptr.cast::<NonNull<()>>().as_ref();
                    let mut array_value = Vec::new();

                    for i in 0..(len_value as usize) {
                        let cur_elem_ptr = NonNull::new(
                            ((data_ptr.as_ptr() as usize) + elem_stride * i) as *mut (),
                        )
                        .unwrap();

                        array_value.push(Self::from_jit(cur_elem_ptr, &info.ty, registry));
                    }

                    libc::free(data_ptr.as_ptr().cast());

                    Self::Array(array_value)
                }
                CoreTypeConcrete::Box(info) => {
                    let inner = *ptr.cast::<NonNull<()>>().as_ptr();
                    let value = Self::from_jit(inner, &info.ty, registry);
                    libc::free(inner.as_ptr().cast());
                    value
                }
                CoreTypeConcrete::EcPoint(_) => {
                    let data = ptr.cast::<[[u8; 32]; 2]>().as_ref();

                    Self::EcPoint(Felt::from_bytes_le(&data[0]), Felt::from_bytes_le(&data[1]))
                }
                CoreTypeConcrete::EcState(_) => {
                    let data = ptr.cast::<[[u8; 32]; 4]>().as_ref();

                    Self::EcState(
                        Felt::from_bytes_le(&data[0]),
                        Felt::from_bytes_le(&data[1]),
                        Felt::from_bytes_le(&data[2]),
                        Felt::from_bytes_le(&data[3]),
                    )
                }
                CoreTypeConcrete::Felt252(_) => {
                    let data = ptr.cast::<[u8; 32]>().as_ref();
                    let data = Felt::from_bytes_le(data);
                    Self::Felt252(data)
                }
                CoreTypeConcrete::Uint8(_) => Self::Uint8(*ptr.cast::<u8>().as_ref()),
                CoreTypeConcrete::Uint16(_) => Self::Uint16(*ptr.cast::<u16>().as_ref()),
                CoreTypeConcrete::Uint32(_) => Self::Uint32(*ptr.cast::<u32>().as_ref()),
                CoreTypeConcrete::Uint64(_) => Self::Uint64(*ptr.cast::<u64>().as_ref()),
                CoreTypeConcrete::Uint128(_) => Self::Uint128(*ptr.cast::<u128>().as_ref()),
                CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
                CoreTypeConcrete::Sint8(_) => Self::Sint8(*ptr.cast::<i8>().as_ref()),
                CoreTypeConcrete::Sint16(_) => Self::Sint16(*ptr.cast::<i16>().as_ref()),
                CoreTypeConcrete::Sint32(_) => Self::Sint32(*ptr.cast::<i32>().as_ref()),
                CoreTypeConcrete::Sint64(_) => Self::Sint64(*ptr.cast::<i64>().as_ref()),
                CoreTypeConcrete::Sint128(_) => Self::Sint128(*ptr.cast::<i128>().as_ref()),
                CoreTypeConcrete::NonZero(info) => Self::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::Nullable(info) => {
                    let inner_ptr = *ptr.cast::<*mut ()>().as_ptr();
                    if inner_ptr.is_null() {
                        Self::Null
                    } else {
                        let value = Self::from_jit(
                            NonNull::new_unchecked(inner_ptr).cast(),
                            &info.ty,
                            registry,
                        );
                        libc::free(inner_ptr.cast());
                        value
                    }
                }
                CoreTypeConcrete::Uninitialized(_) => {
                    todo!("implement uninit from_jit or ignore the return value")
                }
                CoreTypeConcrete::Enum(info) => {
                    let tag_layout = crate::utils::get_integer_layout(match info.variants.len() {
                        0 | 1 => 0,
                        num_variants => {
                            (next_multiple_of_usize(num_variants.next_power_of_two(), 8) >> 3)
                                .try_into()
                                .unwrap()
                        }
                    });
                    let tag_value = match info.variants.len() {
                        0 => {
                            // An enum without variants is basically the `!` (never) type in Rust.
                            panic!("An enum without variants is not a valid type.")
                        }
                        1 => 0,
                        _ => match tag_layout.size() {
                            1 => *ptr.cast::<u8>().as_ref() as usize,
                            2 => *ptr.cast::<u16>().as_ref() as usize,
                            4 => *ptr.cast::<u32>().as_ref() as usize,
                            8 => *ptr.cast::<u64>().as_ref() as usize,
                            _ => unreachable!(),
                        },
                    };

                    let payload_ty = registry.get_type(&info.variants[tag_value]).unwrap();
                    let payload_layout = payload_ty.layout(registry).unwrap();

                    let payload_ptr = NonNull::new(
                        ((ptr.as_ptr() as usize) + tag_layout.extend(payload_layout).unwrap().1)
                            as *mut _,
                    )
                    .unwrap();
                    let payload =
                        JitValue::from_jit(payload_ptr, &info.variants[tag_value], registry);

                    JitValue::Enum {
                        tag: tag_value,
                        value: Box::new(payload),
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Struct(info) => {
                    let mut layout: Option<Layout> = None;
                    let mut members = Vec::with_capacity(info.members.len());

                    for member_ty in &info.members {
                        let member = registry.get_type(member_ty).unwrap();
                        let member_layout = member.layout(registry).unwrap();

                        let (new_layout, offset) = match layout {
                            Some(layout) => layout.extend(member_layout).unwrap(),
                            None => (member_layout, 0),
                        };
                        layout = Some(new_layout);

                        members.push(Self::from_jit(
                            NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
                            member_ty,
                            registry,
                        ));
                    }

                    JitValue::Struct {
                        fields: members,
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Felt252Dict(info)
                | CoreTypeConcrete::SquashedFelt252Dict(info) => {
                    let map = Box::from_raw(
                        ptr.cast::<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>()
                            .as_ptr(),
                    );

                    let mut output_map = HashMap::with_capacity(map.len());

                    for (key, val_ptr) in map.iter() {
                        let key = Felt::from_bytes_le(key);
                        output_map.insert(key, Self::from_jit(val_ptr.cast(), &info.ty, registry));
                    }

                    JitValue::Felt252Dict {
                        value: output_map,
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Felt252DictEntry(_) => {
                    unimplemented!("shouldn't be possible to return")
                }
                CoreTypeConcrete::Pedersen(_)
                | CoreTypeConcrete::Poseidon(_)
                | CoreTypeConcrete::Bitwise(_)
                | CoreTypeConcrete::BuiltinCosts(_)
                | CoreTypeConcrete::RangeCheck(_)
                | CoreTypeConcrete::EcOp(_)
                | CoreTypeConcrete::GasBuiltin(_)
                | CoreTypeConcrete::SegmentArena(_) => {
                    unimplemented!("handled before: {:?}", type_id)
                }
                // Does it make sense for programs to return this? Should it be implemented
                CoreTypeConcrete::StarkNet(selector) => match selector {
                    StarkNetTypeConcrete::ClassHash(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_) => {
                        // felt values
                        let data = ptr.cast::<[u8; 32]>().as_ref();
                        let data = Felt::from_bytes_le(data);
                        JitValue::Felt252(data)
                    }
                    StarkNetTypeConcrete::System(_) => {
                        unimplemented!("should be handled before")
                    }
                    StarkNetTypeConcrete::Secp256Point(info) => {
                        let data = ptr.cast::<[[u128; 2]; 2]>().as_ref();

                        let x = (data[0][0], data[0][1]);
                        let y = (data[1][0], data[1][1]);

                        match info {
                            Secp256PointTypeConcrete::K1(_) => JitValue::Secp256K1Point { x, y },
                            Secp256PointTypeConcrete::R1(_) => JitValue::Secp256R1Point { x, y },
                        }
                    }
                },
                CoreTypeConcrete::Span(_) => todo!("implement span from_jit"),
                CoreTypeConcrete::Snapshot(info) => Self::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::Bytes31(_) => todo!("implement bytes31 from_jit"),

                CoreTypeConcrete::Const(_) => todo!(),
                CoreTypeConcrete::BoundedInt(_) => todo!(),
            }
        }
    }

    /// String to felt
    pub fn felt_str(value: &str) -> Self {
        let value = value.parse::<BigInt>().unwrap();
        let value = match value.sign() {
            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
            _ => value.to_biguint().unwrap(),
        };

        Self::Felt252(Felt::from(&value))
    }
}
