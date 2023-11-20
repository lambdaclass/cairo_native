//! # JIT params and return values de/serialization

//! A Rusty interface to provide parameters to JIT calls.

use std::{alloc::Layout, collections::HashMap, ops::Neg, ptr::NonNull};

use bumpalo::Bump;
use cairo_felt::Felt252;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::StarkNetTypeConcrete,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use num_bigint::{BigInt, Sign};

use crate::{
    error::jit_engine::{make_type_builder_error, ErrorImpl, RunnerError},
    types::{felt252::PRIME, TypeBuilder},
    utils::{
        felt252_bigint, get_integer_layout, layout_repeat, next_multiple_of_usize, u32_vec_to_felt,
    },
};

/// A JITValue is a value that can be passed to the JIT engine as an argument or received as a result.
///
/// They map to the cairo/sierra types.
///
/// The debug_name field on some variants is `Some` when receiving a [`JITValue`] as a result.
#[derive(Educe, Debug, Clone)]
#[educe(PartialEq, Eq)]
pub enum JITValue {
    Felt252(Felt252),
    /// all elements need to be same type
    Array(Vec<Self>),
    Struct {
        fields: Vec<Self>,
        #[educe(PartialEq(ignore))]
        #[educe(Eq(ignore))]
        debug_name: Option<String>,
    }, // element types can differ
    Enum {
        tag: usize,
        value: Box<Self>,
        #[educe(PartialEq(ignore))]
        #[educe(Eq(ignore))]
        debug_name: Option<String>,
    },
    Felt252Dict {
        value: HashMap<Felt252, Self>,
        #[educe(PartialEq(ignore))]
        #[educe(Eq(ignore))]
        debug_name: Option<String>,
    },
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
    EcPoint(Felt252, Felt252),
    EcState(Felt252, Felt252, Felt252, Felt252),
}

// Conversions

impl From<Felt252> for JITValue {
    fn from(value: Felt252) -> Self {
        JITValue::Felt252(value)
    }
}

impl From<u8> for JITValue {
    fn from(value: u8) -> Self {
        JITValue::Uint8(value)
    }
}

impl From<u16> for JITValue {
    fn from(value: u16) -> Self {
        JITValue::Uint16(value)
    }
}

impl From<u32> for JITValue {
    fn from(value: u32) -> Self {
        JITValue::Uint32(value)
    }
}

impl From<u64> for JITValue {
    fn from(value: u64) -> Self {
        JITValue::Uint64(value)
    }
}

impl From<u128> for JITValue {
    fn from(value: u128) -> Self {
        JITValue::Uint128(value)
    }
}

impl<T: Into<JITValue> + Clone> From<&[T]> for JITValue {
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().map(|x| x.clone().into()).collect())
    }
}

impl<T: Into<JITValue>> From<Vec<T>> for JITValue {
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(|x| x.into()).collect())
    }
}

impl<T: Into<JITValue>, const N: usize> From<[T; N]> for JITValue {
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(|x| x.into()).collect())
    }
}

impl JITValue {
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
                JITValue::Felt252(value) => {
                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();

                    let data = felt252_bigint(value.to_bigint());
                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
                    ptr
                }
                JITValue::Array(data) => {
                    if let CoreTypeConcrete::Array(info) = Self::resolve_type(ty, registry) {
                        // todo: if its snapshot  cargo r --example starknet
                        let elem_ty = registry.get_type(&info.ty)?;
                        let elem_layout = elem_ty
                            .layout(registry)
                            .map_err(make_type_builder_error(type_id))?
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
                JITValue::Struct {
                    fields: members, ..
                } => {
                    if let CoreTypeConcrete::Struct(info) = Self::resolve_type(ty, registry) {
                        let mut layout: Option<Layout> = None;
                        let mut data = Vec::with_capacity(info.members.len());

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

                            data.push((
                                member_layout,
                                offset,
                                member.to_jit(arena, registry, member_type_id)?,
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

                        ptr
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got a struct",
                            type_id.debug_name
                        )))?
                    }
                }
                JITValue::Enum { tag, value, .. } => {
                    if let CoreTypeConcrete::Enum(info) = Self::resolve_type(ty, registry) {
                        let tag_value = *tag;
                        assert!(tag_value <= info.variants.len());

                        let payload_type_id = &info.variants[tag_value];
                        let payload = value.to_jit(arena, registry, payload_type_id)?;

                        let (layout, tag_layout, variant_layouts) =
                            crate::types::r#enum::get_layout_for_variants(registry, &info.variants)
                                .unwrap();
                        let ptr = arena.alloc_layout(layout).cast();

                        match tag_layout.size() {
                            1 => *ptr.cast::<u8>().as_mut() = tag_value as u8,
                            2 => *ptr.cast::<u16>().as_mut() = tag_value as u16,
                            4 => *ptr.cast::<u32>().as_mut() = tag_value as u32,
                            8 => *ptr.cast::<u64>().as_mut() = tag_value as u64,
                            _ => unreachable!(),
                        }

                        std::ptr::copy_nonoverlapping(
                            payload.cast::<u8>().as_ptr(),
                            NonNull::new(
                                ((ptr.as_ptr() as usize)
                                    + tag_layout.extend(variant_layouts[tag_value]).unwrap().1)
                                    as *mut u8,
                            )
                            .unwrap()
                            .cast()
                            .as_ptr(),
                            variant_layouts[tag_value].size(),
                        );

                        ptr
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got an enum value",
                            type_id.debug_name
                        )))?
                    }
                }
                JITValue::Felt252Dict { value: map, .. } => {
                    if let CoreTypeConcrete::Felt252Dict(info) = Self::resolve_type(ty, registry) {
                        let elem_ty = registry.get_type(&info.ty).unwrap();
                        let elem_layout = elem_ty.layout(registry).unwrap().pad_to_align();

                        let mut value_map = HashMap::<[u8; 32], NonNull<std::ffi::c_void>>::new();

                        // next key must be called before next_value

                        for (key, value) in map.iter() {
                            let key = key.to_le_bytes();
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

                        let target: NonNull<NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>> =
                            arena
                                .alloc_layout(Layout::new::<
                                    NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>,
                                >())
                                .cast();

                        let map_ptr: NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>> = arena
                            .alloc_layout(
                                Layout::new::<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>()
                                    .pad_to_align(),
                            )
                            .cast();

                        std::ptr::write(map_ptr.as_ptr(), value_map);
                        std::ptr::write(target.as_ptr(), map_ptr);

                        target.cast()
                    } else {
                        Err(ErrorImpl::UnexpectedValue(format!(
                            "expected value of type {:?} but got a felt dict",
                            type_id.debug_name
                        )))?
                    }
                }
                JITValue::Uint8(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
                    *ptr.cast::<u8>().as_mut() = *value;

                    ptr
                }
                JITValue::Uint16(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u16>()).cast();
                    *ptr.cast::<u16>().as_mut() = *value;

                    ptr
                }
                JITValue::Uint32(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u32>()).cast();
                    *ptr.cast::<u32>().as_mut() = *value;

                    ptr
                }
                JITValue::Uint64(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u64>()).cast();
                    *ptr.cast::<u64>().as_mut() = *value;

                    ptr
                }
                JITValue::Uint128(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
                    *ptr.cast::<u128>().as_mut() = *value;

                    ptr
                }
                JITValue::EcPoint(a, b) => {
                    let ptr = arena
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 2).unwrap().0)
                        .cast();

                    let a = felt252_bigint(a.to_bigint());
                    let b = felt252_bigint(b.to_bigint());
                    let data = [a, b];

                    ptr.cast::<[[u32; 8]; 2]>().as_mut().copy_from_slice(&data);

                    ptr
                }
                JITValue::EcState(a, b, c, d) => {
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
            }
        })
    }

    /// From the given pointer acquired from the JIT outputs, convert it to a [`Self`]
    pub(crate) fn from_jit(
        ptr: NonNull<()>,
        type_id: &ConcreteTypeId,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> JITValue {
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

                    Self::Array(array_value)
                }
                CoreTypeConcrete::Box(info) => JITValue::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::EcPoint(_) => {
                    let data = ptr.cast::<[[u32; 8]; 2]>().as_ref();

                    JITValue::EcPoint(u32_vec_to_felt(&data[0]), u32_vec_to_felt(&data[1]))
                }
                CoreTypeConcrete::EcState(_) => {
                    let data = ptr.cast::<[[u32; 8]; 4]>().as_ref();

                    JITValue::EcState(
                        u32_vec_to_felt(&data[0]),
                        u32_vec_to_felt(&data[1]),
                        u32_vec_to_felt(&data[2]),
                        u32_vec_to_felt(&data[3]),
                    )
                }
                CoreTypeConcrete::Felt252(_) => {
                    let data = ptr.cast::<[u32; 8]>().as_ref();
                    let data = u32_vec_to_felt(data);
                    JITValue::Felt252(data)
                }
                CoreTypeConcrete::Uint8(_) => JITValue::Uint8(*ptr.cast::<u8>().as_ref()),
                CoreTypeConcrete::Uint16(_) => JITValue::Uint16(*ptr.cast::<u16>().as_ref()),
                CoreTypeConcrete::Uint32(_) => JITValue::Uint32(*ptr.cast::<u32>().as_ref()),
                CoreTypeConcrete::Uint64(_) => JITValue::Uint64(*ptr.cast::<u64>().as_ref()),
                CoreTypeConcrete::Uint128(_) => JITValue::Uint128(*ptr.cast::<u128>().as_ref()),
                CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
                CoreTypeConcrete::Sint8(_) => todo!(),
                CoreTypeConcrete::Sint16(_) => todo!(),
                CoreTypeConcrete::Sint32(_) => todo!(),
                CoreTypeConcrete::Sint64(_) => todo!(),
                CoreTypeConcrete::Sint128(_) => todo!(),
                CoreTypeConcrete::NonZero(info) => JITValue::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::Nullable(_) => todo!(),
                CoreTypeConcrete::Uninitialized(_) => todo!(),
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
                    let payload = Self::from_jit(payload_ptr, &info.variants[tag_value], registry);

                    Self::Enum {
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

                        members.push(JITValue::from_jit(
                            NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
                            member_ty,
                            registry,
                        ));
                    }

                    JITValue::Struct {
                        fields: members,
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Felt252Dict(info)
                | CoreTypeConcrete::SquashedFelt252Dict(info) => {
                    let ptr = ptr.cast::<NonNull<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>>();
                    let ptr = *ptr.as_ptr();
                    let map = Box::from_raw(ptr.as_ptr());

                    let mut output_map = HashMap::with_capacity(map.len());

                    for (key, val_ptr) in map.iter() {
                        let key = Felt252::from_bytes_le(key.as_slice());
                        output_map.insert(key, Self::from_jit(val_ptr.cast(), &info.ty, registry));
                    }

                    Box::leak(map); // we must leak to avoid a double free

                    JITValue::Felt252Dict {
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
                        let data = ptr.cast::<[u32; 8]>().as_ref();
                        let data = u32_vec_to_felt(data);
                        JITValue::Felt252(data)
                    }
                    StarkNetTypeConcrete::System(_) => unimplemented!("should be handled before"),
                    StarkNetTypeConcrete::Secp256Point(_) => todo!(),
                },
                CoreTypeConcrete::Span(_) => todo!(),
                CoreTypeConcrete::Snapshot(info) => Self::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::Bytes31(_) => todo!(),
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

        Self::Felt252(Felt252::from(value))
    }
}

/// The [`ValueBuilder`] trait is implemented any de/serializable value, which is the `TType`
/// generic.
pub trait ValueBuilder {
    /// Return whether the type is considered complex or simple.
    ///
    /// Complex types are always passed by pointer (both as params and return values) and require a
    /// stack allocation. Examples of complex values include structs and enums, but not felts since
    /// LLVM considers them integers.
    fn is_complex(&self) -> bool;
}

impl ValueBuilder for CoreTypeConcrete {
    fn is_complex(&self) -> bool {
        match self {
            CoreTypeConcrete::Array(_) => true,
            CoreTypeConcrete::Bitwise(_) => false,
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => false,
            CoreTypeConcrete::EcPoint(_) => true,
            CoreTypeConcrete::EcState(_) => true,
            CoreTypeConcrete::Felt252(_) => false,
            CoreTypeConcrete::GasBuiltin(_) => false,
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => false,
            CoreTypeConcrete::Uint16(_) => false,
            CoreTypeConcrete::Uint32(_) => false,
            CoreTypeConcrete::Uint64(_) => false,
            CoreTypeConcrete::Uint128(_) => false,
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => false,
            CoreTypeConcrete::RangeCheck(_) => false,
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => true,
            CoreTypeConcrete::Struct(_) => true,
            CoreTypeConcrete::Felt252Dict(_) => false,
            CoreTypeConcrete::Felt252DictEntry(_) => true,
            CoreTypeConcrete::SquashedFelt252Dict(_) => false,
            CoreTypeConcrete::Pedersen(_) => false,
            CoreTypeConcrete::Poseidon(_) => false,
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::ClassHash(_) => false,
                StarkNetTypeConcrete::ContractAddress(_) => false,
                StarkNetTypeConcrete::StorageBaseAddress(_) => false,
                StarkNetTypeConcrete::StorageAddress(_) => false,
                StarkNetTypeConcrete::System(_) => false,
                StarkNetTypeConcrete::Secp256Point(_) => todo!(),
            },
            CoreTypeConcrete::SegmentArena(_) => false,
            CoreTypeConcrete::Snapshot(_) => false,
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }
    }
}
