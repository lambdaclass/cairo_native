////! # JIT params and return values de/serialization
//! # JIT params and return values de/serialization
//

////! A Rusty interface to provide parameters to JIT calls.
//! A Rusty interface to provide parameters to JIT calls.
//

//use crate::{
use crate::{
//    error::Error,
    error::Error,
//    types::{felt252::PRIME, TypeBuilder},
    types::{felt252::PRIME, TypeBuilder},
//    utils::{felt252_bigint, get_integer_layout, layout_repeat, next_multiple_of_usize},
    utils::{felt252_bigint, get_integer_layout, layout_repeat, next_multiple_of_usize},
//};
};
//use bumpalo::Bump;
use bumpalo::Bump;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
//        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
//        utils::Range,
        utils::Range,
//    },
    },
//    ids::ConcreteTypeId,
    ids::ConcreteTypeId,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use educe::Educe;
use educe::Educe;
//use num_bigint::{BigInt, Sign, ToBigInt};
use num_bigint::{BigInt, Sign, ToBigInt};
//use num_traits::Euclid;
use num_traits::Euclid;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::{alloc::Layout, collections::HashMap, ops::Neg, ptr::NonNull};
use std::{alloc::Layout, collections::HashMap, ops::Neg, ptr::NonNull};
//

///// A JitValue is a value that can be passed to the JIT engine as an argument or received as a result.
/// A JitValue is a value that can be passed to the JIT engine as an argument or received as a result.
/////
///
///// They map to the cairo/sierra types.
/// They map to the cairo/sierra types.
/////
///
///// The debug_name field on some variants is `Some` when receiving a [`JitValue`] as a result.
/// The debug_name field on some variants is `Some` when receiving a [`JitValue`] as a result.
/////
///
///// A Boxed value or a non-null Nullable value is returned with it's inner value.
/// A Boxed value or a non-null Nullable value is returned with it's inner value.
//#[derive(Debug, Clone, Educe)]
#[derive(Debug, Clone, Educe)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//#[educe(Eq, PartialEq)]
#[educe(Eq, PartialEq)]
//pub enum JitValue {
pub enum JitValue {
//    Felt252(Felt),
    Felt252(Felt),
//    Bytes31([u8; 31]),
    Bytes31([u8; 31]),
//    /// all elements need to be same type
    /// all elements need to be same type
//    Array(Vec<Self>),
    Array(Vec<Self>),
//    Struct {
    Struct {
//        fields: Vec<Self>,
        fields: Vec<Self>,
//        #[educe(PartialEq(ignore))]
        #[educe(PartialEq(ignore))]
//        debug_name: Option<String>,
        debug_name: Option<String>,
//    }, // element types can differ
    }, // element types can differ
//    Enum {
    Enum {
//        tag: usize,
        tag: usize,
//        value: Box<Self>,
        value: Box<Self>,
//        #[educe(PartialEq(ignore))]
        #[educe(PartialEq(ignore))]
//        debug_name: Option<String>,
        debug_name: Option<String>,
//    },
    },
//    Felt252Dict {
    Felt252Dict {
//        value: HashMap<Felt, Self>,
        value: HashMap<Felt, Self>,
//        #[educe(PartialEq(ignore))]
        #[educe(PartialEq(ignore))]
//        debug_name: Option<String>,
        debug_name: Option<String>,
//    },
    },
//    Uint8(u8),
    Uint8(u8),
//    Uint16(u16),
    Uint16(u16),
//    Uint32(u32),
    Uint32(u32),
//    Uint64(u64),
    Uint64(u64),
//    Uint128(u128),
    Uint128(u128),
//    Sint8(i8),
    Sint8(i8),
//    Sint16(i16),
    Sint16(i16),
//    Sint32(i32),
    Sint32(i32),
//    Sint64(i64),
    Sint64(i64),
//    Sint128(i128),
    Sint128(i128),
//    EcPoint(Felt, Felt),
    EcPoint(Felt, Felt),
//    EcState(Felt, Felt, Felt, Felt),
    EcState(Felt, Felt, Felt, Felt),
//    Secp256K1Point {
    Secp256K1Point {
//        x: (u128, u128),
        x: (u128, u128),
//        y: (u128, u128),
        y: (u128, u128),
//    },
    },
//    Secp256R1Point {
    Secp256R1Point {
//        x: (u128, u128),
        x: (u128, u128),
//        y: (u128, u128),
        y: (u128, u128),
//    },
    },
//    BoundedInt {
    BoundedInt {
//        value: Felt,
        value: Felt,
//        #[cfg_attr(feature = "with-serde", serde(with = "range_serde"))]
        #[cfg_attr(feature = "with-serde", serde(with = "range_serde"))]
//        range: Range,
        range: Range,
//    },
    },
//    /// Used as return value for Nullables that are null.
    /// Used as return value for Nullables that are null.
//    Null,
    Null,
//}
}
//

//// Conversions
// Conversions
//

//impl From<Felt> for JitValue {
impl From<Felt> for JitValue {
//    fn from(value: Felt) -> Self {
    fn from(value: Felt) -> Self {
//        Self::Felt252(value)
        Self::Felt252(value)
//    }
    }
//}
}
//

//impl From<u8> for JitValue {
impl From<u8> for JitValue {
//    fn from(value: u8) -> Self {
    fn from(value: u8) -> Self {
//        Self::Uint8(value)
        Self::Uint8(value)
//    }
    }
//}
}
//

//impl From<u16> for JitValue {
impl From<u16> for JitValue {
//    fn from(value: u16) -> Self {
    fn from(value: u16) -> Self {
//        Self::Uint16(value)
        Self::Uint16(value)
//    }
    }
//}
}
//

//impl From<u32> for JitValue {
impl From<u32> for JitValue {
//    fn from(value: u32) -> Self {
    fn from(value: u32) -> Self {
//        Self::Uint32(value)
        Self::Uint32(value)
//    }
    }
//}
}
//

//impl From<u64> for JitValue {
impl From<u64> for JitValue {
//    fn from(value: u64) -> Self {
    fn from(value: u64) -> Self {
//        Self::Uint64(value)
        Self::Uint64(value)
//    }
    }
//}
}
//

//impl From<u128> for JitValue {
impl From<u128> for JitValue {
//    fn from(value: u128) -> Self {
    fn from(value: u128) -> Self {
//        Self::Uint128(value)
        Self::Uint128(value)
//    }
    }
//}
}
//

//impl From<i8> for JitValue {
impl From<i8> for JitValue {
//    fn from(value: i8) -> Self {
    fn from(value: i8) -> Self {
//        Self::Sint8(value)
        Self::Sint8(value)
//    }
    }
//}
}
//

//impl From<i16> for JitValue {
impl From<i16> for JitValue {
//    fn from(value: i16) -> Self {
    fn from(value: i16) -> Self {
//        Self::Sint16(value)
        Self::Sint16(value)
//    }
    }
//}
}
//

//impl From<i32> for JitValue {
impl From<i32> for JitValue {
//    fn from(value: i32) -> Self {
    fn from(value: i32) -> Self {
//        Self::Sint32(value)
        Self::Sint32(value)
//    }
    }
//}
}
//

//impl From<i64> for JitValue {
impl From<i64> for JitValue {
//    fn from(value: i64) -> Self {
    fn from(value: i64) -> Self {
//        Self::Sint64(value)
        Self::Sint64(value)
//    }
    }
//}
}
//

//impl From<i128> for JitValue {
impl From<i128> for JitValue {
//    fn from(value: i128) -> Self {
    fn from(value: i128) -> Self {
//        Self::Sint128(value)
        Self::Sint128(value)
//    }
    }
//}
}
//

//impl<T: Into<JitValue> + Clone> From<&[T]> for JitValue {
impl<T: Into<JitValue> + Clone> From<&[T]> for JitValue {
//    fn from(value: &[T]) -> Self {
    fn from(value: &[T]) -> Self {
//        Self::Array(value.iter().map(|x| x.clone().into()).collect())
        Self::Array(value.iter().map(|x| x.clone().into()).collect())
//    }
    }
//}
}
//

//impl<T: Into<JitValue>> From<Vec<T>> for JitValue {
impl<T: Into<JitValue>> From<Vec<T>> for JitValue {
//    fn from(value: Vec<T>) -> Self {
    fn from(value: Vec<T>) -> Self {
//        Self::Array(value.into_iter().map(Into::into).collect())
        Self::Array(value.into_iter().map(Into::into).collect())
//    }
    }
//}
}
//

//impl<T: Into<JitValue>, const N: usize> From<[T; N]> for JitValue {
impl<T: Into<JitValue>, const N: usize> From<[T; N]> for JitValue {
//    fn from(value: [T; N]) -> Self {
    fn from(value: [T; N]) -> Self {
//        Self::Array(value.into_iter().map(Into::into).collect())
        Self::Array(value.into_iter().map(Into::into).collect())
//    }
    }
//}
}
//

//impl JitValue {
impl JitValue {
//    pub(crate) fn resolve_type<'a>(
    pub(crate) fn resolve_type<'a>(
//        ty: &'a CoreTypeConcrete,
        ty: &'a CoreTypeConcrete,
//        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
//    ) -> &'a CoreTypeConcrete {
    ) -> &'a CoreTypeConcrete {
//        match ty {
        match ty {
//            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty).unwrap(),
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty).unwrap(),
//            x => x,
            x => x,
//        }
        }
//    }
    }
//

//    /// Allocates the value in the given arena so it can be passed to the JIT engine.
    /// Allocates the value in the given arena so it can be passed to the JIT engine.
//    pub(crate) fn to_jit(
    pub(crate) fn to_jit(
//        &self,
        &self,
//        arena: &Bump,
        arena: &Bump,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        type_id: &ConcreteTypeId,
        type_id: &ConcreteTypeId,
//    ) -> Result<NonNull<()>, Error> {
    ) -> Result<NonNull<()>, Error> {
//        let ty = registry.get_type(type_id)?;
        let ty = registry.get_type(type_id)?;
//

//        Ok(unsafe {
        Ok(unsafe {
//            match self {
            match self {
//                Self::Felt252(value) => {
                Self::Felt252(value) => {
//                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();
                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();
//

//                    let data = felt252_bigint(value.to_bigint());
                    let data = felt252_bigint(value.to_bigint());
//                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
//                    ptr
                    ptr
//                }
                }
//                Self::BoundedInt {
                Self::BoundedInt {
//                    value,
                    value,
//                    range: Range { lower, upper },
                    range: Range { lower, upper },
//                } => {
                } => {
//                    let value = value.to_bigint();
                    let value = value.to_bigint();
//

//                    if lower < upper {
                    if lower < upper {
//                        return Err(Error::Error("BoundedInt range is invalid".to_string()));
                        return Err(Error::Error("BoundedInt range is invalid".to_string()));
//                    }
                    }
//

//                    let prime = &PRIME.to_bigint().unwrap();
                    let prime = &PRIME.to_bigint().unwrap();
//                    let lower = lower.rem_euclid(prime);
                    let lower = lower.rem_euclid(prime);
//                    let upper = upper.rem_euclid(prime);
                    let upper = upper.rem_euclid(prime);
//

//                    if lower <= upper {
                    if lower <= upper {
//                        if !(lower <= value && value < upper) {
                        if !(lower <= value && value < upper) {
//                            return Err(Error::Error(
                            return Err(Error::Error(
//                                "BoundedInt value is out of range".to_string(),
                                "BoundedInt value is out of range".to_string(),
//                            ));
                            ));
//                        }
                        }
//                    } else if !(upper > value && value >= lower) {
                    } else if !(upper > value && value >= lower) {
//                        return Err(Error::Error("BoundedInt value is out of range".to_string()));
                        return Err(Error::Error("BoundedInt value is out of range".to_string()));
//                    }
                    }
//

//                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();
                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();
//                    let data = felt252_bigint(value);
                    let data = felt252_bigint(value);
//                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
                    ptr.cast::<[u32; 8]>().as_mut().copy_from_slice(&data);
//                    ptr
                    ptr
//                }
                }
//

//                Self::Bytes31(_) => todo!(),
                Self::Bytes31(_) => todo!(),
//                Self::Array(data) => {
                Self::Array(data) => {
//                    if let CoreTypeConcrete::Array(info) = Self::resolve_type(ty, registry) {
                    if let CoreTypeConcrete::Array(info) = Self::resolve_type(ty, registry) {
//                        let elem_ty = registry.get_type(&info.ty)?;
                        let elem_ty = registry.get_type(&info.ty)?;
//                        let elem_layout = elem_ty.layout(registry)?.pad_to_align();
                        let elem_layout = elem_ty.layout(registry)?.pad_to_align();
//

//                        let ptr: *mut NonNull<()> =
                        let ptr: *mut NonNull<()> =
//                            libc::malloc(elem_layout.size() * data.len()).cast();
                            libc::malloc(elem_layout.size() * data.len()).cast();
//                        let len: u32 = data.len().try_into().unwrap();
                        let len: u32 = data.len().try_into().unwrap();
//

//                        for elem in data {
                        for elem in data {
//                            let elem = elem.to_jit(arena, registry, &info.ty)?;
                            let elem = elem.to_jit(arena, registry, &info.ty)?;
//

//                            std::ptr::copy_nonoverlapping(
                            std::ptr::copy_nonoverlapping(
//                                elem.cast::<u8>().as_ptr(),
                                elem.cast::<u8>().as_ptr(),
//                                NonNull::new(
                                NonNull::new(
//                                    ((NonNull::new_unchecked(ptr).as_ptr() as usize)
                                    ((NonNull::new_unchecked(ptr).as_ptr() as usize)
//                                        + len as usize * elem_layout.size())
                                        + len as usize * elem_layout.size())
//                                        as *mut u8,
                                        as *mut u8,
//                                )
                                )
//                                .unwrap()
                                .unwrap()
//                                .cast()
                                .cast()
//                                .as_ptr(),
                                .as_ptr(),
//                                elem_layout.size(),
                                elem_layout.size(),
//                            );
                            );
//                        }
                        }
//

//                        let target = arena.alloc_layout(
                        let target = arena.alloc_layout(
//                            Layout::new::<*mut NonNull<()>>()
                            Layout::new::<*mut NonNull<()>>()
//                                .extend(Layout::new::<u32>())?
                                .extend(Layout::new::<u32>())?
//                                .0
                                .0
//                                .extend(Layout::new::<u32>())?
                                .extend(Layout::new::<u32>())?
//                                .0,
                                .0,
//                        );
                        );
//

//                        *target.cast::<*mut NonNull<()>>().as_mut() = ptr;
                        *target.cast::<*mut NonNull<()>>().as_mut() = ptr;
//

//                        let (layout, offset) =
                        let (layout, offset) =
//                            Layout::new::<*mut NonNull<()>>().extend(Layout::new::<u32>())?;
                            Layout::new::<*mut NonNull<()>>().extend(Layout::new::<u32>())?;
//                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
//                            .unwrap()
                            .unwrap()
//                            .cast()
                            .cast()
//                            .as_mut() = 0;
                            .as_mut() = 0;
//

//                        let (layout, offset) = layout.extend(Layout::new::<u32>())?;
                        let (layout, offset) = layout.extend(Layout::new::<u32>())?;
//                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
//                            .unwrap()
                            .unwrap()
//                            .cast()
                            .cast()
//                            .as_mut() = len;
                            .as_mut() = len;
//

//                        let (_, offset) = layout.extend(Layout::new::<u32>())?;
                        let (_, offset) = layout.extend(Layout::new::<u32>())?;
//                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
                        *NonNull::new(((target.as_ptr() as usize) + offset) as *mut u32)
//                            .unwrap()
                            .unwrap()
//                            .cast()
                            .cast()
//                            .as_mut() = len;
                            .as_mut() = len;
//                        target.cast()
                        target.cast()
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "expected value of type {:?} but got an array",
                            "expected value of type {:?} but got an array",
//                            type_id.debug_name
                            type_id.debug_name
//                        )))?
                        )))?
//                    }
                    }
//                }
                }
//                Self::Struct {
                Self::Struct {
//                    fields: members, ..
                    fields: members, ..
//                } => {
                } => {
//                    if let CoreTypeConcrete::Struct(info) = Self::resolve_type(ty, registry) {
                    if let CoreTypeConcrete::Struct(info) = Self::resolve_type(ty, registry) {
//                        let mut layout: Option<Layout> = None;
                        let mut layout: Option<Layout> = None;
//                        let mut data = Vec::with_capacity(info.members.len());
                        let mut data = Vec::with_capacity(info.members.len());
//

//                        let mut is_memory_allocated = false;
                        let mut is_memory_allocated = false;
//                        for (member_type_id, member) in info.members.iter().zip(members) {
                        for (member_type_id, member) in info.members.iter().zip(members) {
//                            let member_ty = registry.get_type(member_type_id)?;
                            let member_ty = registry.get_type(member_type_id)?;
//                            let member_layout = member_ty.layout(registry)?;
                            let member_layout = member_ty.layout(registry)?;
//

//                            let (new_layout, offset) = match layout {
                            let (new_layout, offset) = match layout {
//                                Some(layout) => layout.extend(member_layout)?,
                                Some(layout) => layout.extend(member_layout)?,
//                                None => (member_layout, 0),
                                None => (member_layout, 0),
//                            };
                            };
//                            layout = Some(new_layout);
                            layout = Some(new_layout);
//

//                            let member_ptr = member.to_jit(arena, registry, member_type_id)?;
                            let member_ptr = member.to_jit(arena, registry, member_type_id)?;
//                            data.push((
                            data.push((
//                                member_layout,
                                member_layout,
//                                offset,
                                offset,
//                                if member_ty.is_memory_allocated(registry) {
                                if member_ty.is_memory_allocated(registry) {
//                                    is_memory_allocated = true;
                                    is_memory_allocated = true;
//

//                                    // Undo the wrapper pointer added because the member's memory
                                    // Undo the wrapper pointer added because the member's memory
//                                    // allocated flag.
                                    // allocated flag.
//                                    *member_ptr.cast::<NonNull<()>>().as_ref()
                                    *member_ptr.cast::<NonNull<()>>().as_ref()
//                                } else {
                                } else {
//                                    member_ptr
                                    member_ptr
//                                },
                                },
//                            ));
                            ));
//                        }
                        }
//

//                        let ptr = arena
                        let ptr = arena
//                            .alloc_layout(layout.unwrap_or(Layout::new::<()>()))
                            .alloc_layout(layout.unwrap_or(Layout::new::<()>()))
//                            .cast();
                            .cast();
//

//                        for (layout, offset, member_ptr) in data {
                        for (layout, offset, member_ptr) in data {
//                            std::ptr::copy_nonoverlapping(
                            std::ptr::copy_nonoverlapping(
//                                member_ptr.cast::<u8>().as_ptr(),
                                member_ptr.cast::<u8>().as_ptr(),
//                                NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut u8)
                                NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut u8)
//                                    .unwrap()
                                    .unwrap()
//                                    .cast()
                                    .cast()
//                                    .as_ptr(),
                                    .as_ptr(),
//                                layout.size(),
                                layout.size(),
//                            );
                            );
//                        }
                        }
//

//                        if is_memory_allocated {
                        if is_memory_allocated {
//                            NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
                            NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
//                                .unwrap()
                                .unwrap()
//                                .cast()
                                .cast()
//                        } else {
                        } else {
//                            ptr
                            ptr
//                        }
                        }
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "expected value of type {:?} but got a struct",
                            "expected value of type {:?} but got a struct",
//                            type_id.debug_name
                            type_id.debug_name
//                        )))?
                        )))?
//                    }
                    }
//                }
                }
//                Self::Enum { tag, value, .. } => {
                Self::Enum { tag, value, .. } => {
//                    if let CoreTypeConcrete::Enum(info) = Self::resolve_type(ty, registry) {
                    if let CoreTypeConcrete::Enum(info) = Self::resolve_type(ty, registry) {
//                        assert!(*tag < info.variants.len(), "Variant index out of range.");
                        assert!(*tag < info.variants.len(), "Variant index out of range.");
//

//                        let payload_type_id = &info.variants[*tag];
                        let payload_type_id = &info.variants[*tag];
//                        let payload = value.to_jit(arena, registry, payload_type_id)?;
                        let payload = value.to_jit(arena, registry, payload_type_id)?;
//

//                        let (layout, tag_layout, variant_layouts) =
                        let (layout, tag_layout, variant_layouts) =
//                            crate::types::r#enum::get_layout_for_variants(registry, &info.variants)
                            crate::types::r#enum::get_layout_for_variants(registry, &info.variants)
//                                .unwrap();
                                .unwrap();
//                        let ptr = arena.alloc_layout(layout).cast::<()>();
                        let ptr = arena.alloc_layout(layout).cast::<()>();
//

//                        match tag_layout.size() {
                        match tag_layout.size() {
//                            0 => panic!("An enum without variants cannot be instantiated."),
                            0 => panic!("An enum without variants cannot be instantiated."),
//                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
                            1 => *ptr.cast::<u8>().as_mut() = *tag as u8,
//                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
                            2 => *ptr.cast::<u16>().as_mut() = *tag as u16,
//                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
                            4 => *ptr.cast::<u32>().as_mut() = *tag as u32,
//                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
                            8 => *ptr.cast::<u64>().as_mut() = *tag as u64,
//                            _ => unreachable!(),
                            _ => unreachable!(),
//                        }
                        }
//

//                        std::ptr::copy_nonoverlapping(
                        std::ptr::copy_nonoverlapping(
//                            payload.cast::<u8>().as_ptr(),
                            payload.cast::<u8>().as_ptr(),
//                            NonNull::new(
                            NonNull::new(
//                                ((ptr.as_ptr() as usize)
                                ((ptr.as_ptr() as usize)
//                                    + tag_layout.extend(variant_layouts[*tag]).unwrap().1)
                                    + tag_layout.extend(variant_layouts[*tag]).unwrap().1)
//                                    as *mut u8,
                                    as *mut u8,
//                            )
                            )
//                            .unwrap()
                            .unwrap()
//                            .cast()
                            .cast()
//                            .as_ptr(),
                            .as_ptr(),
//                            variant_layouts[*tag].size(),
                            variant_layouts[*tag].size(),
//                        );
                        );
//

//                        NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
                        NonNull::new(arena.alloc(ptr.as_ptr()) as *mut _)
//                            .unwrap()
                            .unwrap()
//                            .cast()
                            .cast()
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "expected value of type {:?} but got an enum value",
                            "expected value of type {:?} but got an enum value",
//                            type_id.debug_name
                            type_id.debug_name
//                        )))?
                        )))?
//                    }
                    }
//                }
                }
//                Self::Felt252Dict { value: map, .. } => {
                Self::Felt252Dict { value: map, .. } => {
//                    if let CoreTypeConcrete::Felt252Dict(info) = Self::resolve_type(ty, registry) {
                    if let CoreTypeConcrete::Felt252Dict(info) = Self::resolve_type(ty, registry) {
//                        let elem_ty = registry.get_type(&info.ty).unwrap();
                        let elem_ty = registry.get_type(&info.ty).unwrap();
//                        let elem_layout = elem_ty.layout(registry).unwrap().pad_to_align();
                        let elem_layout = elem_ty.layout(registry).unwrap().pad_to_align();
//

//                        let mut value_map = HashMap::<[u8; 32], NonNull<std::ffi::c_void>>::new();
                        let mut value_map = HashMap::<[u8; 32], NonNull<std::ffi::c_void>>::new();
//

//                        // next key must be called before next_value
                        // next key must be called before next_value
//

//                        for (key, value) in map.iter() {
                        for (key, value) in map.iter() {
//                            let key = key.to_bytes_le();
                            let key = key.to_bytes_le();
//                            let value = value.to_jit(arena, registry, &info.ty)?;
                            let value = value.to_jit(arena, registry, &info.ty)?;
//

//                            let value_malloc_ptr =
                            let value_malloc_ptr =
//                                NonNull::new(libc::malloc(elem_layout.size())).unwrap();
                                NonNull::new(libc::malloc(elem_layout.size())).unwrap();
//

//                            std::ptr::copy_nonoverlapping(
                            std::ptr::copy_nonoverlapping(
//                                value.cast::<u8>().as_ptr(),
                                value.cast::<u8>().as_ptr(),
//                                value_malloc_ptr.cast().as_ptr(),
                                value_malloc_ptr.cast().as_ptr(),
//                                elem_layout.size(),
                                elem_layout.size(),
//                            );
                            );
//

//                            value_map.insert(key, value_malloc_ptr);
                            value_map.insert(key, value_malloc_ptr);
//                        }
                        }
//

//                        NonNull::new_unchecked(Box::into_raw(Box::new(value_map))).cast()
                        NonNull::new_unchecked(Box::into_raw(Box::new(value_map))).cast()
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "expected value of type {:?} but got a felt dict",
                            "expected value of type {:?} but got a felt dict",
//                            type_id.debug_name
                            type_id.debug_name
//                        )))?
                        )))?
//                    }
                    }
//                }
                }
//                Self::Uint8(value) => {
                Self::Uint8(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
//                    *ptr.cast::<u8>().as_mut() = *value;
                    *ptr.cast::<u8>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Uint16(value) => {
                Self::Uint16(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<u16>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<u16>()).cast();
//                    *ptr.cast::<u16>().as_mut() = *value;
                    *ptr.cast::<u16>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Uint32(value) => {
                Self::Uint32(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<u32>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<u32>()).cast();
//                    *ptr.cast::<u32>().as_mut() = *value;
                    *ptr.cast::<u32>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Uint64(value) => {
                Self::Uint64(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<u64>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<u64>()).cast();
//                    *ptr.cast::<u64>().as_mut() = *value;
                    *ptr.cast::<u64>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Uint128(value) => {
                Self::Uint128(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
//                    *ptr.cast::<u128>().as_mut() = *value;
                    *ptr.cast::<u128>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Sint8(value) => {
                Self::Sint8(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<i8>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<i8>()).cast();
//                    *ptr.cast::<i8>().as_mut() = *value;
                    *ptr.cast::<i8>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Sint16(value) => {
                Self::Sint16(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<i16>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<i16>()).cast();
//                    *ptr.cast::<i16>().as_mut() = *value;
                    *ptr.cast::<i16>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Sint32(value) => {
                Self::Sint32(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<i32>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<i32>()).cast();
//                    *ptr.cast::<i32>().as_mut() = *value;
                    *ptr.cast::<i32>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Sint64(value) => {
                Self::Sint64(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<i64>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<i64>()).cast();
//                    *ptr.cast::<i64>().as_mut() = *value;
                    *ptr.cast::<i64>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::Sint128(value) => {
                Self::Sint128(value) => {
//                    let ptr = arena.alloc_layout(Layout::new::<i128>()).cast();
                    let ptr = arena.alloc_layout(Layout::new::<i128>()).cast();
//                    *ptr.cast::<i128>().as_mut() = *value;
                    *ptr.cast::<i128>().as_mut() = *value;
//

//                    ptr
                    ptr
//                }
                }
//                Self::EcPoint(a, b) => {
                Self::EcPoint(a, b) => {
//                    let ptr = arena
                    let ptr = arena
//                        .alloc_layout(layout_repeat(&get_integer_layout(252), 2).unwrap().0)
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 2).unwrap().0)
//                        .cast();
                        .cast();
//

//                    let a = felt252_bigint(a.to_bigint());
                    let a = felt252_bigint(a.to_bigint());
//                    let b = felt252_bigint(b.to_bigint());
                    let b = felt252_bigint(b.to_bigint());
//                    let data = [a, b];
                    let data = [a, b];
//

//                    ptr.cast::<[[u32; 8]; 2]>().as_mut().copy_from_slice(&data);
                    ptr.cast::<[[u32; 8]; 2]>().as_mut().copy_from_slice(&data);
//

//                    ptr
                    ptr
//                }
                }
//                Self::EcState(a, b, c, d) => {
                Self::EcState(a, b, c, d) => {
//                    let ptr = arena
                    let ptr = arena
//                        .alloc_layout(layout_repeat(&get_integer_layout(252), 4).unwrap().0)
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 4).unwrap().0)
//                        .cast();
                        .cast();
//

//                    let a = felt252_bigint(a.to_bigint());
                    let a = felt252_bigint(a.to_bigint());
//                    let b = felt252_bigint(b.to_bigint());
                    let b = felt252_bigint(b.to_bigint());
//                    let c = felt252_bigint(c.to_bigint());
                    let c = felt252_bigint(c.to_bigint());
//                    let d = felt252_bigint(d.to_bigint());
                    let d = felt252_bigint(d.to_bigint());
//                    let data = [a, b, c, d];
                    let data = [a, b, c, d];
//

//                    ptr.cast::<[[u32; 8]; 4]>().as_mut().copy_from_slice(&data);
                    ptr.cast::<[[u32; 8]; 4]>().as_mut().copy_from_slice(&data);
//

//                    ptr
                    ptr
//                }
                }
//                Self::Secp256K1Point { .. } => todo!(),
                Self::Secp256K1Point { .. } => todo!(),
//                Self::Secp256R1Point { .. } => todo!(),
                Self::Secp256R1Point { .. } => todo!(),
//                Self::Null => {
                Self::Null => {
//                    unimplemented!("null is meant as return value for nullable for now")
                    unimplemented!("null is meant as return value for nullable for now")
//                }
                }
//            }
            }
//        })
        })
//    }
    }
//

//    /// From the given pointer acquired from the JIT outputs, convert it to a [`Self`]
    /// From the given pointer acquired from the JIT outputs, convert it to a [`Self`]
//    pub(crate) fn from_jit(
    pub(crate) fn from_jit(
//        ptr: NonNull<()>,
        ptr: NonNull<()>,
//        type_id: &ConcreteTypeId,
        type_id: &ConcreteTypeId,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    ) -> Self {
    ) -> Self {
//        let ty = registry.get_type(type_id).unwrap();
        let ty = registry.get_type(type_id).unwrap();
//

//        unsafe {
        unsafe {
//            match ty {
            match ty {
//                CoreTypeConcrete::Array(info) => {
                CoreTypeConcrete::Array(info) => {
//                    let elem_ty = registry.get_type(&info.ty).unwrap();
                    let elem_ty = registry.get_type(&info.ty).unwrap();
//

//                    let elem_layout = elem_ty.layout(registry).unwrap();
                    let elem_layout = elem_ty.layout(registry).unwrap();
//                    let elem_stride = elem_layout.pad_to_align().size();
                    let elem_stride = elem_layout.pad_to_align().size();
//

//                    let ptr_layout = Layout::new::<*mut ()>();
                    let ptr_layout = Layout::new::<*mut ()>();
//                    let len_layout = crate::utils::get_integer_layout(32);
                    let len_layout = crate::utils::get_integer_layout(32);
//

//                    let (ptr_layout, offset) = ptr_layout.extend(len_layout).unwrap();
                    let (ptr_layout, offset) = ptr_layout.extend(len_layout).unwrap();
//                    let offset_value = *NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ())
                    let offset_value = *NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ())
//                        .unwrap()
                        .unwrap()
//                        .cast::<u32>()
                        .cast::<u32>()
//                        .as_ref();
                        .as_ref();
//                    let (_, offset) = ptr_layout.extend(len_layout).unwrap();
                    let (_, offset) = ptr_layout.extend(len_layout).unwrap();
//                    let length_value = *NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ())
                    let length_value = *NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ())
//                        .unwrap()
                        .unwrap()
//                        .cast::<u32>()
                        .cast::<u32>()
//                        .as_ref();
                        .as_ref();
//

//                    // this pointer can be null if the array has a size of 0.
                    // this pointer can be null if the array has a size of 0.
//                    let init_data_ptr = *ptr.cast::<*mut ()>().as_ref();
                    let init_data_ptr = *ptr.cast::<*mut ()>().as_ref();
//                    let data_ptr = init_data_ptr.byte_add(elem_stride * offset_value as usize);
                    let data_ptr = init_data_ptr.byte_add(elem_stride * offset_value as usize);
//

//                    assert!(length_value >= offset_value);
                    assert!(length_value >= offset_value);
//                    let num_elems = (length_value - offset_value) as usize;
                    let num_elems = (length_value - offset_value) as usize;
//                    let mut array_value = Vec::with_capacity(num_elems);
                    let mut array_value = Vec::with_capacity(num_elems);
//

//                    for i in 0..num_elems {
                    for i in 0..num_elems {
//                        // safe to create a NonNull because if the array has elements, the init_data_ptr can't be null.
                        // safe to create a NonNull because if the array has elements, the init_data_ptr can't be null.
//                        let cur_elem_ptr =
                        let cur_elem_ptr =
//                            NonNull::new(((data_ptr as usize) + elem_stride * i) as *mut ())
                            NonNull::new(((data_ptr as usize) + elem_stride * i) as *mut ())
//                                .unwrap();
                                .unwrap();
//

//                        array_value.push(Self::from_jit(cur_elem_ptr, &info.ty, registry));
                        array_value.push(Self::from_jit(cur_elem_ptr, &info.ty, registry));
//                    }
                    }
//

//                    if !init_data_ptr.is_null() {
                    if !init_data_ptr.is_null() {
//                        libc::free(init_data_ptr.cast());
                        libc::free(init_data_ptr.cast());
//                    }
                    }
//

//                    Self::Array(array_value)
                    Self::Array(array_value)
//                }
                }
//                CoreTypeConcrete::Box(info) => {
                CoreTypeConcrete::Box(info) => {
//                    let inner = *ptr.cast::<NonNull<()>>().as_ptr();
                    let inner = *ptr.cast::<NonNull<()>>().as_ptr();
//                    let value = Self::from_jit(inner, &info.ty, registry);
                    let value = Self::from_jit(inner, &info.ty, registry);
//                    libc::free(inner.as_ptr().cast());
                    libc::free(inner.as_ptr().cast());
//                    value
                    value
//                }
                }
//                CoreTypeConcrete::EcPoint(_) => {
                CoreTypeConcrete::EcPoint(_) => {
//                    let data = ptr.cast::<[[u8; 32]; 2]>().as_ref();
                    let data = ptr.cast::<[[u8; 32]; 2]>().as_ref();
//

//                    Self::EcPoint(Felt::from_bytes_le(&data[0]), Felt::from_bytes_le(&data[1]))
                    Self::EcPoint(Felt::from_bytes_le(&data[0]), Felt::from_bytes_le(&data[1]))
//                }
                }
//                CoreTypeConcrete::EcState(_) => {
                CoreTypeConcrete::EcState(_) => {
//                    let data = ptr.cast::<[[u8; 32]; 4]>().as_ref();
                    let data = ptr.cast::<[[u8; 32]; 4]>().as_ref();
//

//                    Self::EcState(
                    Self::EcState(
//                        Felt::from_bytes_le(&data[0]),
                        Felt::from_bytes_le(&data[0]),
//                        Felt::from_bytes_le(&data[1]),
                        Felt::from_bytes_le(&data[1]),
//                        Felt::from_bytes_le(&data[2]),
                        Felt::from_bytes_le(&data[2]),
//                        Felt::from_bytes_le(&data[3]),
                        Felt::from_bytes_le(&data[3]),
//                    )
                    )
//                }
                }
//                CoreTypeConcrete::Felt252(_) => {
                CoreTypeConcrete::Felt252(_) => {
//                    let data = ptr.cast::<[u8; 32]>().as_ref();
                    let data = ptr.cast::<[u8; 32]>().as_ref();
//                    let data = Felt::from_bytes_le_slice(data);
                    let data = Felt::from_bytes_le_slice(data);
//                    Self::Felt252(data)
                    Self::Felt252(data)
//                }
                }
//                CoreTypeConcrete::Uint8(_) => Self::Uint8(*ptr.cast::<u8>().as_ref()),
                CoreTypeConcrete::Uint8(_) => Self::Uint8(*ptr.cast::<u8>().as_ref()),
//                CoreTypeConcrete::Uint16(_) => Self::Uint16(*ptr.cast::<u16>().as_ref()),
                CoreTypeConcrete::Uint16(_) => Self::Uint16(*ptr.cast::<u16>().as_ref()),
//                CoreTypeConcrete::Uint32(_) => Self::Uint32(*ptr.cast::<u32>().as_ref()),
                CoreTypeConcrete::Uint32(_) => Self::Uint32(*ptr.cast::<u32>().as_ref()),
//                CoreTypeConcrete::Uint64(_) => Self::Uint64(*ptr.cast::<u64>().as_ref()),
                CoreTypeConcrete::Uint64(_) => Self::Uint64(*ptr.cast::<u64>().as_ref()),
//                CoreTypeConcrete::Uint128(_) => Self::Uint128(*ptr.cast::<u128>().as_ref()),
                CoreTypeConcrete::Uint128(_) => Self::Uint128(*ptr.cast::<u128>().as_ref()),
//                CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
                CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
//                CoreTypeConcrete::Sint8(_) => Self::Sint8(*ptr.cast::<i8>().as_ref()),
                CoreTypeConcrete::Sint8(_) => Self::Sint8(*ptr.cast::<i8>().as_ref()),
//                CoreTypeConcrete::Sint16(_) => Self::Sint16(*ptr.cast::<i16>().as_ref()),
                CoreTypeConcrete::Sint16(_) => Self::Sint16(*ptr.cast::<i16>().as_ref()),
//                CoreTypeConcrete::Sint32(_) => Self::Sint32(*ptr.cast::<i32>().as_ref()),
                CoreTypeConcrete::Sint32(_) => Self::Sint32(*ptr.cast::<i32>().as_ref()),
//                CoreTypeConcrete::Sint64(_) => Self::Sint64(*ptr.cast::<i64>().as_ref()),
                CoreTypeConcrete::Sint64(_) => Self::Sint64(*ptr.cast::<i64>().as_ref()),
//                CoreTypeConcrete::Sint128(_) => Self::Sint128(*ptr.cast::<i128>().as_ref()),
                CoreTypeConcrete::Sint128(_) => Self::Sint128(*ptr.cast::<i128>().as_ref()),
//                CoreTypeConcrete::NonZero(info) => Self::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::NonZero(info) => Self::from_jit(ptr, &info.ty, registry),
//                CoreTypeConcrete::Nullable(info) => {
                CoreTypeConcrete::Nullable(info) => {
//                    let inner_ptr = *ptr.cast::<*mut ()>().as_ptr();
                    let inner_ptr = *ptr.cast::<*mut ()>().as_ptr();
//                    if inner_ptr.is_null() {
                    if inner_ptr.is_null() {
//                        Self::Null
                        Self::Null
//                    } else {
                    } else {
//                        let value = Self::from_jit(
                        let value = Self::from_jit(
//                            NonNull::new_unchecked(inner_ptr).cast(),
                            NonNull::new_unchecked(inner_ptr).cast(),
//                            &info.ty,
                            &info.ty,
//                            registry,
                            registry,
//                        );
                        );
//                        libc::free(inner_ptr.cast());
                        libc::free(inner_ptr.cast());
//                        value
                        value
//                    }
                    }
//                }
                }
//                CoreTypeConcrete::Uninitialized(_) => {
                CoreTypeConcrete::Uninitialized(_) => {
//                    todo!("implement uninit from_jit or ignore the return value")
                    todo!("implement uninit from_jit or ignore the return value")
//                }
                }
//                CoreTypeConcrete::Enum(info) => {
                CoreTypeConcrete::Enum(info) => {
//                    let tag_layout = crate::utils::get_integer_layout(match info.variants.len() {
                    let tag_layout = crate::utils::get_integer_layout(match info.variants.len() {
//                        0 | 1 => 0,
                        0 | 1 => 0,
//                        num_variants => {
                        num_variants => {
//                            (next_multiple_of_usize(num_variants.next_power_of_two(), 8) >> 3)
                            (next_multiple_of_usize(num_variants.next_power_of_two(), 8) >> 3)
//                                .try_into()
                                .try_into()
//                                .unwrap()
                                .unwrap()
//                        }
                        }
//                    });
                    });
//                    let tag_value = match info.variants.len() {
                    let tag_value = match info.variants.len() {
//                        0 => {
                        0 => {
//                            // An enum without variants is basically the `!` (never) type in Rust.
                            // An enum without variants is basically the `!` (never) type in Rust.
//                            panic!("An enum without variants is not a valid type.")
                            panic!("An enum without variants is not a valid type.")
//                        }
                        }
//                        1 => 0,
                        1 => 0,
//                        _ => match tag_layout.size() {
                        _ => match tag_layout.size() {
//                            1 => *ptr.cast::<u8>().as_ref() as usize,
                            1 => *ptr.cast::<u8>().as_ref() as usize,
//                            2 => *ptr.cast::<u16>().as_ref() as usize,
                            2 => *ptr.cast::<u16>().as_ref() as usize,
//                            4 => *ptr.cast::<u32>().as_ref() as usize,
                            4 => *ptr.cast::<u32>().as_ref() as usize,
//                            8 => *ptr.cast::<u64>().as_ref() as usize,
                            8 => *ptr.cast::<u64>().as_ref() as usize,
//                            _ => unreachable!(),
                            _ => unreachable!(),
//                        },
                        },
//                    };
                    };
//

//                    let payload_ty = registry.get_type(&info.variants[tag_value]).unwrap();
                    let payload_ty = registry.get_type(&info.variants[tag_value]).unwrap();
//                    let payload_layout = payload_ty.layout(registry).unwrap();
                    let payload_layout = payload_ty.layout(registry).unwrap();
//

//                    let payload_ptr = NonNull::new(
                    let payload_ptr = NonNull::new(
//                        ((ptr.as_ptr() as usize) + tag_layout.extend(payload_layout).unwrap().1)
                        ((ptr.as_ptr() as usize) + tag_layout.extend(payload_layout).unwrap().1)
//                            as *mut _,
                            as *mut _,
//                    )
                    )
//                    .unwrap();
                    .unwrap();
//                    let payload =
                    let payload =
//                        JitValue::from_jit(payload_ptr, &info.variants[tag_value], registry);
                        JitValue::from_jit(payload_ptr, &info.variants[tag_value], registry);
//

//                    JitValue::Enum {
                    JitValue::Enum {
//                        tag: tag_value,
                        tag: tag_value,
//                        value: Box::new(payload),
                        value: Box::new(payload),
//                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
//                    }
                    }
//                }
                }
//                CoreTypeConcrete::Struct(info) => {
                CoreTypeConcrete::Struct(info) => {
//                    let mut layout: Option<Layout> = None;
                    let mut layout: Option<Layout> = None;
//                    let mut members = Vec::with_capacity(info.members.len());
                    let mut members = Vec::with_capacity(info.members.len());
//

//                    for member_ty in &info.members {
                    for member_ty in &info.members {
//                        let member = registry.get_type(member_ty).unwrap();
                        let member = registry.get_type(member_ty).unwrap();
//                        let member_layout = member.layout(registry).unwrap();
                        let member_layout = member.layout(registry).unwrap();
//

//                        let (new_layout, offset) = match layout {
                        let (new_layout, offset) = match layout {
//                            Some(layout) => layout.extend(member_layout).unwrap(),
                            Some(layout) => layout.extend(member_layout).unwrap(),
//                            None => (member_layout, 0),
                            None => (member_layout, 0),
//                        };
                        };
//                        layout = Some(new_layout);
                        layout = Some(new_layout);
//

//                        members.push(Self::from_jit(
                        members.push(Self::from_jit(
//                            NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
                            NonNull::new(((ptr.as_ptr() as usize) + offset) as *mut ()).unwrap(),
//                            member_ty,
                            member_ty,
//                            registry,
                            registry,
//                        ));
                        ));
//                    }
                    }
//

//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: members,
                        fields: members,
//                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
//                    }
                    }
//                }
                }
//                CoreTypeConcrete::Felt252Dict(info)
                CoreTypeConcrete::Felt252Dict(info)
//                | CoreTypeConcrete::SquashedFelt252Dict(info) => {
                | CoreTypeConcrete::SquashedFelt252Dict(info) => {
//                    let map = Box::from_raw(
                    let map = Box::from_raw(
//                        ptr.cast::<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>()
                        ptr.cast::<HashMap<[u8; 32], NonNull<std::ffi::c_void>>>()
//                            .as_ptr(),
                            .as_ptr(),
//                    );
                    );
//

//                    let mut output_map = HashMap::with_capacity(map.len());
                    let mut output_map = HashMap::with_capacity(map.len());
//

//                    for (key, val_ptr) in map.iter() {
                    for (key, val_ptr) in map.iter() {
//                        let key = Felt::from_bytes_le(key);
                        let key = Felt::from_bytes_le(key);
//                        output_map.insert(key, Self::from_jit(val_ptr.cast(), &info.ty, registry));
                        output_map.insert(key, Self::from_jit(val_ptr.cast(), &info.ty, registry));
//                    }
                    }
//

//                    JitValue::Felt252Dict {
                    JitValue::Felt252Dict {
//                        value: output_map,
                        value: output_map,
//                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
//                    }
                    }
//                }
                }
//                CoreTypeConcrete::Felt252DictEntry(_) => {
                CoreTypeConcrete::Felt252DictEntry(_) => {
//                    unimplemented!("shouldn't be possible to return")
                    unimplemented!("shouldn't be possible to return")
//                }
                }
//                CoreTypeConcrete::Pedersen(_)
                CoreTypeConcrete::Pedersen(_)
//                | CoreTypeConcrete::Poseidon(_)
                | CoreTypeConcrete::Poseidon(_)
//                | CoreTypeConcrete::Bitwise(_)
                | CoreTypeConcrete::Bitwise(_)
//                | CoreTypeConcrete::BuiltinCosts(_)
                | CoreTypeConcrete::BuiltinCosts(_)
//                | CoreTypeConcrete::RangeCheck(_)
                | CoreTypeConcrete::RangeCheck(_)
//                | CoreTypeConcrete::EcOp(_)
                | CoreTypeConcrete::EcOp(_)
//                | CoreTypeConcrete::GasBuiltin(_)
                | CoreTypeConcrete::GasBuiltin(_)
//                | CoreTypeConcrete::SegmentArena(_) => {
                | CoreTypeConcrete::SegmentArena(_) => {
//                    unimplemented!("handled before: {:?}", type_id)
                    unimplemented!("handled before: {:?}", type_id)
//                }
                }
//                // Does it make sense for programs to return this? Should it be implemented
                // Does it make sense for programs to return this? Should it be implemented
//                CoreTypeConcrete::StarkNet(selector) => match selector {
                CoreTypeConcrete::StarkNet(selector) => match selector {
//                    StarkNetTypeConcrete::ClassHash(_)
                    StarkNetTypeConcrete::ClassHash(_)
//                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
//                    | StarkNetTypeConcrete::StorageBaseAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_)
//                    | StarkNetTypeConcrete::StorageAddress(_) => {
                    | StarkNetTypeConcrete::StorageAddress(_) => {
//                        // felt values
                        // felt values
//                        let data = ptr.cast::<[u8; 32]>().as_ref();
                        let data = ptr.cast::<[u8; 32]>().as_ref();
//                        let data = Felt::from_bytes_le(data);
                        let data = Felt::from_bytes_le(data);
//                        JitValue::Felt252(data)
                        JitValue::Felt252(data)
//                    }
                    }
//                    StarkNetTypeConcrete::System(_) => {
                    StarkNetTypeConcrete::System(_) => {
//                        unimplemented!("should be handled before")
                        unimplemented!("should be handled before")
//                    }
                    }
//                    StarkNetTypeConcrete::Secp256Point(info) => {
                    StarkNetTypeConcrete::Secp256Point(info) => {
//                        let data = ptr.cast::<[[u128; 2]; 2]>().as_ref();
                        let data = ptr.cast::<[[u128; 2]; 2]>().as_ref();
//

//                        let x = (data[0][0], data[0][1]);
                        let x = (data[0][0], data[0][1]);
//                        let y = (data[1][0], data[1][1]);
                        let y = (data[1][0], data[1][1]);
//

//                        match info {
                        match info {
//                            Secp256PointTypeConcrete::K1(_) => JitValue::Secp256K1Point { x, y },
                            Secp256PointTypeConcrete::K1(_) => JitValue::Secp256K1Point { x, y },
//                            Secp256PointTypeConcrete::R1(_) => JitValue::Secp256R1Point { x, y },
                            Secp256PointTypeConcrete::R1(_) => JitValue::Secp256R1Point { x, y },
//                        }
                        }
//                    }
                    }
//                },
                },
//                CoreTypeConcrete::Span(_) => todo!("implement span from_jit"),
                CoreTypeConcrete::Span(_) => todo!("implement span from_jit"),
//                CoreTypeConcrete::Snapshot(info) => Self::from_jit(ptr, &info.ty, registry),
                CoreTypeConcrete::Snapshot(info) => Self::from_jit(ptr, &info.ty, registry),
//                CoreTypeConcrete::Bytes31(_) => {
                CoreTypeConcrete::Bytes31(_) => {
//                    let data = *ptr.cast::<[u8; 31]>().as_ref();
                    let data = *ptr.cast::<[u8; 31]>().as_ref();
//                    Self::Bytes31(data)
                    Self::Bytes31(data)
//                }
                }
//

//                CoreTypeConcrete::Const(_) => todo!(),
                CoreTypeConcrete::Const(_) => todo!(),
//                CoreTypeConcrete::BoundedInt(info) => {
                CoreTypeConcrete::BoundedInt(info) => {
//                    let data = ptr.cast::<[u8; 32]>().as_ref();
                    let data = ptr.cast::<[u8; 32]>().as_ref();
//                    let data = Felt::from_bytes_le(data);
                    let data = Felt::from_bytes_le(data);
//                    Self::BoundedInt {
                    Self::BoundedInt {
//                        value: data,
                        value: data,
//                        range: info.range.clone(),
                        range: info.range.clone(),
//                    }
                    }
//                }
                }
//                CoreTypeConcrete::Coupon(_) => todo!(),
                CoreTypeConcrete::Coupon(_) => todo!(),
//            }
            }
//        }
        }
//    }
    }
//

//    /// String to felt
    /// String to felt
//    pub fn felt_str(value: &str) -> Self {
    pub fn felt_str(value: &str) -> Self {
//        let value = value.parse::<BigInt>().unwrap();
        let value = value.parse::<BigInt>().unwrap();
//        let value = match value.sign() {
        let value = match value.sign() {
//            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
//            _ => value.to_biguint().unwrap(),
            _ => value.to_biguint().unwrap(),
//        };
        };
//

//        Self::Felt252(Felt::from(&value))
        Self::Felt252(Felt::from(&value))
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::*;
    use super::*;
//    use bumpalo::Bump;
    use bumpalo::Bump;
//    use cairo_lang_sierra::extensions::types::{InfoAndTypeConcreteType, TypeInfo};
    use cairo_lang_sierra::extensions::types::{InfoAndTypeConcreteType, TypeInfo};
//    use cairo_lang_sierra::program::ConcreteTypeLongId;
    use cairo_lang_sierra::program::ConcreteTypeLongId;
//    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::Program;
//    use cairo_lang_sierra::program::TypeDeclaration;
    use cairo_lang_sierra::program::TypeDeclaration;
//    use cairo_lang_sierra::ProgramParser;
    use cairo_lang_sierra::ProgramParser;
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_felt() {
    fn test_jit_value_conversion_felt() {
//        let felt_value: Felt = 42.into();
        let felt_value: Felt = 42.into();
//        let jit_value: JitValue = felt_value.into();
        let jit_value: JitValue = felt_value.into();
//        assert_eq!(jit_value, JitValue::Felt252(Felt::from(42)));
        assert_eq!(jit_value, JitValue::Felt252(Felt::from(42)));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_u8() {
    fn test_jit_value_conversion_u8() {
//        let u8_value: u8 = 10;
        let u8_value: u8 = 10;
//        let jit_value: JitValue = u8_value.into();
        let jit_value: JitValue = u8_value.into();
//        assert_eq!(jit_value, JitValue::Uint8(10));
        assert_eq!(jit_value, JitValue::Uint8(10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_u16() {
    fn test_jit_value_conversion_u16() {
//        let u8_value: u16 = 10;
        let u8_value: u16 = 10;
//        let jit_value: JitValue = u8_value.into();
        let jit_value: JitValue = u8_value.into();
//        assert_eq!(jit_value, JitValue::Uint16(10));
        assert_eq!(jit_value, JitValue::Uint16(10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_u32() {
    fn test_jit_value_conversion_u32() {
//        let u32_value: u32 = 10;
        let u32_value: u32 = 10;
//        let jit_value: JitValue = u32_value.into();
        let jit_value: JitValue = u32_value.into();
//        assert_eq!(jit_value, JitValue::Uint32(10));
        assert_eq!(jit_value, JitValue::Uint32(10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_u64() {
    fn test_jit_value_conversion_u64() {
//        let u64_value: u64 = 10;
        let u64_value: u64 = 10;
//        let jit_value: JitValue = u64_value.into();
        let jit_value: JitValue = u64_value.into();
//        assert_eq!(jit_value, JitValue::Uint64(10));
        assert_eq!(jit_value, JitValue::Uint64(10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_u128() {
    fn test_jit_value_conversion_u128() {
//        let u128_value: u128 = 10;
        let u128_value: u128 = 10;
//        let jit_value: JitValue = u128_value.into();
        let jit_value: JitValue = u128_value.into();
//        assert_eq!(jit_value, JitValue::Uint128(10));
        assert_eq!(jit_value, JitValue::Uint128(10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_i8() {
    fn test_jit_value_conversion_i8() {
//        let i8_value: i8 = -10;
        let i8_value: i8 = -10;
//        let jit_value: JitValue = i8_value.into();
        let jit_value: JitValue = i8_value.into();
//        assert_eq!(jit_value, JitValue::Sint8(-10));
        assert_eq!(jit_value, JitValue::Sint8(-10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_i16() {
    fn test_jit_value_conversion_i16() {
//        let i16_value: i16 = -10;
        let i16_value: i16 = -10;
//        let jit_value: JitValue = i16_value.into();
        let jit_value: JitValue = i16_value.into();
//        assert_eq!(jit_value, JitValue::Sint16(-10));
        assert_eq!(jit_value, JitValue::Sint16(-10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_i32() {
    fn test_jit_value_conversion_i32() {
//        let i32_value: i32 = -10;
        let i32_value: i32 = -10;
//        let jit_value: JitValue = i32_value.into();
        let jit_value: JitValue = i32_value.into();
//        assert_eq!(jit_value, JitValue::Sint32(-10));
        assert_eq!(jit_value, JitValue::Sint32(-10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_i64() {
    fn test_jit_value_conversion_i64() {
//        let i64_value: i64 = -10;
        let i64_value: i64 = -10;
//        let jit_value: JitValue = i64_value.into();
        let jit_value: JitValue = i64_value.into();
//        assert_eq!(jit_value, JitValue::Sint64(-10));
        assert_eq!(jit_value, JitValue::Sint64(-10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_i128() {
    fn test_jit_value_conversion_i128() {
//        let i128_value: i128 = -10;
        let i128_value: i128 = -10;
//        let jit_value: JitValue = i128_value.into();
        let jit_value: JitValue = i128_value.into();
//        assert_eq!(jit_value, JitValue::Sint128(-10));
        assert_eq!(jit_value, JitValue::Sint128(-10));
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_array_from_slice() {
    fn test_jit_value_conversion_array_from_slice() {
//        let array_slice: &[u8] = &[1, 2, 3];
        let array_slice: &[u8] = &[1, 2, 3];
//        let jit_value: JitValue = array_slice.into();
        let jit_value: JitValue = array_slice.into();
//        assert_eq!(
        assert_eq!(
//            jit_value,
            jit_value,
//            JitValue::Array(vec![
            JitValue::Array(vec![
//                JitValue::Uint8(1),
                JitValue::Uint8(1),
//                JitValue::Uint8(2),
                JitValue::Uint8(2),
//                JitValue::Uint8(3)
                JitValue::Uint8(3)
//            ])
            ])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_array_from_vec() {
    fn test_jit_value_conversion_array_from_vec() {
//        let array_vec: Vec<u8> = vec![1, 2, 3];
        let array_vec: Vec<u8> = vec![1, 2, 3];
//        let jit_value: JitValue = array_vec.into();
        let jit_value: JitValue = array_vec.into();
//        assert_eq!(
        assert_eq!(
//            jit_value,
            jit_value,
//            JitValue::Array(vec![
            JitValue::Array(vec![
//                JitValue::Uint8(1),
                JitValue::Uint8(1),
//                JitValue::Uint8(2),
                JitValue::Uint8(2),
//                JitValue::Uint8(3)
                JitValue::Uint8(3)
//            ])
            ])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_jit_value_conversion_array_from_fixed_size_array() {
    fn test_jit_value_conversion_array_from_fixed_size_array() {
//        let array_fixed: [u8; 3] = [1, 2, 3];
        let array_fixed: [u8; 3] = [1, 2, 3];
//        let jit_value: JitValue = array_fixed.into();
        let jit_value: JitValue = array_fixed.into();
//        assert_eq!(
        assert_eq!(
//            jit_value,
            jit_value,
//            JitValue::Array(vec![
            JitValue::Array(vec![
//                JitValue::Uint8(1),
                JitValue::Uint8(1),
//                JitValue::Uint8(2),
                JitValue::Uint8(2),
//                JitValue::Uint8(3)
                JitValue::Uint8(3)
//            ])
            ])
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_resolve_type_snapshot() {
    fn test_resolve_type_snapshot() {
//        let ty = CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType {
        let ty = CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType {
//            info: TypeInfo {
            info: TypeInfo {
//                long_id: ConcreteTypeLongId {
                long_id: ConcreteTypeLongId {
//                    generic_id: "generic_type_id".into(),
                    generic_id: "generic_type_id".into(),
//                    generic_args: vec![],
                    generic_args: vec![],
//                },
                },
//                storable: false,
                storable: false,
//                droppable: false,
                droppable: false,
//                duplicatable: false,
                duplicatable: false,
//                zero_sized: false,
                zero_sized: false,
//            },
            },
//            ty: "test_id".into(),
            ty: "test_id".into(),
//        });
        });
//

//        let program = Program {
        let program = Program {
//            type_declarations: vec![TypeDeclaration {
            type_declarations: vec![TypeDeclaration {
//                id: "test_id".into(),
                id: "test_id".into(),
//                long_id: ConcreteTypeLongId {
                long_id: ConcreteTypeLongId {
//                    generic_id: "u128".into(),
                    generic_id: "u128".into(),
//                    generic_args: vec![],
                    generic_args: vec![],
//                },
                },
//                declared_type_info: None,
                declared_type_info: None,
//            }],
            }],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![],
            funcs: vec![],
//        };
        };
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            JitValue::resolve_type(&ty, &registry).integer_width(),
            JitValue::resolve_type(&ty, &registry).integer_width(),
//            Some(128)
            Some(128)
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_felt252() {
    fn test_to_jit_felt252() {
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse("type felt252 = felt252;")
            .parse("type felt252 = felt252;")
//            .unwrap();
            .unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Felt252(Felt::from(42))
                *JitValue::Felt252(Felt::from(42))
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<[u32; 8]>()
                    .cast::<[u32; 8]>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            [42, 0, 0, 0, 0, 0, 0, 0]
            [42, 0, 0, 0, 0, 0, 0, 0]
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Felt252(Felt::MAX)
                *JitValue::Felt252(Felt::MAX)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<[u32; 8]>()
                    .cast::<[u32; 8]>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            // 0x800000000000011000000000000000000000000000000000000000000000001 - 1
            // 0x800000000000011000000000000000000000000000000000000000000000001 - 1
//            [0, 0, 0, 0, 0, 0, 17, 134217728]
            [0, 0, 0, 0, 0, 0, 17, 134217728]
//        );
        );
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Felt252(Felt::MAX + Felt::ONE)
                *JitValue::Felt252(Felt::MAX + Felt::ONE)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<[u32; 8]>()
                    .cast::<[u32; 8]>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            [0, 0, 0, 0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0, 0, 0, 0]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_uint8() {
    fn test_to_jit_uint8() {
//        let program = ProgramParser::new().parse("type u8 = u8;").unwrap();
        let program = ProgramParser::new().parse("type u8 = u8;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Uint8(9)
                *JitValue::Uint8(9)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<u8>()
                    .cast::<u8>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            9
            9
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_uint16() {
    fn test_to_jit_uint16() {
//        let program = ProgramParser::new().parse("type u16 = u16;").unwrap();
        let program = ProgramParser::new().parse("type u16 = u16;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Uint16(17)
                *JitValue::Uint16(17)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<u16>()
                    .cast::<u16>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            17
            17
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_uint32() {
    fn test_to_jit_uint32() {
//        let program = ProgramParser::new().parse("type u32 = u32;").unwrap();
        let program = ProgramParser::new().parse("type u32 = u32;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Uint32(33)
                *JitValue::Uint32(33)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<u32>()
                    .cast::<u32>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            33
            33
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_uint64() {
    fn test_to_jit_uint64() {
//        let program = ProgramParser::new().parse("type u64 = u64;").unwrap();
        let program = ProgramParser::new().parse("type u64 = u64;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Uint64(65)
                *JitValue::Uint64(65)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<u64>()
                    .cast::<u64>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            65
            65
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_uint128() {
    fn test_to_jit_uint128() {
//        let program = ProgramParser::new().parse("type u128 = u128;").unwrap();
        let program = ProgramParser::new().parse("type u128 = u128;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Uint128(129)
                *JitValue::Uint128(129)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<u128>()
                    .cast::<u128>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            129
            129
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_sint8() {
    fn test_to_jit_sint8() {
//        let program = ProgramParser::new().parse("type i8 = i8;").unwrap();
        let program = ProgramParser::new().parse("type i8 = i8;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Sint8(-9)
                *JitValue::Sint8(-9)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<i8>()
                    .cast::<i8>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            -9
            -9
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_sint16() {
    fn test_to_jit_sint16() {
//        let program = ProgramParser::new().parse("type i16 = i16;").unwrap();
        let program = ProgramParser::new().parse("type i16 = i16;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Sint16(-17)
                *JitValue::Sint16(-17)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<i16>()
                    .cast::<i16>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            -17
            -17
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_sint32() {
    fn test_to_jit_sint32() {
//        let program = ProgramParser::new().parse("type i32 = i32;").unwrap();
        let program = ProgramParser::new().parse("type i32 = i32;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Sint32(-33)
                *JitValue::Sint32(-33)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<i32>()
                    .cast::<i32>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            -33
            -33
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_sint64() {
    fn test_to_jit_sint64() {
//        let program = ProgramParser::new().parse("type i64 = i64;").unwrap();
        let program = ProgramParser::new().parse("type i64 = i64;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Sint64(-65)
                *JitValue::Sint64(-65)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<i64>()
                    .cast::<i64>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            -65
            -65
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_sint128() {
    fn test_to_jit_sint128() {
//        let program = ProgramParser::new().parse("type i128 = i128;").unwrap();
        let program = ProgramParser::new().parse("type i128 = i128;").unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::Sint128(-129)
                *JitValue::Sint128(-129)
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<i128>()
                    .cast::<i128>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            -129
            -129
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_ec_point() {
    fn test_to_jit_ec_point() {
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse("type EcPoint = EcPoint;")
            .parse("type EcPoint = EcPoint;")
//            .unwrap();
            .unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::EcPoint(Felt::from(1234), Felt::from(4321))
                *JitValue::EcPoint(Felt::from(1234), Felt::from(4321))
//                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                    .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                    .unwrap()
                    .unwrap()
//                    .cast::<[[u32; 8]; 2]>()
                    .cast::<[[u32; 8]; 2]>()
//                    .as_ptr()
                    .as_ptr()
//            },
            },
//            [[1234, 0, 0, 0, 0, 0, 0, 0], [4321, 0, 0, 0, 0, 0, 0, 0]]
            [[1234, 0, 0, 0, 0, 0, 0, 0], [4321, 0, 0, 0, 0, 0, 0, 0]]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_ec_state() {
    fn test_to_jit_ec_state() {
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse("type EcState = EcState;")
            .parse("type EcState = EcState;")
//            .unwrap();
            .unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        assert_eq!(
        assert_eq!(
//            unsafe {
            unsafe {
//                *JitValue::EcState(
                *JitValue::EcState(
//                    Felt::from(1234),
                    Felt::from(1234),
//                    Felt::from(4321),
                    Felt::from(4321),
//                    Felt::from(3333),
                    Felt::from(3333),
//                    Felt::from(4444),
                    Felt::from(4444),
//                )
                )
//                .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
                .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//                .unwrap()
                .unwrap()
//                .cast::<[[u32; 8]; 4]>()
                .cast::<[[u32; 8]; 4]>()
//                .as_ptr()
                .as_ptr()
//            },
            },
//            [
            [
//                [1234, 0, 0, 0, 0, 0, 0, 0],
                [1234, 0, 0, 0, 0, 0, 0, 0],
//                [4321, 0, 0, 0, 0, 0, 0, 0],
                [4321, 0, 0, 0, 0, 0, 0, 0],
//                [3333, 0, 0, 0, 0, 0, 0, 0],
                [3333, 0, 0, 0, 0, 0, 0, 0],
//                [4444, 0, 0, 0, 0, 0, 0, 0]
                [4444, 0, 0, 0, 0, 0, 0, 0]
//            ]
            ]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_enum() {
    fn test_to_jit_enum() {
//        // Parse the program
        // Parse the program
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse(
            .parse(
//                "type u8 = u8;
                "type u8 = u8;
//                type MyEnum = Enum<ut@MyEnum, u8, u8>;",
                type MyEnum = Enum<ut@MyEnum, u8, u8>;",
//            )
            )
//            .unwrap();
            .unwrap();
//

//        // Create the registry for the program
        // Create the registry for the program
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        // Call to_jit to get the value of the enum
        // Call to_jit to get the value of the enum
//        let result = JitValue::Enum {
        let result = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Uint8(10)),
            value: Box::new(JitValue::Uint8(10)),
//            debug_name: None,
            debug_name: None,
//        }
        }
//        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
//

//        // Assertion to verify that the value returned by to_jit is not NULL
        // Assertion to verify that the value returned by to_jit is not NULL
//        assert!(result.is_ok());
        assert!(result.is_ok());
//    }
    }
//

//    #[test]
    #[test]
//    #[should_panic(expected = "Variant index out of range.")]
    #[should_panic(expected = "Variant index out of range.")]
//    fn test_to_jit_enum_variant_out_of_range() {
    fn test_to_jit_enum_variant_out_of_range() {
//        // Parse the program
        // Parse the program
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse(
            .parse(
//                "type u8 = u8;
                "type u8 = u8;
//            type MyEnum = Enum<ut@MyEnum, u8, u8>;",
            type MyEnum = Enum<ut@MyEnum, u8, u8>;",
//            )
            )
//            .unwrap();
            .unwrap();
//

//        // Create the registry for the program
        // Create the registry for the program
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        // Call to_jit to get the value of the enum with tag value out of range
        // Call to_jit to get the value of the enum with tag value out of range
//        let _ = JitValue::Enum {
        let _ = JitValue::Enum {
//            tag: 2,
            tag: 2,
//            value: Box::new(JitValue::Uint8(10)),
            value: Box::new(JitValue::Uint8(10)),
//            debug_name: None,
            debug_name: None,
//        }
        }
//        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
//    }
    }
//

//    #[test]
    #[test]
//    #[should_panic(expected = "An enum without variants cannot be instantiated.")]
    #[should_panic(expected = "An enum without variants cannot be instantiated.")]
//    fn test_to_jit_enum_no_variant() {
    fn test_to_jit_enum_no_variant() {
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse(
            .parse(
//                "type u8 = u8;
                "type u8 = u8;
//                type MyEnum = Enum<ut@MyEnum, u8>;",
                type MyEnum = Enum<ut@MyEnum, u8>;",
//            )
            )
//            .unwrap();
            .unwrap();
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        let _ = JitValue::Enum {
        let _ = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Uint8(10)),
            value: Box::new(JitValue::Uint8(10)),
//            debug_name: None,
            debug_name: None,
//        }
        }
//        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
        .to_jit(&Bump::new(), &registry, &program.type_declarations[1].id);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_enum_type_error() {
    fn test_to_jit_enum_type_error() {
//        // Parse the program
        // Parse the program
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse(
            .parse(
//                "type felt252 = felt252;
                "type felt252 = felt252;
//                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
//            )
            )
//            .unwrap();
            .unwrap();
//

//        // Creating a registry for the program.
        // Creating a registry for the program.
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        // Invoking to_jit method on a JitValue::Enum to convert it to a JIT representation.
        // Invoking to_jit method on a JitValue::Enum to convert it to a JIT representation.
//        // Generating an error by providing an enum value instead of the expected type.
        // Generating an error by providing an enum value instead of the expected type.
//        let result = JitValue::Enum {
        let result = JitValue::Enum {
//            tag: 0,
            tag: 0,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![JitValue::from(2u32)],
                fields: vec![JitValue::from(2u32)],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        }
        }
//        .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
        .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//        .unwrap_err(); // Unwrapping the error
        .unwrap_err(); // Unwrapping the error
//

//        // Matching the error result to verify the error type and message.
        // Matching the error result to verify the error type and message.
//        match result {
        match result {
//            Error::UnexpectedValue(expected_msg) => {
            Error::UnexpectedValue(expected_msg) => {
//                // Asserting that the error message matches the expected message.
                // Asserting that the error message matches the expected message.
//                assert_eq!(
                assert_eq!(
//                    expected_msg,
                    expected_msg,
//                    format!(
                    format!(
//                        "expected value of type {:?} but got an enum value",
                        "expected value of type {:?} but got an enum value",
//                        program.type_declarations[0].id.debug_name
                        program.type_declarations[0].id.debug_name
//                    )
                    )
//                );
                );
//            }
            }
//            _ => panic!("Unexpected error type: {:?}", result),
            _ => panic!("Unexpected error type: {:?}", result),
//        }
        }
//    }
    }
//

//    #[test]
    #[test]
//    fn test_to_jit_struct_type_error() {
    fn test_to_jit_struct_type_error() {
//        // Parse the program
        // Parse the program
//        let program = ProgramParser::new()
        let program = ProgramParser::new()
//            .parse(
            .parse(
//                "type felt252 = felt252;
                "type felt252 = felt252;
//                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
//            )
            )
//            .unwrap();
            .unwrap();
//

//        // Creating a registry for the program.
        // Creating a registry for the program.
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        // Invoking to_jit method on a JitValue::Struct to convert it to a JIT representation.
        // Invoking to_jit method on a JitValue::Struct to convert it to a JIT representation.
//        // Generating an error by providing a struct value instead of the expected type.
        // Generating an error by providing a struct value instead of the expected type.
//        let result = JitValue::Struct {
        let result = JitValue::Struct {
//            fields: vec![JitValue::from(2u32)],
            fields: vec![JitValue::from(2u32)],
//            debug_name: None,
            debug_name: None,
//        }
        }
//        .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
        .to_jit(&Bump::new(), &registry, &program.type_declarations[0].id)
//        .unwrap_err(); // Unwrapping the error
        .unwrap_err(); // Unwrapping the error
//

//        // Matching the error result to verify the error type and message.
        // Matching the error result to verify the error type and message.
//        match result {
        match result {
//            Error::UnexpectedValue(expected_msg) => {
            Error::UnexpectedValue(expected_msg) => {
//                // Asserting that the error message matches the expected message.
                // Asserting that the error message matches the expected message.
//                assert_eq!(
                assert_eq!(
//                    expected_msg,
                    expected_msg,
//                    format!(
                    format!(
//                        "expected value of type {:?} but got a struct",
                        "expected value of type {:?} but got a struct",
//                        program.type_declarations[0].id.debug_name
                        program.type_declarations[0].id.debug_name
//                    )
                    )
//                );
                );
//            }
            }
//            _ => panic!("Unexpected error type: {:?}", result),
            _ => panic!("Unexpected error type: {:?}", result),
//        }
        }
//    }
    }
//}
}
//

//#[cfg(feature = "with-serde")]
#[cfg(feature = "with-serde")]
//mod range_serde {
mod range_serde {
//    use std::fmt;
    use std::fmt;
//

//    use cairo_lang_sierra::extensions::utils::Range;
    use cairo_lang_sierra::extensions::utils::Range;
//    use serde::{
    use serde::{
//        de::{self, Visitor},
        de::{self, Visitor},
//        ser::SerializeStruct,
        ser::SerializeStruct,
//        Deserializer, Serializer,
        Deserializer, Serializer,
//    };
    };
//

//    pub fn serialize<S>(range: &Range, ser: S) -> Result<S::Ok, S::Error>
    pub fn serialize<S>(range: &Range, ser: S) -> Result<S::Ok, S::Error>
//    where
    where
//        S: Serializer,
        S: Serializer,
//    {
    {
//        let mut state = ser.serialize_struct("Range", 2)?;
        let mut state = ser.serialize_struct("Range", 2)?;
//

//        state.serialize_field("lower", &range.lower)?;
        state.serialize_field("lower", &range.lower)?;
//        state.serialize_field("upper", &range.upper)?;
        state.serialize_field("upper", &range.upper)?;
//

//        state.end()
        state.end()
//    }
    }
//

//    pub fn deserialize<'de, D>(de: D) -> Result<Range, D::Error>
    pub fn deserialize<'de, D>(de: D) -> Result<Range, D::Error>
//    where
    where
//        D: Deserializer<'de>,
        D: Deserializer<'de>,
//    {
    {
//        struct RangeVisitor;
        struct RangeVisitor;
//

//        impl<'de> Visitor<'de> for RangeVisitor {
        impl<'de> Visitor<'de> for RangeVisitor {
//            type Value = Range;
            type Value = Range;
//

//            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//                formatter.write_str("an integer between -2^31 and 2^31")
                formatter.write_str("an integer between -2^31 and 2^31")
//            }
            }
//

//            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
//            where
            where
//                A: de::SeqAccess<'de>,
                A: de::SeqAccess<'de>,
//            {
            {
//                let lower = seq
                let lower = seq
//                    .next_element()?
                    .next_element()?
//                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
//                let upper = seq
                let upper = seq
//                    .next_element()?
                    .next_element()?
//                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
//

//                Ok(Range { lower, upper })
                Ok(Range { lower, upper })
//            }
            }
//

//            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
//            where
            where
//                A: de::MapAccess<'de>,
                A: de::MapAccess<'de>,
//            {
            {
//                let mut lower = None;
                let mut lower = None;
//                let mut upper = None;
                let mut upper = None;
//

//                while let Some((field, value)) = map.next_entry()? {
                while let Some((field, value)) = map.next_entry()? {
//                    match field {
                    match field {
//                        "lower" => {
                        "lower" => {
//                            lower = Some(value);
                            lower = Some(value);
//                        }
                        }
//                        "upper" => {
                        "upper" => {
//                            upper = Some(value);
                            upper = Some(value);
//                        }
                        }
//                        _ => return Err(de::Error::unknown_field(field, &["lower", "upper"])),
                        _ => return Err(de::Error::unknown_field(field, &["lower", "upper"])),
//                    }
                    }
//                }
                }
//

//                Ok(Range {
                Ok(Range {
//                    lower: lower.ok_or_else(|| de::Error::missing_field("lower"))?,
                    lower: lower.ok_or_else(|| de::Error::missing_field("lower"))?,
//                    upper: upper.ok_or_else(|| de::Error::missing_field("upper"))?,
                    upper: upper.ok_or_else(|| de::Error::missing_field("upper"))?,
//                })
                })
//            }
            }
//        }
        }
//

//        de.deserialize_struct("Range", &["lower", "upper"], RangeVisitor)
        de.deserialize_struct("Range", &["lower", "upper"], RangeVisitor)
//    }
    }
//}
}
