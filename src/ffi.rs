#![allow(dead_code)]

use melior::ir::{r#type::MemRefType, Type, TypeLike};
use std::ffi::c_void;

extern "C" {
    // fn Type_getABIAlignment(mod_ptr: *const c_void, ty_ptr: *const c_void) -> u32;
    // fn Type_getPreferredAlignment(mod_ptr: *const c_void, ty_ptr: *const c_void) -> u32;
    // fn Type_getSize(mod_ptr: *const c_void, ty_ptr: *const c_void) -> u32;
    // fn Type_getSizeInBits(mod_ptr: *const c_void, ty_ptr: *const c_void) -> u32;

    fn LLVMPointerType_getElementType(ty_ptr: *const c_void) -> *const c_void;
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;
    fn MemRefType_getElementType(ty_ptr: *const c_void) -> *const c_void;
}

// pub fn get_abi_alignment(module: &Module, r#type: &Type) -> usize {
//     let mod_ptr = module.to_raw().ptr;
//     let ty_ptr = r#type.to_raw().ptr;

//     unsafe { Type_getABIAlignment(mod_ptr, ty_ptr) as usize }
// }

// pub fn get_preferred_alignment(module: &Module, r#type: &Type) -> usize {
//     let mod_ptr = module.to_raw().ptr;
//     let ty_ptr = r#type.to_raw().ptr;

//     unsafe { Type_getPreferredAlignment(mod_ptr, ty_ptr) as usize }
// }

// pub fn get_size(module: &Module, r#type: &Type) -> usize {
//     let mod_ptr = module.to_raw().ptr;
//     let ty_ptr = r#type.to_raw().ptr;

//     unsafe { Type_getSize(mod_ptr, ty_ptr) as usize }
// }

// pub fn get_size_in_bits(module: &Module, r#type: &Type) -> usize {
//     let mod_ptr = module.to_raw().ptr;
//     let ty_ptr = r#type.to_raw().ptr;

//     unsafe { Type_getSizeInBits(mod_ptr, ty_ptr) as usize }
// }

pub fn get_pointer_element_type<'c>(r#type: &Type<'c>) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMPointerType_getElementType(ty_ptr.ptr) };
    unsafe { Type::from_raw(ty_ptr) }
}

pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    unsafe { Type::from_raw(ty_ptr) }
}

pub fn get_memref_element_type<'c>(r#type: &MemRefType<'c>) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { MemRefType_getElementType(ty_ptr.ptr) };
    unsafe { Type::from_raw(ty_ptr) }
}
