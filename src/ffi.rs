#![allow(dead_code)]

use melior::ir::{r#type::MemRefType, Type, TypeLike};
use std::ffi::c_void;

extern "C" {
    fn LLVMPointerType_getElementType(ty_ptr: *const c_void) -> *const c_void;
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;
    fn MemRefType_getElementType(ty_ptr: *const c_void) -> *const c_void;
}

/// For any `!llvm.ptr<T>` type, return the MLIR type corresponding to `T`.
pub fn get_pointer_element_type<'c>(r#type: &Type<'c>) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMPointerType_getElementType(ty_ptr.ptr) };
    unsafe { Type::from_raw(ty_ptr) }
}

/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    unsafe { Type::from_raw(ty_ptr) }
}

/// For any `memref<T>` type (any type of memref, not just memrefs to scalars), return the MLIR type
/// corresponding to `T`.
pub fn get_memref_element_type<'c>(r#type: &MemRefType<'c>) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { MemRefType_getElementType(ty_ptr.ptr) };
    unsafe { Type::from_raw(ty_ptr) }
}
