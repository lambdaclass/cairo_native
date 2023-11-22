use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef};
use melior::ir::{Type, TypeLike};
use mlir_sys::MlirOperation;
use std::ffi::c_void;

extern "C" {
    fn LLVMStructType_getFieldTypeAt(ty_ptr: *const c_void, index: u32) -> *const c_void;

    /// Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context.
    /// This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.
    pub fn mlirTranslateModuleToLLVMIR(
        module_operation_ptr: MlirOperation,
        llvm_context: LLVMContextRef,
    ) -> LLVMModuleRef;
}

/// For any `!llvm.struct<...>` type, return the MLIR type of the field at the requested index.
pub fn get_struct_field_type_at<'c>(r#type: &Type<'c>, index: usize) -> Type<'c> {
    let mut ty_ptr = r#type.to_raw();

    ty_ptr.ptr = unsafe { LLVMStructType_getFieldTypeAt(ty_ptr.ptr, index as u32) };
    unsafe { Type::from_raw(ty_ptr) }
}
