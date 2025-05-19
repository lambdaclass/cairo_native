use std::ffi::c_void;

use llvm_sys::{
    core::{
        LLVMGetFirstBasicBlock, LLVMGetFirstFunction, LLVMGetFirstInstruction,
        LLVMGetNextBasicBlock, LLVMGetNextFunction, LLVMGetNextInstruction,
    },
    prelude::{LLVMModuleRef, LLVMValueRef},
    LLVMBasicBlock, LLVMValue,
};
use melior::ir::OperationRef;
use mlir_sys::{MlirOperation, MlirWalkResult};

type OperationWalkCallback =
    unsafe extern "C" fn(MlirOperation, *mut ::std::os::raw::c_void) -> MlirWalkResult;

/// TODO: Can we receive a closure instead?
/// We may need to save a pointer to the closure
/// inside of the callback data.
pub fn walk_mlir_operations<T: Sized>(
    top_op: OperationRef,
    f: OperationWalkCallback,
    initial: T,
) -> T {
    let mut data = Box::new(initial);
    unsafe {
        mlir_sys::mlirOperationWalk(
            top_op.to_raw(),
            Some(f),
            data.as_mut() as *mut _ as *mut c_void,
            mlir_sys::MlirWalkOrder_MlirWalkPreOrder,
        );
    };
    *data
}

pub unsafe fn walk_llvm_instructions(llvm_module: LLVMModuleRef, mut f: impl FnMut(LLVMValueRef)) {
    let new_value = |function_ptr: *mut LLVMValue| {
        if function_ptr.is_null() {
            None
        } else {
            Some(function_ptr)
        }
    };
    let new_block = |function_ptr: *mut LLVMBasicBlock| {
        if function_ptr.is_null() {
            None
        } else {
            Some(function_ptr)
        }
    };

    let mut current_function = new_value(LLVMGetFirstFunction(llvm_module));
    while let Some(function) = current_function {
        let mut current_block = new_block(LLVMGetFirstBasicBlock(function));
        while let Some(block) = current_block {
            let mut current_instruction = new_value(LLVMGetFirstInstruction(block));
            while let Some(instruction) = current_instruction {
                f(instruction);

                current_instruction = new_value(LLVMGetNextInstruction(instruction));
            }
            current_block = new_block(LLVMGetNextBasicBlock(block));
        }
        current_function = new_value(LLVMGetNextFunction(function));
    }
}
