use llvm_sys::{
    core::{
        LLVMCountParams, LLVMGetFirstBasicBlock, LLVMGetFirstFunction, LLVMGetFirstInstruction,
        LLVMGetNextBasicBlock, LLVMGetNextFunction, LLVMGetNextInstruction,
    },
    prelude::{LLVMModuleRef, LLVMValueRef},
    LLVMBasicBlock, LLVMValue,
};
use melior::ir::{BlockLike, BlockRef, OperationRef};

/// Traverses the given operation tree in preorder.
///
/// Calls `f` on each operation encountered.
pub fn walk_mlir_operations(top_op: OperationRef, f: &mut impl FnMut(OperationRef)) {
    f(top_op);

    for region in top_op.regions() {
        let mut next_block = region.first_block();

        while let Some(block) = next_block {
            if let Some(operation) = block.first_operation() {
                walk_mlir_block_operations(operation, f);
            }

            next_block = block.next_in_region();
        }
    }
}

/// Traverses all following operations in the current block
///
/// Calls `f` on each operation encountered.
///
/// NOTE: The lifetime of each operation is bound to the previous operation,
/// so the only way I found to comply with the borrow checker was to make the
/// function recursive. This convinces the compiler that the full operation
/// chain is in scope. This has been fixed in the latest melior release, but
/// updating the dependency requires us to update to LLVM 20.
pub fn walk_mlir_block_operations(operation: OperationRef, f: &mut impl FnMut(OperationRef)) {
    walk_mlir_operations(operation, f);

    if let Some(next_operation) = operation.next_in_block() {
        walk_mlir_block_operations(next_operation, f);
    }
}

/// Traverses from start block to end block (including) in preorder.
///
/// Calls `f` on each operation encountered.
pub fn walk_mlir_block(
    start_block: BlockRef,
    end_block: BlockRef,
    f: &mut impl FnMut(OperationRef),
) {
    let mut next_block = Some(start_block);

    while let Some(block) = next_block {
        if let Some(operation) = block.first_operation() {
            walk_mlir_block_operations(operation, f);
        }

        if block == end_block {
            return;
        }

        next_block = block.next_in_region();
    }
}

/// Traverses the whole LLVM Module, calling `f` on each instruction.
///
/// As this function receives a closure rather than a function, there is no need
/// to receive initial data, and can instead modify the captured environment.
pub unsafe fn walk_llvm_instructions(
    llvm_module: LLVMModuleRef,
    llvm_max_params: &mut u32,
    mut f: impl FnMut(LLVMValueRef),
) {
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
        let curr_params_len = LLVMCountParams(function);
        if curr_params_len > *llvm_max_params {
            *llvm_max_params = curr_params_len;
        }
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
