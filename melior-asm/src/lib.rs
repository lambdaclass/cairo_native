#[doc(hidden)]
pub use melior_asm_proc::mlir_asm as mlir_asm_impl;
#[doc(hidden)]
pub use melior_next;

/// Parse MLIR IR and add its contents into the specified block.
#[macro_export]
macro_rules! mlir_asm {
    ( $target:expr $( , opt( $( $flag:literal ),* $(,)? ) )? => $( $inner:tt )* ) => {{
        use $crate::mlir_asm_impl;
        use melior_next::ir::{BlockRef, OperationRef};
        use std::ops::Deref;

        // Forward to the procedural macro.
        let parent_op = $target
            .parent_operation()
            .expect("Block should have a parent operation");
        let module = mlir_asm_impl!( &parent_op.context() $(opt($($flag),*))? => $( $inner )* );

        // Transfer its contents to the target block.
        fn push_recursive(m: BlockRef, op: OperationRef) {
            m.append_operation(op.deref().to_owned());
            if let Some(next_op) = op.next_in_block() {
                push_recursive(m, next_op);
            }
        }

        let module_body = module.body();
        push_recursive($target, module_body.first_operation().unwrap());
    }};
}
