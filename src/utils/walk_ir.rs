use std::ffi::c_void;

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
