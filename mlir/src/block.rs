use std::ptr;

use llvm_mlir_sys::{mlirBlockCreate, mlirBlockDestroy, MlirBlock};

use crate::{location::Location, mlir_type::Type};

#[derive(Debug)]
pub struct Block {
    pub(crate) inner: MlirBlock,
}

impl Block {
    /// Creates a new empty block with the given argument types (and their locations).
    pub fn new(args: Option<Vec<(Type, Location)>>) -> Self {
        let inner = args.filter(|args| !args.is_empty()).map_or_else(
            || unsafe { mlirBlockCreate(0, ptr::null(), ptr::null()) },
            |args| {
                let types: Vec<_> = args.iter().map(|x| x.0.inner).collect();
                let locs: Vec<_> = args.iter().map(|x| x.1.inner).collect();
                unsafe {
                    mlirBlockCreate(
                        args.len().try_into().unwrap(),
                        types.as_ptr(),
                        locs.as_ptr(),
                    )
                }
            },
        );

        Self { inner }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.inner) }
    }
}
