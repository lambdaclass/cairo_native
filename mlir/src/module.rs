use llvm_mlir_sys::*;

use crate::location::Location;

#[derive(Debug)]
pub struct Module {
    pub(crate) inner: MlirModule,
}

impl Module {
    pub fn new(location: Location) -> Self {
        Self {
            inner: unsafe { mlirModuleCreateEmpty(location.inner) },
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            mlirModuleDestroy(self.inner);
        }
    }
}
