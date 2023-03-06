use llvm_mlir_sys::*;

#[derive(Debug)]
pub struct Registry {
    pub(crate) inner: MlirDialectRegistry,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            inner: unsafe { mlirDialectRegistryCreate() },
        }
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        unsafe {
            mlirDialectRegistryDestroy(self.inner)
        }
    }
}

pub struct Dialect {
    _inner: MlirDialect,
}
