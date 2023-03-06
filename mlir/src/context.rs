use llvm_mlir_sys::*;

use crate::dialects::Registry;

#[derive(Debug)]
pub struct Context {
    pub(crate) inner: MlirContext,
}

impl Context {
    /// Creates an MLIR context.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: unsafe { mlirContextCreate() },
        }
    }

    /// Append the contents of the given dialect registry to the registry associated with the context.
    #[inline]
    pub fn append_registry(&self, registry: &Registry) {
        unsafe {
            mlirContextAppendDialectRegistry(self.inner, registry.inner);
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            mlirContextDestroy(self.inner);
        }
    }
}

impl PartialEq for Context {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.inner, other.inner) }
    }
}

impl Eq for Context {}
