use std::ffi::CString;

use llvm_mlir_sys::{mlirStringRefCreateFromCString, MlirStringRef};

#[derive(Debug)]
pub struct LLVMString {
    data: CString,
    pub(crate) inner: MlirStringRef,
}

impl From<&str> for LLVMString {
    fn from(value: &str) -> Self {
        let data = CString::new(value).unwrap();
        let inner = unsafe { mlirStringRefCreateFromCString(data.as_ptr()) };

        Self { data, inner }
    }
}

impl LLVMString {
    pub fn as_str(&self) -> &str {
        self.data.to_str().unwrap()
    }
}
