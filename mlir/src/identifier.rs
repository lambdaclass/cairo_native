use std::{marker::PhantomData};

use llvm_mlir_sys::{mlirIdentifierGet, MlirIdentifier};

use crate::{context::Context, llvm_string::LLVMString};

#[derive(Debug)]
pub struct Identifier<'ctx> {
    pub(crate) inner: MlirIdentifier,
    name: LLVMString,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Identifier<'ctx> {
    pub fn new(ctx: &'ctx Context, name: &str) -> Self {
        let name = LLVMString::from(name);

        let inner = unsafe {
            mlirIdentifierGet(ctx.inner, name.inner)
        };

        Self {
            inner,
            name,
            _ctx: PhantomData,
        }
    }

    pub fn as_str(&self) -> &str {
        self.name.as_str()
    }
}
