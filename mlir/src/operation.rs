use std::{marker::PhantomData, ffi::CString};

use llvm_mlir_sys::{MlirOperation, MlirOperationState, mlirOperationStateGet, mlirStringRefCreateFromCString};

use crate::{location::Location, context::Context, llvm_string::LLVMString, attribute::{Attribute, NamedAttribute}};



pub struct Operation {
    inner: MlirOperation,
    name: LLVMString,
}

impl Operation {
    pub fn new() -> Self {
        todo!()
    }
}

pub struct OperationState<'ctx> {
    inner: MlirOperationState,
    name: LLVMString,
    _ctx: PhantomData<&'ctx Context>
}

impl<'ctx> OperationState<'ctx> {
    pub fn new(name: &str, loc: Location) -> Self {
        let name = LLVMString::from(name);

        let inner = unsafe {
            mlirOperationStateGet(name.inner, loc.inner)
        };
        
        Self {
            inner,
            name,
            _ctx: PhantomData
        }
    }

    pub fn add_attributes<'a>(&mut self, attributes: &[NamedAttribute<'ctx, 'a>]) {

    }
}
