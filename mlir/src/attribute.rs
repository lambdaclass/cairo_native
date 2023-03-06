use std::marker::PhantomData;

use llvm_mlir_sys::{
    mlirNamedAttributeGet, mlirStringAttrGet, MlirAttribute, MlirIdentifier, MlirNamedAttribute,
};

use crate::{context::Context, identifier::Identifier, llvm_string::LLVMString};

pub struct Attribute<'ctx> {
    inner: MlirAttribute,
    name: LLVMString,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Attribute<'ctx> {
    pub fn new(ctx: &'ctx Context, name: &str) -> Self {
        let name = LLVMString::from(name);

        let inner = unsafe { mlirStringAttrGet(ctx.inner, name.inner) };

        Self {
            inner,
            name,
            _ctx: PhantomData,
        }
    }
}

pub struct NamedAttribute<'ctx, 'a> {
    inner: MlirNamedAttribute,
    name: &'a Identifier<'ctx>,
    attr: &'a Attribute<'ctx>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx, 'a> NamedAttribute<'ctx, 'a> {
    pub fn new(name: &'a Identifier<'ctx>, attr: &'a Attribute<'ctx>) -> Self {
        let inner = unsafe { mlirNamedAttributeGet(name.inner, attr.inner) };
        Self {
            inner,
            name,
            attr,
            _ctx: PhantomData,
        }
    }
}
