use std::marker::PhantomData;

use llvm_mlir_sys::{MlirLocation, *};

use crate::context::Context;

/// A MLIR location.
#[derive(Debug)]
pub struct Location<'ctx> {
    pub(crate) inner: MlirLocation,
    _ctx: PhantomData<&'ctx Context>
}

impl<'ctx> Location<'ctx> {
    /// Creates a location with unknown position owned by the given context.
    pub fn new(ctx: &'ctx Context) -> Self {
        Self {
            inner: unsafe { mlirLocationUnknownGet(ctx.inner) },
            _ctx: PhantomData,
        }
    }
}
