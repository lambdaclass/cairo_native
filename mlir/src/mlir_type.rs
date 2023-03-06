use std::marker::PhantomData;

use llvm_mlir_sys::{
    mlirIntegerTypeGet, mlirIntegerTypeGetWidth, mlirIntegerTypeSignedGet,
    mlirIntegerTypeUnsignedGet, MlirType,
};

use crate::context::Context;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataType {
    SignedInt { bitwidth: u32 },
    Int { bitwidth: u32 },
    UnsignedInt { bitwidth: u32 },
}

impl DataType {
    pub const fn is_int(&self) -> bool {
        matches!(
            self,
            Self::SignedInt { bitwidth: _ }
                | Self::Int { bitwidth: _ }
                | Self::UnsignedInt { bitwidth: _ }
        )
    }
}

/// A MLIR type.
#[derive(Debug)]
pub struct Type<'ctx> {
    pub(crate) inner: MlirType,
    pub(crate) data_type: DataType,
    _phantom: PhantomData<&'ctx ()>,
}

impl<'ctx> Type<'ctx> {
    pub fn new(ctx: &'ctx Context, data_type: DataType) -> Self {
        let inner = match data_type {
            DataType::SignedInt { bitwidth } => unsafe {
                mlirIntegerTypeSignedGet(ctx.inner, bitwidth)
            },
            DataType::Int { bitwidth } => unsafe { mlirIntegerTypeGet(ctx.inner, bitwidth) },
            DataType::UnsignedInt { bitwidth } => unsafe {
                mlirIntegerTypeUnsignedGet(ctx.inner, bitwidth)
            },
        };

        Self {
            inner,
            data_type,
            _phantom: PhantomData,
        }
    }

    /// Gets the bit width of this type, if it is an integer type.
    pub fn get_width(&'ctx self) -> Option<u32> {
        if self.data_type.is_int() {
            Some(unsafe { mlirIntegerTypeGetWidth(self.inner) })
        } else {
            None
        }
    }
}
