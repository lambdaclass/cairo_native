pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};

mod aot;
mod jit;

pub enum NativeExecutor<'m> {
    Aot(AotNativeExecutor<CoreType, CoreLibfunc>),
    Jit(JitNativeExecutor<'m>),
}

impl<'m> From<AotNativeExecutor<CoreType, CoreLibfunc>> for NativeExecutor<'m> {
    fn from(value: AotNativeExecutor<CoreType, CoreLibfunc>) -> Self {
        Self::Aot(value)
    }
}

impl<'m> From<JitNativeExecutor<'m>> for NativeExecutor<'m> {
    fn from(value: JitNativeExecutor<'m>) -> Self {
        Self::Jit(value)
    }
}
