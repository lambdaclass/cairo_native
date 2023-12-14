pub use self::{aot::AotNativeExecutor, jit::JitNativeExecutor};
use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};
use libc::c_void;
use std::arch::global_asm;

mod aot;
mod jit;

#[cfg(target_arch = "aarch64")]
global_asm!(include_str!("arch/aarch64.s"));
#[cfg(target_arch = "x86_64")]
global_asm!(include_str!("arch/x86_64.s"));

extern "C" {
    /// Invoke an AOT-compiled function.
    ///
    /// The `ret_ptr` argument is only used when the first argument (the actual return pointer) is
    /// unused. Used for u8, u16, u32, u64, u128 and felt252, but not for arrays, enums or structs.
    fn aot_trampoline(
        fn_ptr: *mut c_void,
        args_ptr: *const u64,
        args_len: usize,
        ret_ptr: &mut [u64; 4],
    );
}

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
