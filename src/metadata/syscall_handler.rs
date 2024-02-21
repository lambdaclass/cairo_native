//! # Syscall handler metadata

use crate::starknet::{handler::StarkNetSyscallHandlerCallbacks, StarkNetSyscallHandler};
use std::{alloc::Layout, fmt::Debug, ptr::NonNull};

/// Syscall handler metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyscallHandlerMeta {
    handler: NonNull<()>,
    layout: Layout,
}

impl SyscallHandlerMeta {
    /// Create the syscall handler metadata from a syscall handler.
    pub fn new<T>(handler_impl: &mut T) -> Self
    where
        T: StarkNetSyscallHandler,
    {
        let layout = Layout::new::<StarkNetSyscallHandlerCallbacks<T>>();
        let mut handler = unsafe {
            NonNull::new_unchecked(
                std::alloc::alloc(layout) as *mut StarkNetSyscallHandlerCallbacks<T>
            )
        };

        unsafe {
            *handler.as_mut() = StarkNetSyscallHandlerCallbacks::new(handler_impl);
        }

        Self {
            handler: handler.cast(),
            layout,
        }
    }

    /// Return a pointer to the syscall handler.
    pub fn as_ptr(&self) -> NonNull<()> {
        self.handler
    }
}

impl Drop for SyscallHandlerMeta {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.handler.as_mut() as *mut _ as *mut u8, self.layout);
        }
    }
}
