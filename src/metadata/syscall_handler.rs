use crate::starknet::{handler::StarkNetSyscallHandlerCallbacks, StarkNetSyscallHandler};
use std::{
    alloc::Layout,
    fmt::Debug,
    ptr::{addr_of, NonNull},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyscallHandlerMeta {
    handler: NonNull<()>,
    layout: Layout,
}

impl SyscallHandlerMeta {
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

    pub fn as_ptr(&self) -> NonNull<()> {
        // TODO: Check and explain why this is correct, its risks (when invoking the JIT engine
        //   manually, etc...).
        unsafe { NonNull::new_unchecked(addr_of!(self.handler) as *mut NonNull<()>) }.cast()
    }
}

impl Drop for SyscallHandlerMeta {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.handler.as_mut() as *mut _ as *mut u8, self.layout);
        }
    }
}
