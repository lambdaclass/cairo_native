use crate::starknet::{handler::StarkNetSyscallHandlerCallbacks, StarkNetSyscallHandler};
use std::{
    alloc::Layout,
    fmt::Debug,
    marker::PhantomData,
    ptr::{addr_of, NonNull},
};

pub struct SyscallHandlerMeta {
    handler: NonNull<()>,
    layout: Layout,
    // phantom: PhantomData<&'a ()>,
}

impl SyscallHandlerMeta {
    // impl<'a> SyscallHandlerMeta<'a> {
    pub fn new<T>(handler_impl: &T) -> Self
    where
        T: Debug + StarkNetSyscallHandler,
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
            // phantom: PhantomData,
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
