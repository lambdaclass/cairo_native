//! # Tail recursion information
//!
//! Whenever the compiler detects a direct tail-recursive function call it'll insert this metadata.
//! If the libfunc handler decides to use it by setting a return target, the compiler will insert
//! the required instructions to make it really tail-recursive.
//!
//! PLT: consider simply applying to all tail-calls. If the stack data can be recycled, just jump.
//! Directly recursive functions are detected by checking if the current statement is a function
//! call. Indirect recursion is not handled and generates normal recursive code.
//!
//! Next, the compiler check whether those direct recursive calls are in fact tail recursive.
//! Recursive function calls are tail recursive if nothing declared before the function call is used
//! after it. Due to the way Sierra works, this is as simple as checking whether the state is empty
//! after taking the variables sent to the function call from itself.
//!
//! When tail recursion is detected, a counter is added which counts the current recursion depth.
//! The counter being zero means that no recursion has taken place (either because the program
//! hasn't reached the recursion point yet, or because it has completely unwinded already).
//!
//! Every time the recursive function call is invoked, the libfunc builder should increment this
//! counter by one and jump to the recursion target block provided by the meta. This recursion
//! target is provided by the compiler and should point to the beginning of the function and have
//! the same arguments. The function call libfunc builder should then set the return target to a
//! block which jumps to the next libfunc (a standard libfunc builder terminator as in every other
//! libfunc).
//!
//! When the compiler generates the code for a return statement of a tail recursive function, it'll
//! first check whether the depth counter is zero or not. If zero, a normal return statement will be
//! generated since it'd mean our parent frame is a different function. However if the counter is
//! not zero, the counter is decremented by one and a jump to the return target is generated, which
//! will return control to the function call libfunc builder. The builder should then jump to the
//! next libfunc statement as if it were a normal function call.
//!
//! After calling the libfunc builder, the metadata is removed from storage to avoid collisions
//! later on.
//!
//! The same algorithm can be applied multiple times if there are multiple tail-recursive calls
//! within a function. The compiler should create a different depth counter for each recursive call
//! in the function.

use melior::ir::{Block, BlockRef, Value, ValueLike};
use mlir_sys::{MlirBlock, MlirValue};

/// The tail recursion metadata.
///
/// Check out [the module](self) for more information about how tail recursion works.
// TODO: Find a way to pass stuff with lifetimes while keeping the compiler happy.
#[derive(Debug)]
pub struct TailRecursionMeta {
    depth_counter: MlirValue,

    recursion_target: MlirBlock,
    return_target: Option<MlirBlock>,
}

impl TailRecursionMeta {
    /// Create the tail recursion meta.
    pub fn new(depth_counter: Value, recursion_target: &Block) -> Self {
        Self {
            depth_counter: depth_counter.to_raw(),
            recursion_target: recursion_target.to_raw(),
            return_target: None,
        }
    }

    /// Return the current depth counter value.
    pub fn depth_counter<'ctx, 'this>(&self) -> Value<'ctx, 'this> {
        unsafe { Value::from_raw(self.depth_counter) }
    }

    /// Return the recursion target block.
    pub fn recursion_target<'ctx, 'this>(&self) -> BlockRef<'ctx, 'this> {
        unsafe { BlockRef::from_raw(self.recursion_target) }
    }

    /// Return the return target block, if set.
    pub fn return_target<'ctx, 'this>(&self) -> Option<BlockRef<'ctx, 'this>> {
        self.return_target
            .map(|ptr| unsafe { BlockRef::from_raw(ptr) })
    }

    /// Set the return target block.
    pub fn set_return_target(&mut self, block: &Block) {
        self.return_target = Some(block.to_raw());
    }
}
// PLT: TODO: do a second round after checking usage.
// PLT: ACK
