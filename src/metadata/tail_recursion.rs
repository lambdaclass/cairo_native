use melior::ir::{Block, BlockRef, Value, ValueLike};
use mlir_sys::{MlirBlock, MlirValue};

// TODO: Find a way to pass stuff with lifetimes while keeping the compiler happy.
#[derive(Debug)]
pub struct TailRecursionMeta {
    depth_counter: MlirValue,

    recursion_target: MlirBlock,
    return_target: Option<MlirBlock>,
}

impl TailRecursionMeta {
    pub fn new(depth_counter: Value, recursion_target: &Block) -> Self {
        Self {
            depth_counter: depth_counter.to_raw(),
            recursion_target: recursion_target.to_raw(),
            return_target: None,
        }
    }

    pub fn depth_counter<'ctx, 'this>(&self) -> Value<'ctx, 'this> {
        unsafe { Value::from_raw(self.depth_counter) }
    }

    pub fn recursion_target<'ctx, 'this>(&self) -> BlockRef<'ctx, 'this> {
        unsafe { BlockRef::from_raw(self.recursion_target) }
    }

    pub fn return_target<'ctx, 'this>(&self) -> Option<BlockRef<'ctx, 'this>> {
        self.return_target
            .map(|ptr| unsafe { BlockRef::from_raw(ptr) })
    }

    pub fn set_return_target(&mut self, block: &Block) {
        self.return_target = Some(block.to_raw());
    }
}
