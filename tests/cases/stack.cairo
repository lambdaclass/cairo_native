use core::nullable::{NullableTrait};
use core::num::traits::Bounded;
use core::starknet::EthAddress;
use core::dict::{Felt252Dict, Felt252DictTrait};

#[derive(Destruct, Default)]
pub struct Stack {
    pub items: Felt252Dict<Nullable<u256>>,
    pub len: usize,
}

pub trait StackTrait {
    fn new() -> Stack;
    fn push(ref self: Stack, item: u256) -> Result<(), ()>;
    fn pop(ref self: Stack) -> Result<u256, ()>;
    fn peek_at(ref self: Stack, index: usize) -> Result<u256, ()>;
    fn len(self: @Stack) -> usize;
}

impl StackImpl of StackTrait {
    #[inline(always)]
    fn new() -> Stack {
        Default::default()
    }

    /// Pushes a new bytes32 word onto the stack.
    ///
    /// When pushing an item to the stack, we will compute
    /// an index which corresponds to the index in the dict the item will be stored at.
    /// The internal index is computed as follows:
    ///
    /// index = len(Stack_i) + i * STACK_SEGMENT_SIZE
    ///
    /// # Errors
    ///
    /// If the stack is full, returns with a StackOverflow error.
    #[inline(always)]
    fn push(ref self: Stack, item: u256) -> Result<(), ()> {
        let length = self.len();

        self.items.insert(length.into(), NullableTrait::new(item));
        self.len += 1;
        println!("stack_push top {:?}", self.peek_at(0).unwrap());
        Result::Ok(())
    }

    /// Pops the top item off the stack.
    ///
    /// # Errors
    ///
    /// If the stack is empty, returns with a StackOverflow error.
    #[inline(always)]
    fn pop(ref self: Stack) -> Result<u256, ()> {
        self.len -= 1;
        let item = self.items.get(self.len().into());
        println!("stack_pop top {:?}", item.deref());
        Result::Ok(item.deref())
    }

    #[inline(always)]
    fn peek_at(ref self: Stack, index: usize) -> Result<u256, ()> {
        let position = self.len() - 1 - index;
        let item = self.items.get(position.into());

        Result::Ok(item.deref())
    }

    #[inline(always)]
    fn len(self: @Stack) -> usize {
        *self.len
    }
}

pub fn main() -> u256 {
    let mut stack = StackImpl::new();

    stack.push(0).unwrap();
    stack.push(0).unwrap();
    stack.push(0).unwrap();
    stack.push(0).unwrap();
    stack.push(0).unwrap();
    stack.push(4).unwrap();
    stack.pop().unwrap()
}
