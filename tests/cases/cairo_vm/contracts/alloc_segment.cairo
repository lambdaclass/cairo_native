#[starknet::contract]
mod AllocSegment {
    use dict::{Felt252DictTrait, Felt252DictEntryTrait};
    use traits::Index;
    use array::{ArrayTrait, SpanTrait};

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn alloc_segment(self: @ContractState) {
        // generates hint AllocSegment for felt252 dict when compiled to casm
        let mut dict: Felt252Dict<felt252> = Default::default();
        dict.squash();

        // generates hint AllocSegment for array
        let mut arr: Array<felt252> = ArrayTrait::new();

        arr.append(10);
        assert(*arr[0] == 10, 'array[0] == 10');
    }
}

