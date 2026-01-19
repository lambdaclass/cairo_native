#[starknet::contract]
mod SegmentArenaIndex {
    #[storage]
    struct Storage {}

    use dict::Felt252DictTrait;

    #[external(v0)]
    fn test_arena_index(self: @ContractState) -> bool {
        let mut dict: Felt252Dict<felt252> = Default::default();
        dict.squash();
        return true;
    }
}
