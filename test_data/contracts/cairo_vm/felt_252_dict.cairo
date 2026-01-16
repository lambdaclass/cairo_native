#[starknet::contract]
mod Felt252Dict {
    #[storage]
    struct Storage {}

    use dict::{felt252_dict_entry_finalize, Felt252DictTrait};

    /// An external method that requires the `segment_arena` builtin.
    #[external(v0)]
    fn squash_empty_dict(self: @ContractState) -> bool {
        let x = felt252_dict_new::<felt252>();
        x.squash();
        return true;
    }
}
