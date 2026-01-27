#[starknet::contract]
mod Felt252Dict {
    use dict::{felt252_dict_entry_finalize, Felt252DictTrait};

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn update_dict(self: @ContractState) {
        let mut dict = felt252_dict_new::<felt252>();
        dict.insert(1, 64);
        dict.insert(2, 75);
        dict.insert(3, 75);
        dict.squash();
    }
}
