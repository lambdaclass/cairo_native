#[starknet::contract]
mod Felt252Dict {
    use dict::{felt252_dict_entry_finalize, Felt252DictTrait};

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn update_dict(self: @ContractState) -> (felt252, felt252) {
        let mut dict = felt252_dict_new::<felt252>();
        dict.insert(1, 64);
        let val = dict.get(1);
        dict.insert(1, 75);
        let val2 = dict.get(1);
        dict.squash();
        return (val, val2);
    }
}
