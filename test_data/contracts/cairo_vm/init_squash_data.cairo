#[starknet::contract]
mod TestDict {
    #[storage]
    struct Storage {}

    use dict::Felt252DictTrait;
    use nullable::NullableTrait;
    use core::ops::index::Index;

    #[external(v0)]
    fn test_dict_init(self: @ContractState, test_value: felt252) -> felt252 {
        let mut dict: Felt252Dict<felt252> = Default::default();

        dict.insert(10, test_value);
        let (_entry, value) = dict.entry(10);
        assert(value == test_value, 'dict[10] == test_value');

        return test_value;
    }
}
