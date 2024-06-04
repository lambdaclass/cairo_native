#[contract]
mod TestDict {
    use dict::Felt252DictTrait;
    use nullable::NullableTrait;
    use traits::Index;

    #[external]
    fn test_dict_init(test_value: felt252) -> felt252 {
        let mut dict: Felt252Dict<felt252> = Felt252DictTrait::new();

        dict.insert(10, test_value);
        let (entry, value) = dict.entry(10);
        assert(value == test_value, 'dict[10] == test_value');

        return test_value;
    }
}
