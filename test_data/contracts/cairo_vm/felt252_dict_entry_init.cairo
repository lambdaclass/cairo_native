#[starknet::contract]
mod DictEntryInitTest {
    use dict::{Felt252DictTrait, Felt252DictEntryTrait};
    use core::ops::index::Index;
    use array::{ArrayTrait, SpanTrait};

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn felt252_dict_entry_init(self: @ContractState) {
        let mut dict: Felt252Dict<felt252> = Default::default();

        // Generates hint Felt252DictEntryInit by creating a new dict entry
        dict.insert(10, 110);
        dict.insert(11, 111);
        let val10 = dict[10];
        let val11 = dict[11];
        assert(val10 == 110, 'dict[10] == 110');
        assert(val11 == 111, 'dict[11] == 111');

        dict.squash();
    }
}
