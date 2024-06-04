#[contract]
mod DictEntryInitTest{

    use dict::{Felt252DictTrait, Felt252DictEntryTrait};
    use traits::Index;
    use array::{ArrayTrait, SpanTrait};

    #[external]
    fn felt252_dict_entry_init() {
        let mut dict: Felt252Dict<felt252> = Felt252DictTrait::new();

        // Generates hint Felt252DictEntryInit by creating a new dict entry
        dict.insert(10, 110);
        dict.insert(11, 111);
        let val10 = dict.index(10);
        let val11 = dict.index(11);
        assert(val10 == 110, 'dict[10] == 110');
        assert(val11 == 111, 'dict[11] == 111');

        dict.squash();

  }
}
