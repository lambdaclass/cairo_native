use traits::Default;
use dict::Felt252DictTrait;

fn run_test(key: felt252, val: felt252) -> felt252 {
    let mut dict: Felt252Dict<felt252> = Default::default();
    dict.insert(key, val);
    dict.get(key)
}
