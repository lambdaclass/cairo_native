use traits::Default;
use dict::Felt252DictTrait;

fn run_test() -> u64 {
    let mut dict: Felt252Dict<u64> = Default::default();
    dict.insert(200000000, 4_u64);
    dict.get(200000000)
}