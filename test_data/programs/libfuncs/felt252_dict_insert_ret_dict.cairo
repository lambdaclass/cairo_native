use traits::Default;
use dict::Felt252DictTrait;

fn run_test() -> Felt252Dict<u32> {
    let mut dict: Felt252Dict<u32> = Default::default();
    dict.insert(1, 2_u32);
    dict.insert(2, 3_u32);
    dict.insert(3, 4_u32);
    dict.insert(4, 5_u32);
    dict.insert(5, 6_u32);
    dict
}
