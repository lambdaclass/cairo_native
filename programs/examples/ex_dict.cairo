use traits::Default;
use dict::Felt252DictTrait;

fn main() -> u32 {
    let mut dict: Felt252Dict<u32> = Default::default();
    dict.insert(2, 1_u32);
    dict.get(2)
}
