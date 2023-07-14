use traits::Default;
use dict::Felt252DictTrait;

fn main() {
    let mut dict: Felt252Dict<u32> = Default::default();
    dict.insert(2, 1_u32);
}
