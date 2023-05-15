use dict::Felt252DictTrait;

fn main() -> u32 {
    let mut dict: Felt252Dict<u32> = Felt252DictTrait::new();
    dict.insert(4, 2_u32);
    let inserted = dict.get(4);
    inserted
}
