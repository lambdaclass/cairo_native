use dict::Felt252DictTrait;
use traits::Default;

fn main() -> (u32, u32, u32,
              u32, u32, u32,
              u32, u32, u32, u32) {
    let mut dict: Felt252Dict<u32> = Default::default();
    dict.insert(1, 1_u32);
    dict.insert(2, 2_u32);
    dict.insert(3, 3_u32);
    dict.insert(4, 4_u32);
    dict.insert(5, 5_u32);
    dict.insert(6, 6_u32);
    dict.insert(7, 7_u32);
    dict.insert(8, 8_u32);
    dict.insert(9, 9_u32);
    dict.insert(10, 10_u32);
    dict.insert(11, 11_u32);
    dict.insert(12, 12_u32);
    dict.insert(13, 13_u32);
    dict.insert(14, 14_u32);
    dict.insert(15, 15_u32);
    dict.insert(16, 16_u32);
    dict.insert(17, 17_u32);
    (
        dict.get(8), dict.get(9), dict.get(10),
        dict.get(11), dict.get(12), dict.get(13),
        dict.get(14), dict.get(15), dict.get(16),  dict.get(17),
    )
}
