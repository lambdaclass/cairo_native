use dict::Felt252DictTrait;

fn main() -> felt252 {
    let mut dict_u8 = felt252_dict_new::<u8>();
    let mut dict_felt = felt252_dict_new::<felt252>();
    let _dict_felt2 = felt252_dict_new::<felt252>();

    dict_u8.insert(10, 110);
    dict_u8.insert(10, 110);

    let _val10 = dict_u8[10]; // 110
    let _val11 = dict_felt[11]; // 0
    dict_felt.insert(11, 1024);
    dict_felt[11] // 1024
}
