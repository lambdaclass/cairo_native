use array::Array;
use array::ArrayTrait;

fn main() -> u8 {
    let mut data: Array<u8> = ArrayTrait::new();
    data.append(4_u8);
    data.append(5_u8);
    data.append(4_u8);
    data.append(4_u8);
    data.append(4_u8);
    data.append(1_u8);
    data.append(1_u8);
    data.append(1_u8);
    data.append(2_u8);
    *data[0]
}
