use array::Array;
use array::ArrayTrait;

fn main() -> (u32, u32, u32, u32, u32) {
    let mut data: Array<u32> = ArrayTrait::new();
    data.append(1_u32);
    data.append(2_u32);
    data.append(3_u32);
    data.pop_front();
    data.append(7_u32);
    data.append(5_u32);
    data.append(data.len());
    (*data[0], *data[1], *data[2], *data[3], *data[4])
}
