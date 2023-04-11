use array::Array;
use array::ArrayTrait;

fn main() -> Array<u32> {
    let mut data: Array<u32> = ArrayTrait::new();
    data.append(1_u32);
    data.append(2_u32);
    data.append(3_u32);
    data.append(4_u32);
    data.append(5_u32);
    data
}
