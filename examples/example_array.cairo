use array::Array;
use array::ArrayTrait;

fn main() -> Array<u64> {
    let mut data: Array<u64> = ArrayTrait::new();
    data.append(4_u64);
    data
}
