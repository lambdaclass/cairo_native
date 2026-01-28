use array::Array;
use array::ArrayTrait;

fn main() -> Option<u32> {
    let mut data: Array<u32> = ArrayTrait::new();
    // This should return none
    data.pop_front()
}
