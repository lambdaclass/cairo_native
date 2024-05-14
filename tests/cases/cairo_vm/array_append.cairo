use array::ArrayTrait;

fn main() -> Array<u32> {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(2_u32);
    let _x = numbers.pop_front();
    numbers
}
