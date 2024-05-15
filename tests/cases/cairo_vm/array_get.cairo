use array::ArrayTrait;

fn main() -> u32 {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(3_u32);
    numbers.append(2_u32);
    numbers.append(1_u32);
    *numbers.at(1)
}
