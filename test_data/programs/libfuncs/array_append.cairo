use array::ArrayTrait;

fn run_test() -> Array<u32> {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers
}
