use array::ArrayTrait;

fn run_test() -> u32 {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(3_u32);
    let _ = numbers.pop_front();
    numbers.append(1_u32);
    *numbers.at(0)
}