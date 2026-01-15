use array::ArrayTrait;

fn run_test() -> u32 {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(3_u32);
    match numbers.pop_front_consume() {
        Option::Some((_, x)) => x,
        Option::None(()) => 0_u32,
    }
}