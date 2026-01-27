use array::ArrayTrait;

fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(3_u32);
    numbers.append(1_u32);
    let mut numbers = numbers.span();
    (
        numbers.pop_back(),
        numbers.pop_back(),
        numbers.pop_back(),
        numbers.pop_back(),
    )
}
