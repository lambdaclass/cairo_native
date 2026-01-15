use array::ArrayTrait;

fn run_test() -> Span<u32> {
    let mut numbers = ArrayTrait::new();
    numbers.append(1_u32);
    numbers.append(2_u32);
    numbers.append(3_u32);
    let mut numbers = numbers.span();
    let _ = numbers.pop_back();
    numbers
}