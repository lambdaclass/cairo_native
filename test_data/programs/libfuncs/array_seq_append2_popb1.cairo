use array::ArrayTrait;

fn run_test() -> Span<u32> {
    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    let mut data = data.span();
    let _ = data.pop_back();
    data
}
