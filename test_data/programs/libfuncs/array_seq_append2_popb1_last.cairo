use array::ArrayTrait;

fn run_test() -> u32 {
    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    let mut data_span = data.span();
    let _ = data_span.pop_back();
    let last = data_span.len() - 1;
    *data_span.at(last)
}
