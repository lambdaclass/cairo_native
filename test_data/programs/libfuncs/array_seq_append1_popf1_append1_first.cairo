use array::ArrayTrait;

fn run_test() -> u32 {
    let mut data = ArrayTrait::new();
    data.append(1);
    let _ = data.pop_front();
    data.append(2);
    *data.at(0)
}