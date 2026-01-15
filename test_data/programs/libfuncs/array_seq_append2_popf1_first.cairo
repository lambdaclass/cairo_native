use array::ArrayTrait;

fn run_test() -> u32 {
    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(2);
    let _ = data.pop_front();
    *data.at(0)
}