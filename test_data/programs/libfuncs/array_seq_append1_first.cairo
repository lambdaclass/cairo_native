use array::ArrayTrait;

fn run_test() -> u32 {
    let mut data = ArrayTrait::new();
    data.append(1);
    *data.at(0)
}