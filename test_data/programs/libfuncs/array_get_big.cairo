use array::ArrayTrait;

fn run_test() -> (u32, u32, u32, u32) {
    let mut numbers = ArrayTrait::new();
    numbers.append(4_u32);
    numbers.append(3_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(2_u32);
    numbers.append(17_u32);
    numbers.append(17_u32);
    numbers.append(18_u32);
    numbers.append(19_u32);
    numbers.append(20_u32);
    numbers.append(21_u32);
    numbers.append(22_u32);
    numbers.append(23_u32);
    (
        *numbers.at(20),
        *numbers.at(21),
        *numbers.at(22),
        *numbers.at(23),
    )
}
