use array::ArrayTrait;
use traits::TryInto;
use core::option::OptionTrait;

fn run_test(value: felt252, idx: felt252) -> felt252 {
    let mut numbers: Array<felt252> = ArrayTrait::new();

    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    numbers.append(value);
    *numbers.at(idx.try_into().unwrap())
}
