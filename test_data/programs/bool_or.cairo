use array::ArrayTrait;
use traits::TryInto;
use core::option::OptionTrait;

fn felt_to_bool(x: felt252) -> bool {
    x.try_into().unwrap() == 1_u8
}

fn program(a: bool, b: bool) -> bool {
    a || b
}

fn run_test(a: felt252, b: felt252) -> bool {
    program(felt_to_bool(a), felt_to_bool(b))
}
