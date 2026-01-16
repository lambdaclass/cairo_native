use array::ArrayTrait;
use core::bool_to_felt252;
use traits::TryInto;
use core::option::OptionTrait;

fn felt_to_bool(x: felt252) -> bool {
    x.try_into().unwrap() == 1_u8
}

fn program(a: bool) -> felt252 {
    bool_to_felt252(a)
}

fn run_test(a: felt252) -> felt252 {
    program(felt_to_bool(a))
}
