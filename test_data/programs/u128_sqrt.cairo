use core::integer::u128_sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u128) -> u128 {
    u128_sqrt(value)
}

fn run_test(value: felt252) -> u128 {
    program(value.try_into().unwrap())
}
