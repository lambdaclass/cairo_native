use core::integer::u8_sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u8) -> u8 {
    u8_sqrt(value)
}

fn run_test(value: felt252) -> u8 {
    program(value.try_into().unwrap())
}
