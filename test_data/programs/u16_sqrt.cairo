use core::integer::u16_sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u16) -> u16 {
    u16_sqrt(value)
}

fn run_test(value: felt252) -> u16 {
    program(value.try_into().unwrap())
}
