use core::num::traits::Sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u8) -> u8 {
    value.sqrt()
}

fn run_test(value: felt252) -> u8 {
    program(value.try_into().unwrap())
}
