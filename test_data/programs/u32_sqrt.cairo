use core::num::traits::Sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u32) -> u16 {
    value.sqrt()
}

fn run_test(value: felt252) -> u16 {
    program(value.try_into().unwrap())
}
