use core::integer::u32_sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u32) -> u32 {
    u32_sqrt(value)
}

fn run_test(value: felt252) -> u32 {
    program(value.try_into().unwrap())
}
