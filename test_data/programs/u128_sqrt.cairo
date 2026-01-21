use core::num::traits::Sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u128) -> u64 {
    value.sqrt()
}

fn run_test(value: felt252) -> u64 {
    program(value.try_into().unwrap())
}
