use core::num::traits::Sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u64) -> u32 {
    value.sqrt()
}

fn run_test(value: felt252) -> u32 {
    program(value.try_into().unwrap())
}
