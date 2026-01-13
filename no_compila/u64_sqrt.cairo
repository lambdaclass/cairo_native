use core::integer::u64_sqrt;
use traits::TryInto;
use core::option::OptionTrait;

fn program(value: u64) -> u64 {
    u64_sqrt(value)
}

fn run_test(value: felt252) -> u64 {
    program(value.try_into().unwrap())
}
