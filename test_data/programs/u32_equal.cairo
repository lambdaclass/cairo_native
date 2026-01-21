use traits::TryInto;
use core::option::OptionTrait;

fn program(lhs: u32, rhs: u32) -> bool {
    lhs == rhs
}

fn run_test(lhs: felt252, rhs: felt252) -> bool {
    program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
}
