use traits::TryInto;
use core::option::OptionTrait;

fn program(lhs: u32, rhs: u32) -> (u32, u32) {
    let q = lhs / rhs;
    let r = lhs % rhs;

    (q, r)
}

fn run_test(lhs: felt252, rhs: felt252) -> (u32, u32) {
    program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
}
