use traits::TryInto;
use core::option::OptionTrait;

fn program(lhs: u128, rhs: u128) -> (u128, u128) {
    let q = lhs / rhs;
    let r = lhs % rhs;

    (q, r)
}

fn run_test(lhs: felt252, rhs: felt252) -> (u128, u128) {
    program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
}
