use traits::TryInto;
use core::option::OptionTrait;

fn program(lhs: u8, rhs: u8) -> (u8, u8) {
    let q = lhs / rhs;
    let r = lhs % rhs;

    (q, r)
}

fn run_test(lhs: felt252, rhs: felt252) -> (u8, u8) {
    program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
}
