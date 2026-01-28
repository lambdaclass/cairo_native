use traits::TryInto;
use core::option::OptionTrait;

fn program(lhs: u16, rhs: u16) -> (u16, u16) {
    let q = lhs / rhs;
    let r = lhs % rhs;

    (q, r)
}

fn run_test(lhs: felt252, rhs: felt252) -> (u16, u16) {
    program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
}
