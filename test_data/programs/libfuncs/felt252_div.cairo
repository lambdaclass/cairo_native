fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
    felt252_div(lhs, rhs.try_into().unwrap())
}