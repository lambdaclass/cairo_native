use array::ArrayTrait;

fn felt_to_bool(x: felt252) -> bool {
    x == 1
}

fn run_test(a: felt252) -> bool {
    felt_to_bool(a)
}
