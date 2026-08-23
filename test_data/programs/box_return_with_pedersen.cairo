use core::pedersen::pedersen;

fn run_test(a: felt252, b: felt252) -> Box<felt252> {
    BoxTrait::new(pedersen(a, b))
}
