
fn something(a: felt252) -> (felt252, felt252) {
    (a + 2, a - 2)
}

fn main() -> (felt252, felt252) {
    something(4)
}
