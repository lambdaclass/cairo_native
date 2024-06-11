// This function is NOT considered tail recursive and will not be optimized
// because the state is not empty (it needs `i`).
fn inner(i: felt252) -> felt252 {
    match i {
        0 => 0,
        _ => i + inner(i - 1),
    }
}

fn main() -> felt252 {
    inner(100)
}
