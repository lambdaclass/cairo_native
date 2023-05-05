fn main() -> felt252 {
    42
}

fn is_zero(x: felt252) -> felt252 {
    match x {
        0 => 0,
        _ => 1,
    }
}
