fn main() -> felt252 {
    42
}

fn is_zero(x: felt252) -> felt252 {
    match x {
        0 => 1,
        _ => 0,
    }
}
