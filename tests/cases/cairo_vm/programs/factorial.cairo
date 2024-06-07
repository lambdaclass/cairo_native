use core::felt252;

fn main() -> felt252 {
    let n = 10;
    let result = factorial(n);
    result
}

fn factorial(n: felt252) -> felt252 {
    match n {
        0 => 1,
        _ => n * factorial(n - 1),
    }
}
