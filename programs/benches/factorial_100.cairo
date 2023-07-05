fn factorial(value: felt252, n: felt252) -> felt252 {
    if (n == 1) {
        value
    } else {
        factorial(value * n, n - 1)
    }
}

#[test]
fn main() -> felt252 {
    factorial(1, 100)
}
