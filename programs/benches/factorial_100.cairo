fn factorial(n: felt252) -> felt252 {
    if (n == 1) {
        n
    } else {
        n * factorial(n - 1)
    }
}

#[test]
fn main() -> felt252 {
    factorial(100)
}
