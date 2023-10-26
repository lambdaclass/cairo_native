fn factorial(value: felt252, n: felt252) -> felt252 {
    if (n == 1) {
        value
    } else {
        factorial(value * n, n - 1)
    }
}

#[test]
fn main() -> felt252 {
    // Make sure that factorial(10) == 3628800
    let y: felt252 = factorial(1, 10);
    assert(3628800 == y, 'failed test');
    factorial(1, 2000000)
}
