fn factorial(value: felt252, n: felt252) -> felt252 {
    if (n == 1) {
        value
    } else {
        factorial(value * n, n - 1)
    }
}

fn run_test(n: felt252) -> felt252 {
    factorial(1, n)
}
