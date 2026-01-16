fn fibonacci(n: felt252) -> felt252 {
    if (n == 0 || n == 1) {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}
