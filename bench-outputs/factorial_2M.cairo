fn factorial(value: felt252, n: felt252) -> felt252 {
    if (n == 1) {
        value
    } else {
        factorial(value * n, n - 1)
    }
}

fn main() {
    // Make sure that factorial(10) == 3628800
    let y: felt252 = factorial(1, 10);
    assert(3628800 == y, 'failed test');

    let result = factorial(1, 2000000);
    assert(
        result == 0x4d6e41de886ac83938da3456ccf1481182687989ead34d9d35236f0864575a0,
        'invalid result'
    );
}
