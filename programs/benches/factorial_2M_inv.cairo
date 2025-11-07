fn factorial_inv(value: felt252, n: felt252) -> felt252 {
    if (n == 1) {
        value
    } else {
        factorial_inv(felt252_div(value, n.try_into().unwrap()), n - 1)
    }
}

fn main() {
    let result = factorial_inv(0x4d6e41de886ac83938da3456ccf1481182687989ead34d9d35236f0864575a0, 2_000_000);
    assert(
        result == 1,
        'invalid result'
    );
}
