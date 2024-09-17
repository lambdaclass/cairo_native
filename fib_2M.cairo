fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
    match n {
        0 => a,
        _ => fib(b, a + b, n - 1),
    }
}

fn main() {
    let y: felt252 = fib(0, 1, 10);
    assert(55 == y, 'failed test'); // check fib is correct

    let result = fib(0, 1, 2000000);
    assert(
        result == 0x79495858064f7881b9eff3a923642b2990b5a4342da5470eb2251df58d9acfb,
        'invalid result'
    );
}
