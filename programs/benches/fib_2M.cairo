fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
    match n {
        0 => a,
        _ => fib(b, a + b, n - 1),
    }
}

fn main() -> felt252 {
    let y: felt252 = fib(0, 1, 10);
    assert(55 == y, 'failed test'); // check fib is correct
    fib(0, 1, 2000000) // 2m
}
